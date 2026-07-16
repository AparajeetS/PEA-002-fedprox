#!/usr/bin/env python3
"""PEA-002: independent, modern reproduction of the FedProx synthetic experiment.

This script intentionally has only a NumPy dependency. It consumes the official
repository's bundled Synthetic(1,1) JSON files and reproduces the three-way
comparison used by the paper:

* FedAvg: stragglers are dropped.
* FedProx, mu=0: partial work is retained, without a proximal penalty.
* FedProx, mu=1: partial work is retained, with the proximal penalty.

It is a clean-room numerical reproduction, not a bitwise port of TensorFlow 1.10.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = {
    "fedavg_drop": {"partial": False, "mu": 0.0},
    "fedprox_mu0": {"partial": True, "mu": 0.0},
    "fedprox_mu1": {"partial": True, "mu": 1.0},
}


@dataclass
class Client:
    user: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def weight(self) -> int:
        return int(self.y_train.size)


def read_clients(source: Path) -> list[Client]:
    root = source / "data" / "synthetic_1_1" / "data"
    with (root / "train" / "mytrain.json").open(encoding="utf-8") as f:
        train = json.load(f)
    with (root / "test" / "mytest.json").open(encoding="utf-8") as f:
        test = json.load(f)
    clients = []
    for user in sorted(train["user_data"]):
        tr, te = train["user_data"][user], test["user_data"][user]
        clients.append(
            Client(
                user,
                np.asarray(tr["x"], dtype=np.float64),
                np.asarray(tr["y"], dtype=np.int64),
                np.asarray(te["x"], dtype=np.float64),
                np.asarray(te["y"], dtype=np.int64),
            )
        )
    return clients


def init_model(seed: int, d: int = 60, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    # TensorFlow 1.x tf.layers.dense defaults to Glorot-uniform kernel + zero bias.
    rng = np.random.RandomState(123 + seed)
    limit = math.sqrt(6.0 / (d + k))
    return rng.uniform(-limit, limit, (d, k)), np.zeros(k, dtype=np.float64)


def fixed_permutation(n: int) -> np.ndarray:
    # The released helper resets NumPy to seed 100 for every epoch.
    rng = np.random.RandomState(100)
    return rng.permutation(n)


def local_sgd(
    client: Client,
    global_w: np.ndarray,
    global_b: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    w, b = global_w.copy(), global_b.copy()
    x, y = client.x_train, client.y_train
    order = np.arange(y.size)
    effective_batch = y.size if batch_size <= 0 else batch_size
    perm = fixed_permutation(y.size)
    for _ in range(epochs):
        order = order[perm]
        for start in range(0, y.size, effective_batch):
            idx = order[start : start + effective_batch]
            xb, yb = x[idx], y[idx]
            logits = xb @ w + b
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            prob[np.arange(yb.size), yb] -= 1.0
            prob /= yb.size
            grad_w = xb.T @ prob + mu * (w - global_w)
            grad_b = prob.sum(axis=0) + mu * (b - global_b)
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
    return w, b


def evaluate(clients: list[Client], w: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    correct = total = 0
    loss_sum = 0.0
    for c in clients:
        logits = c.x_test @ w + b
        correct += int((logits.argmax(axis=1) == c.y_test).sum())
        total += c.y_test.size
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss_sum -= float(log_probs[np.arange(c.y_test.size), c.y_test].sum())
    return correct / total, loss_sum / total


def selected_clients(round_id: int, n_clients: int, count: int) -> np.ndarray:
    return np.random.RandomState(round_id).choice(n_clients, count, replace=False)


def active_clients(round_id: int, selected: np.ndarray, drop: float) -> set[int]:
    count = int(round(len(selected) * (1.0 - drop)))
    if count == 0:
        return set()
    chosen = np.random.RandomState(round_id).choice(selected, count, replace=False)
    return {int(x) for x in chosen}


def run_one(
    clients: list[Client],
    method: str,
    drop: float,
    seed: int,
    rounds: int,
    eval_every: int,
    clients_per_round: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> list[dict[str, float | int | str]]:
    spec = METHODS[method]
    w, b = init_model(seed)
    curve: list[dict[str, float | int | str]] = []

    for round_id in range(rounds + 1):
        if round_id % eval_every == 0 or round_id == rounds:
            accuracy, loss = evaluate(clients, w, b)
            curve.append(
                {
                    "method": method,
                    "drop_fraction": drop,
                    "seed": seed,
                    "round": round_id,
                    "test_accuracy": accuracy,
                    "test_loss": loss,
                }
            )
        if round_id == rounds:
            break

        selected = selected_clients(round_id, len(clients), clients_per_round)
        active = active_clients(round_id, selected, drop)
        local_models: list[tuple[int, np.ndarray, np.ndarray]] = []
        for client_id in selected:
            cid = int(client_id)
            if cid in active:
                local_epochs = epochs
            elif spec["partial"]:
                # Independent deterministic draw from the paper's Uniform[1, E).
                local_epochs = int(np.random.RandomState(seed * 100000 + round_id * 100 + cid).randint(1, epochs))
            else:
                continue
            lw, lb = local_sgd(
                clients[cid], w, b, local_epochs, batch_size, learning_rate, float(spec["mu"])
            )
            local_models.append((clients[cid].weight, lw, lb))

        if not local_models:
            continue
        denom = float(sum(weight for weight, _, _ in local_models))
        w = sum(weight * lw for weight, lw, _ in local_models) / denom
        b = sum(weight * lb for weight, _, lb in local_models) / denom
    return curve


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Clone of litian96/FedProx")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--drops", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 uses deterministic full-batch updates; 10 matches the released scripts but is much slower",
    )
    args = parser.parse_args()

    clients = read_clients(args.source)
    rows: list[dict] = []
    started = time.time()
    total = len(METHODS) * len(args.drops) * len(args.seeds)
    done = 0
    for method in METHODS:
        for drop in args.drops:
            for seed in args.seeds:
                done += 1
                print(f"[{done:02d}/{total:02d}] {method} drop={drop:g} seed={seed}", flush=True)
                rows.extend(
                    run_one(
                        clients,
                        method,
                        drop,
                        seed,
                        args.rounds,
                        args.eval_every,
                        clients_per_round=10,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=0.01,
                    )
                )
            write_csv(args.output / "curves.csv", rows)

    finals = [r for r in rows if r["round"] == args.rounds]
    summary = []
    for method in METHODS:
        for drop in args.drops:
            subset = [r for r in finals if r["method"] == method and r["drop_fraction"] == drop]
            acc = np.asarray([r["test_accuracy"] for r in subset], dtype=float)
            loss = np.asarray([r["test_loss"] for r in subset], dtype=float)
            summary.append(
                {
                    "method": method,
                    "drop_fraction": drop,
                    "n_seeds": len(subset),
                    "accuracy_mean": float(acc.mean()),
                    "accuracy_sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                    "loss_mean": float(loss.mean()),
                    "loss_sd": float(loss.std(ddof=1)) if len(loss) > 1 else 0.0,
                }
            )
    write_csv(args.output / "summary.csv", summary)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source": str(args.source.resolve()),
        "source_repository": "https://github.com/litian96/FedProx",
        "source_commit": "d2a4501f319f1594b732d88315c5ca1a72855f50",
        "dataset": "synthetic_1_1",
        "rounds": args.rounds,
        "seeds": args.seeds,
        "drops": args.drops,
        "local_epochs": args.epochs,
        "batch_size": "full" if args.batch_size <= 0 else args.batch_size,
        "runtime_seconds": time.time() - started,
        "clients": len(clients),
        "train_samples": sum(c.y_train.size for c in clients),
        "test_samples": sum(c.y_test.size for c in clients),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
