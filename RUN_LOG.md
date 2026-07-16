# Run Log

Audit date: 2026-07-16  
Auditor: Aparajeet Shadangi, Phason Labs

## Source acquisition

```text
Repository: https://github.com/litian96/FedProx
Commit: d2a4501f319f1594b732d88315c5ca1a72855f50
Last upstream commit date: 2021-04-14
```

## Attempt 1 — exact dependency path

Command:

```text
python -m pip install --dry-run tensorflow-gpu==1.10
```

Result: failed before installation. Pip reported no matching distribution for `tensorflow-gpu==1.10` in the current Python 3.14 environment.

Interpretation: the README's direct installation path is no longer executable on a current Python stack. No dependency was installed or changed by this dry run.

## Attempt 2 — clean-room implementation smoke test

The NumPy implementation was compiled and exercised for two rounds across all three methods and two straggler settings. It completed successfully and verified that FedAvg and μ=0 are numerically identical when no clients are partial.

## Attempt 3 — initial full mini-batch matrix

An initial plan attempted 27 full mini-batch runs (3 methods × 3 straggler rates × 3 seeds × 200 rounds). It was stopped before the first checkpoint because the faithful nested mini-batch loop was unnecessarily expensive for the audit question. No partial results from that stopped run are included.

## Attempt 4 — three-seed solver robustness matrix

Command:

```text
python audit.py --source C:\tmp\phason-fedprox-source --output results \
  --rounds 200 --eval-every 10 --seeds 0 1 2 --drops 0 0.5 0.9 \
  --epochs 20 --batch-size 0
```

Result: completed 27/27 conditions in 203.6 seconds. Outputs are in `results/`.

## Attempt 5 — short released-solver trajectory check

Command:

```text
python audit.py --source C:\tmp\phason-fedprox-source --output results_minibatch25 \
  --rounds 25 --eval-every 5 --seeds 0 --drops 0 0.5 0.9 \
  --epochs 20 --batch-size 10
```

Result: completed 9/9 conditions in 46.4 seconds. Outputs are in `results_minibatch25/`. This run is retained as an early-trajectory diagnostic, not used for the final endpoint verdict.

## Attempt 6 — primary paper-like endpoint

Command:

```text
python audit.py --source C:\tmp\phason-fedprox-source --output results_minibatch200_drop90 \
  --rounds 200 --eval-every 10 --seeds 0 --drops 0.9 \
  --epochs 20 --batch-size 10
```

Result: completed 3/3 conditions in 78.9 seconds. Outputs are in `results_minibatch200_drop90/` and form the primary evidence for the verdict.

## Runtime

```text
Python: 3.14.3
NumPy: 2.4.3
OS: Windows 11
Hardware: local CPU
```
