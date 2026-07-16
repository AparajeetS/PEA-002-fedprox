# PEA-002 Audit Report: FedProx

## Executive finding

**Verdict: Supported with qualifications.**

The central qualitative result survived a scoped independent reproduction: in the bundled statistically heterogeneous `Synthetic(1,1)` task with 90% simulated stragglers, accepting partial client work materially outperformed dropping stragglers, and adding the paper's proximal penalty improved the endpoint further.

The exact legacy program could not be executed in the current environment because its pinned dependency, `tensorflow-gpu==1.10`, has no compatible distribution for current Python. The evidence below therefore comes from a documented clean-room NumPy implementation using the upstream data and published protocol—not from a bitwise rerun of TensorFlow 1.x.

## Claim under audit

The paper and repository make two related empirical claims:

1. FedProx is more robust than FedAvg under systems and statistical heterogeneity.
2. In highly heterogeneous settings, FedProx improves absolute test accuracy by 22% on average across the evaluated suite.

This audit directly tests the first claim on the most heterogeneous bundled synthetic condition. It cannot adjudicate the five-dataset average in the second claim because the four real-data experiments were outside this audit's compute and compatibility scope.

## Artifact assessment

### What is strong

- All four synthetic datasets used in the paper are included.
- The repository exposes the core algorithms, experiment scripts, hyperparameters, and plotting code.
- The paper clearly distinguishes dropped stragglers, retained partial work with μ=0, and the proximal term.
- FedAvg and FedProx share client selections and simulated straggler assignments in the released code.

### What has decayed

- `requirements.txt` pins `tensorflow-gpu==1.10`, released for an obsolete Python/CUDA ecosystem.
- A 2026 dry-run resolves no compatible `tensorflow-gpu==1.10` package; only a much newer placeholder version is visible.
- There is no container, lockfile, archived wheel set, or modern compatibility branch.
- Paper-era raw logs are not included, so the supplied plotting scripts cannot independently verify published values without rerunning training.

This is reproducibility debt, not evidence that the historical result is false.

## Protocol

### Upstream source

- Repository: `litian96/FedProx`
- Commit: `d2a4501f319f1594b732d88315c5ca1a72855f50`
- Dataset: bundled `Synthetic(1,1)`
- 30 clients; 9,600 train and 1,084 test examples

### Primary reproduction

- 200 communication rounds
- 10 uniformly sampled clients per round
- 20 local epochs
- Mini-batch size 10
- Learning rate 0.01
- 90% simulated stragglers
- Seed 0
- Three methods:
  - `fedavg_drop`: stragglers excluded from aggregation
  - `fedprox_mu0`: stragglers contribute partial work; μ=0
  - `fedprox_mu1`: stragglers contribute partial work; μ=1

### Independent-implementation differences

- NumPy softmax regression replaces TensorFlow 1.x operations.
- Glorot initialization follows the same family but is not bitwise identical to TensorFlow's RNG.
- Client selection and active-set selection follow the released seed-by-round scheme.
- Partial epoch counts use an independent deterministic draw from the paper's stated uniform range. The legacy code's exact random state is entangled with repeated in-place data shuffling, so a bitwise match is not claimed.

### Robustness run

A separate 3-seed, 200-round run changed only the local solver to deterministic full-batch updates and evaluated 0%, 50%, and 90% stragglers. This asks whether the qualitative conclusion depends on the mini-batch solver. It is explicitly a stress test, not a faithful reproduction.

## Results

### Primary: released mini-batch setting, 90% stragglers

| Method | Accuracy | Test loss | Accuracy change |
| --- | ---: | ---: | ---: |
| FedAvg, drop stragglers | 58.58% | 3.8465 | — |
| Retain partial work, μ=0 | 69.00% | 0.9028 | +10.42 pp |
| FedProx, μ=1 | 71.96% | 0.6382 | +13.38 pp vs FedAvg; +2.95 pp vs μ=0 |

The ordering matches the paper's mechanism-level story: partial work provides most of the gain in this synthetic run, while the proximal penalty supplies an additional improvement.

### Robustness: full-batch local solver, three seeds

| Stragglers | FedAvg | Partial, μ=0 | FedProx, μ=1 |
| ---: | ---: | ---: | ---: |
| 0% | 70.30 ± 1.12% | 70.30 ± 1.12% | 70.08 ± 1.41% |
| 50% | 62.88 ± 0.87% | 69.53 ± 1.03% | 69.22 ± 1.07% |
| 90% | 58.03 ± 1.31% | 68.82 ± 0.97% | 68.70 ± 1.15% |

Retaining partial work remains strongly beneficial after changing the local solver. The fixed μ=1 penalty is slightly worse than μ=0 under full-batch updates, showing that the benefit of a particular μ is solver- and tuning-dependent. This agrees with the authors' own warning that μ must be tuned by dataset and objective.

## Interpretation

The available evidence supports the core qualitative claim on the hardest bundled synthetic condition. It also sharpens the mechanism:

- The dominant gain here comes from retaining partial client updates instead of discarding 90% of selected clients.
- The proximal term adds a smaller but real improvement in the paper-like mini-batch run.
- A fixed μ=1 is not universally superior under a changed local solver, so “FedProx wins” should not be read as a parameter-free guarantee.

The observed +13.38-point primary improvement is not a failed reproduction of the paper's 22-point headline: the latter is an average over five different datasets, while this audit covers one synthetic dataset with an independent implementation.

## Recommendations

1. Add a paper-version tag and archive the exact TensorFlow/CUDA/Python environment in a container.
2. Publish the raw paper-era logs and machine-readable per-seed result tables.
3. Add a maintained implementation in a current framework with regression tests against archived curves.
4. Report the μ grid and selection rule next to every benchmark result.
5. Include a component table separating gains from partial-work retention and the proximal penalty.

## Maintainer response

No response recorded yet. Corrections and missing paper-era artifacts will be incorporated in a versioned revision.
