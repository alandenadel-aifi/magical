# Performance targets

Baseline established by running `tutorial.R` with the shipped demo dataset at
`iteration_num = 1000`, single chain, R backend (original code, unmodified).

## Baseline: original R implementation

| Machine | Date       | Iterations | user (s)  | system (s) | elapsed (s) | elapsed (min) |
|---------|------------|-----------:|----------:|-----------:|------------:|--------------:|
| M-series MacBook (alan.denadel) | 2026-07-22 |       1000 |  1702.927 |    161.600 |    1870.942 |         31.18 |

Reproduction:
```
Rscript tutorial.R
```
The `proc.time()` timing block is around lines 45–50 of `tutorial.R`.

## Python numpy port vs R original (stage P1 complete, 2026-07-23)

Same machine, same demo dataset, `iteration_num = 1000`, single chain, two seeds
each. R uses [`re-implementation/scripts/gen_demo_parity_snapshots.R`](re-implementation/scripts/gen_demo_parity_snapshots.R);
Python uses [`re-implementation/scripts/save_py_snapshots.py`](re-implementation/scripts/save_py_snapshots.py).

| Backend | Seed       | Estimation (s) | End-to-end (s) | End-to-end (mm:ss) |
|---------|-----------:|---------------:|---------------:|-------------------:|
| R       | 20260723   |        1874.80 |        1944    |             32:24  |
| R       | 42         |        1893.40 |        1962    |             32:42  |
| Py numpy| 20260723   |         621.30 |         648.90 |             10:49  |
| Py numpy| 42         |         617.19 |         644.75 |             10:45  |

**Speedup (mean of the two seeds):**

|                              | R    | Python | Speedup |
|------------------------------|-----:|-------:|--------:|
| Estimation only              | 1884.1 s | 619.2 s | **3.04×** |
| End-to-end (load + circuits + init + estimation) | 1953 s | 646.8 s | **3.02×** |

Both implementations are single-threaded (≈100 % of one core). The end-to-end
gap tracks the estimation gap almost exactly — data-loading, candidate-circuit
construction and initialization together account for <5 % of R's runtime and
run in ~28 s in Python (a mix of faster and slightly slower per-stage than R,
but negligible next to the sampler loop).

### Statistical equivalence at n_iter = 1000

All 4-choose-2 pairwise comparisons across the 4 runs above. Circuit-Jaccard
uses the paper's selection thresholds (B ≥ 0.8, L ≥ 0.95). B / L posterior
correlations are Pearson r over the full matrices.

|                     | R@20260723 | R@42  | Py@20260723 | Py@42 |
|---------------------|:----------:|:-----:|:-----------:|:-----:|
| **R@20260723**      | 1.000      | 0.901 | 0.904       | 0.907 |
| **R@42**            | 0.901      | 1.000 | 0.907       | **0.919** |
| **Py@20260723**     | 0.904      | 0.907 | 1.000       | 0.908 |
| **Py@42**           | 0.907      | 0.919 | 0.908       | 1.000 |

| Comparison                | Triple Jaccard | B posterior r | L posterior r |
|---------------------------|:--------------:|:-------------:|:-------------:|
| R ↔ R  (intra, MCMC noise)  | 0.9012 | 0.99966 | 0.99988 |
| Py ↔ Py (intra, MCMC noise) | 0.9076 | 0.99966 | 0.99990 |
| R ↔ Py (cross, mean of 4)   | **0.9090** | 0.99966 | 0.99990 |

**Δ = cross − mean(intra) = +0.005.** Cross-language agreement is
indistinguishable from intra-language MCMC sampler variance; the single highest
Jaccard in the matrix (0.919) is a cross-language pair (R@42 ↔ Py@42). TF sets
are identical in every pair (Jaccard = 1.000).

Reproduce with [`re-implementation/scripts/plot_variability.py`](re-implementation/scripts/plot_variability.py);
plots are in [`benchmarks/plots/variability_*.png`](re-implementation/benchmarks/plots/) and
per-pair metrics in [`benchmarks/variability_metrics.tsv`](re-implementation/benchmarks/variability_metrics.tsv).

## Target for the Rcpp/Armadillo re-implementation

The re-implementation lives under `re-implementation/` and must **beat 619 s
elapsed** (Python numpy port, estimation only, mean of 2 seeds) on the same
machine, same demo dataset, same `iteration_num = 1000`, using
`MAGICAL_estimation(..., backend = "cpp")`. The soft target relative to the
original R baseline is unchanged: ≥ 10× on estimation.

## Future benchmark entries

Add rows to the tables above as backends land. Include commit hash and any
non-default flags (e.g., number of parallel chains) so entries are
reproducible.
