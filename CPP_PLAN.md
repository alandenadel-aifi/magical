# R / C++ re-implementation — status & plan

Location: [`re-implementation/`](re-implementation/) (this doc lives at the repo root alongside [BENCHMARKS.md](BENCHMARKS.md), [PYTHON_PLAN.md](PYTHON_PLAN.md), and [TODO.md](TODO.md)).

Goal: drop-in replacement for the R API in [`R/MAGICAL_functions.R`](R/MAGICAL_functions.R) with a compiled C++ backend, without touching the original code.

Target to beat: **1870.942 s** elapsed on the demo dataset at `iteration_num = 1000` (baseline recorded in [BENCHMARKS.md](BENCHMARKS.md)).

---

## Current state (stages 1 + 2 complete)

| Piece | Status |
|---|---|
| Directory skeleton (`R/`, `src/`, `tests/`) | ✅ |
| R wrapper with same public API as original | ✅ delegates to original R code |
| `backend = "r"` path | ✅ works |
| `backend = "cpp"` path | ✅ implemented and benchmarked |
| Synthetic test fixture | ✅ [`tests/testthat/helper-synthetic.R`](re-implementation/tests/testthat/helper-synthetic.R) |
| API-contract tests | ✅ 4 passing |
| Initialization tests | ✅ 11 passing |
| Sampler unit tests (R-side) | ✅ 14 passing |
| Estimation edge-case tests | ✅ 6 passing |
| Output-file tests | ✅ 3 passing |
| r-vs-r cross-seed equivalence | ✅ 1 passing |
| cpp-vs-r equivalence | ✅ passing (correlation ≥ 0.90) |
| Rcpp toolchain build probe | ✅ passing |
| Full demo-data integration | ✅ benchmarked at 1000 iter on Linux |

**Test totals: 44 tests. 40 passing (+1 with `MAGICAL_TEST_CPP_BUILD=1` = 41), 3–4 skipped, 0 failing.**

Run: `Rscript re-implementation/tests/testthat.R`  
Opt-in flags: `MAGICAL_TEST_CPP_BUILD=1`, `MAGICAL_RUN_SLOW_TESTS=1`

### Linux benchmark (2026-07-24, Intel Xeon Platinum 8581C @ 2.10 GHz)

| Backend | BLAS | 1000-iter estimation (mean 2 seeds) | vs R |
|---|---|---:|---:|
| R (ref BLAS) | libRblas | 2832.9 s | 1.00× |
| **C++ / Rcpp (ref BLAS)** | libRblas | **1790.4 s** | **1.58×** |
| Python numpy (OpenBLAS Haswell) | scipy-openblas | 990.0 s | 2.86× |

C++ currently trails Python because both R and the Rcpp `.so` share R's ref BLAS.
Once R is switched to OpenBLAS (Path C in [TODO.md](TODO.md)), C++ is projected
to reach ~160–280 s (≥10× vs R), overtaking Python.

### 6-way statistical equivalence (Linux, 1000 iter)

All cross-language Jaccard values fall within the intra-language MCMC noise
floor (~0.90). C++ output is statistically indistinguishable from R and Python.
See [BENCHMARKS.md](BENCHMARKS.md) for the full 6×6 matrix.

---

## Blockers before stage 2 can start

~~1. gfortran runtime not installed~~ — **resolved 2026-07-23.**
~~2. Decision on statistical equivalence~~ — **confirmed: correlation ≥ 0.90.**

**No blockers remaining. Stage 2 is complete. Next: § "Extract the C++ core" in [TODO.md](TODO.md).**

---

## Design decisions locked in (from earlier conversation)

- **Not-a-package**: `Rcpp::sourceCpp()` at load time, no `DESCRIPTION`/`NAMESPACE`. Matches the original repo style.
- **Keep in R, do not port**: `Data_loading`, `Candidate_circuits_construction_*`, `MAGICAL_initialization`, `MAGICAL_circuits_output`. These are I/O and one-off setup — not the bottleneck.
- **Port to C++**: `MAGICAL_estimation` + the 5 samplers it calls in the hot loop.
- **RNG**: Armadillo's native RNG (via `arma::arma_rng`) — not R's `R::rnorm` etc. Higher throughput, break bit-parity with R, but posterior distributions match within tolerance.
- **Fixture strategy**: small synthetic fixture (P=20, G=10, M=5, S=6) for CI; demo dataset gated behind `MAGICAL_RUN_SLOW_TESTS=1`.

---

## Latent bug discovered (worth fixing during stage 2)

`R/MAGICAL_functions.R` line ~961: `iteration_seg = round(iteration_num/10)`. When `iteration_num < 6`, this becomes 0 and the progress-print `if (i %% iteration_seg == 0)` errors on `NaN`. Trivial fix: `iteration_seg = max(1L, round(iteration_num/10))`. The C++ port should embed the fix from the start.

---

## Stage 2 plan (the actual C++ port)

### Shared C++ core (also used by Python; see [PYTHON_PLAN.md](PYTHON_PLAN.md))

Rather than writing R-specific C++, put the numerical guts in **language-agnostic** headers/source so the same code powers both the R and Python backends:

```
src/
  magical_core.{cpp,hpp}   # pure C++/Armadillo, no Rcpp or pybind11
  magical_rcpp.cpp         # thin Rcpp adapter (unpacks R types, calls core)
  magical.cpp              # ← existing stub; replace with magical_rcpp.cpp
```

Core exposes:
- `TFA tf_activity_sample(...)` — samples `T_A`, `T_R`, `T_sample`
- `arma::mat tf_peak_binding_sample(...)`
- `pair<mat,mat> tf_peak_binary_binding_sample(...)`
- `arma::mat peak_gene_looping_sample(...)`
- `pair<mat,mat> peak_gene_binary_looping_sample(...)`
- `EstimationResult magical_estimation_core(...)` — the outer loop

### Port order (each step ships tests before moving on)

1. **`tf_activity_sample`** — smallest sampler, only Gaussian draws. Cross-check the sampler-unit test with `expect_equal(cor(r_out, cpp_out), 1, tolerance = 0.15)` style tests (loose because RNG streams differ).
2. **`tf_peak_binding_sample`** — same structure, no Bernoulli step.
3. **`peak_gene_looping_sample`** — analogous to (2).
4. **`tf_peak_binary_binding_sample`** — first sampler with the mixture-of-Gaussians Bernoulli step. Test that outputs are binary and respect the zero-prior mask.
5. **`peak_gene_binary_looping_sample`** — analogous to (4).
6. **`magical_estimation_core`** — outer loop composing all five, plus the `sigma_A_noise` / `sigma_R_noise` gamma updates. Test end-to-end correlation ≥ 0.90 (matches the existing skipped test in [`tests/testthat/test-equivalence.R`](re-implementation/tests/testthat/test-equivalence.R)).

### Wiring back to R

- `re-implementation/R/MAGICAL_functions.R` grows a `backend = "cpp"` branch that:
  1. Calls `magical_load_cpp()` (existing helper) to compile if needed.
  2. Extracts matrices from `loaded_data` / `Candidate_circuits` / `Initial_model` into plain dense/sparse types.
  3. Calls `magical_estimation_core_rcpp(...)`.
  4. Repackages the return list with the exact same `TF_Peak_Binding_prob` / `Peak_Gene_Looping_prob` / `Noise_parameters` names.
- The user's `tutorial.R` gains a one-line change: `backend = "cpp"` in the estimation call. Nothing else moves.

### Testing gates

After each sampler port, the test count that must be green:
- **Steps 1–5:** the 40 currently-passing tests + a new "cpp version of sampler N returns same shape / respects same masks as R version" test. Approximately +5 tests per step.
- **Step 6:** unskip the `cpp-vs-r` correlation test in [`tests/testthat/test-equivalence.R`](re-implementation/tests/testthat/test-equivalence.R) and unskip the demo-scale version in [`tests/testthat/test-demo-slow.R`](re-implementation/tests/testthat/test-demo-slow.R). Both must pass at correlation ≥ 0.90.

### Benchmark gate

Once step 6 is green, add a row to [BENCHMARKS.md](BENCHMARKS.md) from the same tutorial.R run with `backend = "cpp"`. Must beat 1870 s. If it doesn't, don't ship — profile first.

---

## Files to create / touch in stage 2

- **new** `src/magical_core.hpp` — declarations
- **new** `src/magical_core.cpp` — the 5 samplers + driver, no R/Python deps
- **rename/replace** [`src/magical.cpp`](re-implementation/src/magical.cpp) → `src/magical_rcpp.cpp` (Rcpp adapter only)
- **edit** [`R/MAGICAL_functions.R`](re-implementation/R/MAGICAL_functions.R) — implement `backend = "cpp"` branch
- **edit** [`tests/testthat/test-equivalence.R`](re-implementation/tests/testthat/test-equivalence.R) — unskip
- **edit** [`tests/testthat/test-demo-slow.R`](re-implementation/tests/testthat/test-demo-slow.R) — unskip
- **edit** [BENCHMARKS.md](BENCHMARKS.md) — add cpp row

---

## Pickup checklist

When resuming this work:

1. ~~Install gfortran~~ ✅ done.
2. Verify build probe: `MAGICAL_TEST_CPP_BUILD=1 Rscript re-implementation/tests/testthat.R` — the probe test must pass. (Currently: passing.)
3. Read [PYTHON_PLAN.md](PYTHON_PLAN.md) — the C++ core is shared. Coordinate on `magical_core.hpp` API before you write code.
4. Start on port step 1 (`tf_activity_sample`). Reference is [`R/MAGICAL_functions.R`](R/MAGICAL_functions.R) lines 560–598.
5. Keep the whole suite green after every commit: `Rscript re-implementation/tests/testthat.R` should always be 40+/40+ passing.
