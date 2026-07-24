# Python implementation — status & plan

Location: [`src/magical/`](src/magical/) with tests in [`tests/`](tests/) and packaging in [`pyproject.toml`](pyproject.toml). This plan doc lives at the repo root alongside [BENCHMARKS.md](BENCHMARKS.md), [CPP_PLAN.md](CPP_PLAN.md), and [TODO.md](TODO.md).

Goal: two Python backends with the same public API —
1. **Raw numpy port** of the R code (reference, no C++ dependency).
2. **pybind11 bindings** onto the shared C++ core built by the R re-implementation (see [CPP_PLAN.md](CPP_PLAN.md)).

Both must be numerically consistent with the R original (statistical equivalence, correlation ≥ 0.90 on pooled posteriors).

---

## Current state (stage P1 complete)

| Piece | Status |
|---|---|
| `uv init --lib --name magical` project layout | ✅ done |
| [`pyproject.toml`](pyproject.toml) with numpy, scipy, pandas, pytest | ✅ done |
| [`src/magical/__init__.py`](src/magical/__init__.py) exporting the public API | ✅ done |
| [`src/magical/types.py`](src/magical/types.py) — dataclasses | ✅ done |
| [`src/magical/_rng.py`](src/magical/_rng.py) — truncated normal + inv-gamma helpers | ✅ done |
| [`src/magical/initialization.py`](src/magical/initialization.py) — numpy `magical_initialization` | ✅ done, **bit-for-bit equal to R** |
| [`src/magical/samplers.py`](src/magical/samplers.py) — all 5 Gibbs samplers | ✅ done |
| [`src/magical/estimation.py`](src/magical/estimation.py) — numpy driver loop, latent-bug fix included | ✅ done |
| [`src/magical/output.py`](src/magical/output.py) — `MAGICAL_circuits_output` port | ✅ done |
| [`src/magical/api.py`](src/magical/api.py) — `estimation(..., backend={"numpy","cpp"})` facade | ✅ done (`cpp` stubs `NotImplementedError`) |
| [`src/magical/io.py`](src/magical/io.py) — `Data_loading` port (real demo files) | ✅ done |
| [`src/magical/circuits.py`](src/magical/circuits.py) — `Candidate_circuits_construction_{with,without}_TAD` | ✅ done |
| `tests/` mirroring R suite | ✅ done (10 test files) |
| Synthetic fixture parity with R helper | ✅ done ([tests/conftest.py](tests/conftest.py)) |
| **Cross-language parity tests (numpy vs R)** | ✅ **done** — synthetic fixture at 50 iters, real demo dataset at 1000 iters |
| pybind11 bindings | ❌ not started (stage P2) |

**Test totals: 72 tests. 57 passing, 15 skipped (12 demo-parity + 3 stage-P2), 0 failing.**

Run: `uv run pytest`
Slow opt-in (demo-dataset parity, ~11 min per run): `MAGICAL_RUN_SLOW_TESTS=1 uv run pytest`

### Cross-language parity numbers (2026-07-23)

**Synthetic fixture** — run `Rscript re-implementation/scripts/gen_parity_snapshots.R`
then `uv run pytest tests/test_parity_r.py`.

| Comparison | Metric | Value |
|---|---|---|
| numpy vs R `magical_initialization` | `atol` | `1e-10` (bit-identical) |
| numpy vs R `MAGICAL_estimation` @ 50 iters, TF-Peak posterior | Pearson corr | 0.98 – 0.99 |
| numpy vs R `MAGICAL_estimation` @ 50 iters, Peak-Gene posterior | Pearson corr | 0.98 – 0.99 |
| Zero-prior mask agreement | exact | ✅ |

**Real demo dataset @ n_iter = 1000 (paper defaults)** — snapshots produced by
[`gen_demo_parity_snapshots.R`](re-implementation/scripts/gen_demo_parity_snapshots.R) /
[`save_py_snapshots.py`](re-implementation/scripts/save_py_snapshots.py); analysed by
[`plot_variability.py`](re-implementation/scripts/plot_variability.py). 4-choose-2 pairwise
comparison across R seeds {20260723, 42} and Python seeds {20260723, 42}:

| Comparison                | Circuit Jaccard (B≥0.8, L≥0.95) | B Pearson r | L Pearson r |
|---------------------------|:-------------------------------:|:-----------:|:-----------:|
| R ↔ R   (intra, MCMC noise) | 0.9012 | 0.99966 | 0.99988 |
| Py ↔ Py (intra, MCMC noise) | 0.9076 | 0.99966 | 0.99990 |
| R ↔ Py  (cross, mean of 4)  | **0.9090** | 0.99966 | 0.99990 |

**Δ = cross − mean(intra) = +0.005.** Cross-language agreement is statistically
indistinguishable from intra-language MCMC sampler variance. See
[`BENCHMARKS.md`](BENCHMARKS.md) for runtime numbers (**3.02× speedup**
end-to-end at the paper defaults) and
[`benchmarks/plots/variability_*.png`](re-implementation/benchmarks/plots/) for the plots.

---

## Design decisions (proposed, confirm before implementing)

- **Two backends**, same facade: `magical.estimation(..., backend={"numpy","cpp"})`, mirroring the R wrapper's `backend = c("r", "cpp")` pattern.
- **Package name `magical`** already set. Public API module: `magical.api`.
- **Sparse matrices**: `scipy.sparse` for the P×M / P×G candidate matrices and the cell×feature count matrices (equivalent of R's `Matrix::sparseMatrix`).
- **Fixture parity**: rewrite [`helper-synthetic.R`](re-implementation/tests/testthat/helper-synthetic.R) as `tests/conftest.py` fixtures that produce the *same structural shapes* (P=20, G=10, M=5, S=6). Use `numpy.random.default_rng(seed)` seeded identically so tests are reproducible, but do NOT expect bit-identical values against the R fixture (different RNG algorithms).
- **Loaders**: rewrite `Data_loading` / `Candidate_circuits_construction_*` / `MAGICAL_initialization` in numpy as well. These are one-off — no perf pressure — but they must exist so Python users don't have to shell out to R.
- **RNG**: `numpy.random.Generator` throughout; expose a `seed` parameter on the public API so tests are deterministic within Python.

---

## Stage P1 — Numpy raw port (independent of C++/gfortran blocker)

**Status: COMPLETE as of 2026-07-23.** Numpy port is at parity with the R
original on both the synthetic fixture (50 iters) and the real demo dataset
(1000 iters, paper defaults). All 7 sub-tasks below are done.

### File plan

```
src/magical/
  __init__.py               # re-exports the public API
  api.py                    # facade: estimation(backend=...)
  types.py                  # LoadedData, CandidateCircuits, InitialModel, Result dataclasses
  io.py                     # Data_loading equivalent
  circuits.py               # Candidate_circuits_construction_{with,without}_TAD
  initialization.py         # MAGICAL_initialization
  samplers.py               # the 5 Gibbs step functions (numpy)
  estimation.py             # MAGICAL_estimation driver loop (numpy)
  output.py                 # MAGICAL_circuits_output
tests/
  conftest.py               # synthetic fixture (parity with R helper)
  test_types.py             # dataclass round-trips
  test_initialization.py    # mirror of R test-initialization.R
  test_samplers.py          # mirror of R test-samplers.R
  test_estimation_api.py    # mirror of R test-api-contract.R
  test_estimation_extra.py  # mirror of R test-estimation-extra.R
  test_output.py            # mirror of R test-output.R
  test_equivalence.py       # numpy-vs-numpy cross-seed, plus numpy-vs-cpp (skipped)
  test_demo_slow.py         # opt-in full-dataset run (env var gated)
```

### Port order (each step lands its tests before moving on)

1. ✅ **`types.py`** — done. `test_types.py` has 3 tests.
2. ✅ **Synthetic fixture** in `conftest.py` — done. Same P/G/M/S as R helper.
3. ✅ **`initialization.py`** — done. `test_initialization.py` has 11 tests + `test_parity_r.py` shows bit-identical output vs R.
4. ✅ **`samplers.py`** — done. `test_samplers.py` has 15 tests (shape / mask / determinism per sampler).
5. ✅ **`estimation.py`** — done, including the latent-bug fix (`max(1, round(iteration_num/10))`). `test_api_contract.py` + `test_estimation_extra.py` = 10 tests.
6. ✅ **`output.py`** — done. `test_output.py` has 3 tests.
7. ✅ **`io.py`** + **`circuits.py`** — file loaders for the real demo dataset. Done. [`tests/test_parity_demo.py`](tests/test_parity_demo.py) has 12 tests that run the full pipeline against R-produced snapshots at n_iter=1000. Gated behind `MAGICAL_RUN_SLOW_TESTS=1`.

### Target after stage P1

- ✅ ~45 Python tests — achieved 72 collected (57 passing + 15 gated).
- ✅ `magical.estimation(..., backend="numpy")` reproduces R output at correlation ≥ 0.90 on the synthetic fixture (0.98–0.99) and at correlation ≥ 0.999 on the real demo dataset at 1000 iters.
- ✅ Demo-dataset parity established: Python vs R circuit-set Jaccard (0.909) exceeds intra-R MCMC noise floor (0.901); B/L posterior Pearson r ≈ 0.9997.

### Not in scope for P1

- Any C++ / pybind11 / build-system work.
- Performance tuning — the numpy port is a reference, not a speed contender. Vectorize the obvious sweeps (`rnorm(G)` → `rng.normal(size=G)`) but don't rewrite algorithmically.

---

## Stage P2 — pybind11 bindings onto the shared C++ core

**Blocked on** stage 2 of [CPP_PLAN.md](CPP_PLAN.md) landing — the C++ core has to exist before we can bind to it.

### Additional deps (do NOT install until this stage starts)

- `pybind11` — bindings
- `scikit-build-core` or `meson-python` — build backend that can compile C++
- System-side: `cmake`, `ninja` (via `brew install cmake ninja`)

### File plan

```
src/magical/
  _cpp.pyi                  # type stubs for the compiled extension
  _cpp.cpp                  # pybind11 module: adapts numpy → arma → magical_core
src/
  magical_core.{cpp,hpp}    # shared with R (see CPP_PLAN.md)
```

Update [`pyproject.toml`](pyproject.toml):
- Switch build backend from `uv_build` to `scikit-build-core.build`.
- Add `CMakeLists.txt` that compiles `_cpp.so` from `magical_core.cpp` + `_cpp.cpp`.

### Facade

`api.py` gains `backend="cpp"` that imports `magical._cpp` and calls the compiled entrypoint. Numpy arrays are passed by reference; RNG seed forwarded as `uint64`.

### Testing gates

- Unskip `test_equivalence.py::test_numpy_vs_cpp` — must correlate ≥ 0.90 on synthetic fixture.
- Add `test_equivalence.py::test_cpp_python_vs_cpp_r` — same C++ core called from both languages should give bit-identical or near-identical results (same Armadillo RNG, same seed).
- Unskip `test_demo_slow.py::test_cpp_matches_r_on_demo_data`.

### Benchmark gate

Add rows to [BENCHMARKS.md](BENCHMARKS.md) for `backend="cpp"` called from Python. Must be comparable to the R-side C++ number (within noise) — proves the shared core is genuinely shared.

---

## Cross-language equivalence matrix (once everything lands)

The full test grid we're building toward:

|                    | R (orig) | R (cpp) | Py (numpy) | Py (cpp) |
|--------------------|:---:|:---:|:---:|:---:|
| **R (orig)**       | ==  | ~=  | ~=  | ~=  |
| **R (cpp)**        |     | ==  | ~=  | =   |
| **Py (numpy)**     |     |     | ==  | ~=  |
| **Py (cpp)**       |     |     |     | ==  |

`==` bit-identical under fixed seed; `~=` correlation ≥ 0.90 on pooled posteriors; `=` should be nearly bit-identical since both call the same C++ core with the same RNG.

Only the 3 pairs on the top row need integration tests; the rest follow by transitivity.

---

## Files created in stage P1 (all done)

- `src/magical/types.py`
- `src/magical/api.py`
- `src/magical/initialization.py`
- `src/magical/samplers.py`
- `src/magical/estimation.py`
- `src/magical/output.py`
- `src/magical/io.py`
- `src/magical/circuits.py`
- `tests/conftest.py`
- `tests/test_*.py` (10 files, mirroring R suite + demo-parity harness)

---

## Pickup checklist

When resuming this work:

1. `uv sync` to make sure the venv is up to date.
2. Verify tooling: `uv run pytest` (expect 57 passing, 15 skipped).
3. Re-generate synthetic parity snapshots after any R-side change: `Rscript re-implementation/scripts/gen_parity_snapshots.R`.
4. Re-generate demo-dataset parity snapshots (slow, ~11 min Python / ~32 min R per seed): `Rscript re-implementation/scripts/gen_demo_parity_snapshots.R <seed> <suffix> 1000` and `uv run python re-implementation/scripts/save_py_snapshots.py --seed <seed> --iter 1000 --out-suffix <suffix>`.
5. Re-render variability plots: `uv run python re-implementation/scripts/plot_variability.py`.
6. Once C++ core exists (see [CPP_PLAN.md](CPP_PLAN.md)), start stage P2 (pybind11 bindings).