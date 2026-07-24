# TODO — what remains

Living document for the MAGICAL re-implementation work. Cross-refs to the
per-workstream plans:

- [BENCHMARKS.md](BENCHMARKS.md) — measured wall-clock numbers per backend
- [CPP_PLAN.md](CPP_PLAN.md) — R + Rcpp/Armadillo C++ port
- [PYTHON_PLAN.md](PYTHON_PLAN.md) — numpy port + future pybind11 bindings

---

## Status snapshot (as of 2026-07-23)

| Workstream | State |
|---|---|
| Python numpy port (stage P1) | ✅ complete — parity with R at demo scale, 3.02× speedup |
| C++ port via Rcpp/Armadillo (stage 2) | ✅ complete — 3-way parity confirmed at 50 iter |
| C++ standalone / pybind11 (stage P2) | ⏳ not started |
| Test suites | ✅ R: 44 tests (40 pass / 4 opt-in). Python: 72 tests (57 pass / 15 gated). |
| Cross-language parity (R ↔ Py ↔ Cpp) | ✅ measured manually at 50 iter — not yet encoded as tests |
| BLAS story documented | ✅ (this doc, § "BLAS backend") |
| Linux setup / Dockerfile | ⏳ not started |
| CI | ⏳ not started |

---

## 1. BLAS backend — the big story

**Symptom** (seen 2026-07-23, MacBook Apple Silicon, 50-iter demo estimation):

| Backend | Estimation | vs R |
|---|---:|---:|
| R (baseline) | ~94 s | 1.00× |
| Python numpy | 31.6 s | 2.97× |
| **C++ (Rcpp/Armadillo)** | **60.3 s** | 1.56× |

C++ **slower** than Python — surprising, because the algorithm and iteration
counts are identical. Root cause is not the C++ code.

### Root cause

R (from the CRAN macOS installer) links to its own bundled **reference BLAS**:

```
BLAS:   /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRblas.0.dylib
```

`libRblas` is a portable scalar C implementation with no SIMD, no threading,
no cache blocking. RcppArmadillo (and therefore [`re-implementation/src/magical.cpp`](re-implementation/src/magical.cpp))
inherits R's BLAS at load time, so the compiled `.so` also goes through
`libRblas`.

NumPy on macOS uses **Apple Accelerate** (vecLib) — multi-threaded, NEON + AMX,
50–200× more GFLOPS on ~1000×1000 GEMM.

The dominant kernel in the sampler is `L.T @ A_estimate` inside
`peak_gene_looping_sampling` (~12 B FLOPs per iteration). At Accelerate speeds
that's a fraction of a second; at `libRblas` speeds it's the whole runtime.

### Why CRAN R uses reference BLAS

1. **Deterministic reproducibility** — reference BLAS is single-threaded so
   reductions happen in a fixed serial order, giving bit-identical output
   across machines.
2. **Historical Fortran/vecLib ABI bugs** in older macOS BLAS integer sizes.
3. **Thread over-subscription risk** if users are already parallelising
   in R (`mclapply` + N-thread BLAS = N² threads).

None of these apply to MAGICAL:

- MAGICAL is MCMC — outputs are stochastic by definition, we're already
  4+ significant figures of correlation between chains
- ABI bugs are fixed in modern macOS
- MAGICAL is single-chain per process

### Fix options

Roughly in order of effort:

- **Path A — swap R's BLAS to Accelerate (macOS)**: officially documented in
  [R Installation and Administration §B.4](https://cran.r-project.org/doc/manuals/r-release/R-admin.html#Accelerate).
  One symlink:
  ```bash
  cd /Library/Frameworks/R.framework/Resources/lib
  ln -sf /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/libBLAS.dylib libRblas.dylib
  ```
  Both R baseline and the Rcpp `.so` immediately use Accelerate. Reversible.
  **Not our priority — Linux is the deployment target.**

- **Path B — link Accelerate directly from `sourceCpp` (macOS)**: add
  `PKG_LIBS: -framework Accelerate` to `magical.cpp`. Leaves R alone; only the
  compiled `.so` benefits. Only useful if we want the C++ path to be fast on
  macOS without touching the system R.

- **Path C — Linux + OpenBLAS**: on Debian/Ubuntu:
  ```bash
  sudo apt install libopenblas-dev liblapack-dev
  sudo update-alternatives --config libblas.so.3-$(dpkg --print-architecture | sed 's/amd64/x86_64/;s/arm64/aarch64/')-linux-gnu
  sudo update-alternatives --config liblapack.so.3-$(...)-linux-gnu
  ```
  All three implementations (R, RcppArmadillo, Python) pick up OpenBLAS
  transparently. No source changes. Comparable to Accelerate for our matrix
  sizes.
  **This is the intended production path.** See § 3 below.

- **Path D — standalone C++ binary (any OS)**: strip R+Python dependencies
  entirely, link `-lopenblas -llapack` (Linux) or `-framework Accelerate`
  (macOS). Requires extracting the core out of `magical.cpp` first — see § 2.

### Expected numbers once BLAS is fixed

Rough projection on a mid-range x86 Linux box with OpenBLAS:

| implementation | 50-iter estimation |
|---|---:|
| R baseline (OpenBLAS) | ~15–25 s |
| C++ via Rcpp (OpenBLAS) | ~6–12 s |
| NumPy (bundled OpenBLAS/MKL) | ~25–35 s |
| **Standalone C++ (OpenBLAS + buffer reuse)** | **~3–8 s** |

The "≥10× over R" target from [CPP_PLAN.md](CPP_PLAN.md) becomes trivial on
Linux with a configured BLAS.

### Threading gotcha

OpenBLAS / Accelerate default to all cores. If you're running multiple MAGICAL
processes in parallel (multiple cell types, seeds, or jobs), set:

```bash
export OPENBLAS_NUM_THREADS=1        # Linux
export VECLIB_MAXIMUM_THREADS=1      # macOS Accelerate
```

so inner-BLAS threads don't fight outer parallelism.

---

## 2. Extract the C++ core (`magical_core.hpp` / `.cpp`)

**Precondition to stages 3, 4, and 5 below.** Today
[`re-implementation/src/magical.cpp`](re-implementation/src/magical.cpp) mixes:

- pure C++/Armadillo samplers (anonymous namespace)
- Rcpp-specific `[[Rcpp::export]]` adapters

Split into:

```
re-implementation/src/
├── magical_core.hpp       # sampler + driver declarations, pure C++/Armadillo
├── magical_core.cpp       # implementations, no R or Python deps
└── magical_rcpp.cpp       # Rcpp adapters only
```

Then the same `.o` from `magical_core.cpp` links into:

- the R backend (via `magical_rcpp.cpp` + `sourceCpp`)
- the Python backend (via `_cpp.cpp` + pybind11 — see § 4)
- a standalone CLI (§ 5)

**Effort**: ~1 hour. Mostly moving code, no logic changes. Verify by re-running
the R testthat suite and confirming numeric parity is preserved.

---

## 3. Linux setup + Dockerfile

Rationale: Linux is the deployment target, and reproducible benchmarks belong
in a container.

### Deliverables

- **`scripts/setup_linux_blas.sh`** — one-shot installer that runs
  `apt install libopenblas-dev liblapack-dev` and configures the alternatives
  for the current architecture. Idempotent; safe to re-run.
- **`Dockerfile`** at repo root, targeting Ubuntu 22.04 LTS (matches most
  cluster/CI environments). Draft:
  ```dockerfile
  FROM rocker/r-ver:4.5.2
  RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev liblapack-dev python3-pip build-essential git && \
      ARCH=$(dpkg --print-architecture | sed 's/amd64/x86_64/;s/arm64/aarch64/') && \
      update-alternatives --set libblas.so.3-$ARCH-linux-gnu \
        /usr/lib/$ARCH-linux-gnu/openblas-pthread/libblas.so.3 && \
      update-alternatives --set liblapack.so.3-$ARCH-linux-gnu \
        /usr/lib/$ARCH-linux-gnu/openblas-pthread/liblapack.so.3 && \
      rm -rf /var/lib/apt/lists/*
  RUN R -e 'install.packages(c("Rcpp","RcppArmadillo","Matrix","dplyr","testthat"), repos="https://cloud.r-project.org")'
  RUN pip install --no-cache-dir numpy pandas scipy pytest
  WORKDIR /work
  ```
- **`docker-compose.yml` (optional)** — mount the repo into `/work` and expose
  R + Python + pytest as one-shot services.
- **`scripts/bench_all.sh`** — runs 50-iter and 1000-iter benchmarks for all
  three backends inside the container and writes a fresh `BENCHMARKS.md`
  table.

### Non-goals for this task

- Windows support (defer indefinitely — no user demand)
- ARM Linux specifically (the Dockerfile handles both x86 and ARM via the
  `$ARCH` detection above, but hasn't been tested on ARM Linux)

---

## 4. Stage P2 — pybind11 bindings

**Blocked on § 2** (need the split core first).

Concrete tasks, from [PYTHON_PLAN.md](PYTHON_PLAN.md) stage P2:

- Add `pybind11`, `scikit-build-core`, `cmake`, `ninja` as build deps.
- Write `src/magical/_cpp.cpp` — pybind11 module adapting numpy arrays into
  `arma::mat` views (no copies where possible) and calling
  `magical_core::run_estimation(...)`.
- Add `CMakeLists.txt` that builds `_cpp.so` from `magical_core.cpp` + `_cpp.cpp`.
- Switch [`pyproject.toml`](pyproject.toml) build backend from `uv_build` to
  `scikit-build-core.build`.
- Wire `magical.api.estimation(backend="cpp")` to `import magical._cpp`.
- Un-skip `tests/test_equivalence.py::test_numpy_vs_cpp` — expect correlation
  ≥ 0.90 vs numpy.
- **New test**: `test_cpp_python_vs_cpp_r` — same C++ core called from both
  Python and R with same seed should give correlation ≥ 0.99 (proves the core
  is genuinely shared, not two divergent ports).

**Success criteria**:

- Python `backend="cpp"` runs the demo dataset in the same wall-clock as the
  R-side `backend="cpp"` (within 10%). Any big gap points to a marshalling bug
  in one of the adapters.
- All existing 57 passing Python tests stay green.

---

## 5. Standalone C++ CLI

**Blocked on § 2.** Useful for:
- Benchmarking without R or Python overhead (the "true" C++ speed)
- Deployment scenarios where neither R nor Python is available
- Validating that the "shared core" story really is portable

Deliverable: `re-implementation/src/magical_cli.cpp` — reads TSV inputs, runs
`magical_core::run_estimation`, writes TSV outputs. Same file format as
[`re-implementation/scripts/gen_demo_parity_snapshots.R`](re-implementation/scripts/gen_demo_parity_snapshots.R)
so the existing parity harness works unchanged.

Build via `Makefile` or `CMakeLists.txt`:
```
g++ -std=c++17 -O3 -march=native \
  magical_core.cpp magical_cli.cpp \
  -o magical -larmadillo -lopenblas -llapack
```

---

## 6. Test coverage gaps

Detailed in the "coverage matrix" from the recent conversation. The gaps:

1. **No live R-vs-Py at test time.** Python parity tests read pre-computed R
   snapshots; if R code changes and snapshots aren't regenerated, no test
   catches it.
   - **Fix**: `scripts/regen_all_snapshots.sh` + a CI job that runs it on
     R-code changes (via a `paths` filter).

2. **No 3-way parity test.** The R↔Py↔Cpp comparison we ran manually
   (correlation 0.9933–0.9983, Jaccard 0.923–0.996 at threshold 0.5) isn't
   asserted anywhere.
   - **Fix**: extend [`tests/test_parity_demo.py`](tests/test_parity_demo.py)
     with a `test_cpp_snapshot_matches_python` that loads
     `tests/snapshots_demo_Cpp_<iter>iter_<seed>/` and asserts correlation
     ≥ 0.99 + Jaccard@0.5 ≥ 0.90 against the Py snapshot.
   - Symmetric R-side test in `re-implementation/tests/testthat/`.

3. **R testthat cpp-vs-r equivalence test uses tiny synthetic fixture only.**
   The strong ≥0.99 correlation we saw at demo scale isn't asserted.
   - **Fix**: add a demo-scale cpp-vs-r test gated on
     `MAGICAL_RUN_SLOW_TESTS=1` + `MAGICAL_TEST_CPP_BUILD=1`.

4. **C++ backend is only reachable from R.** Once § 4 lands, Python tests can
   also exercise it, closing the last quadrant of the coverage matrix from
   [PYTHON_PLAN.md](PYTHON_PLAN.md).

---

## 7. Benchmarks still to run

### At demo scale (paper defaults, `iteration_num = 1000`)

Already have:
- ✅ R × 2 seeds
- ✅ Py × 2 seeds

Still need:
- ⏳ **Cpp × 2 seeds at 1000 iter** — the current C++ number for BENCHMARKS.md
  should be from a full 1000-iter run, not projected from 50-iter. Estimated
  wall-clock ~20 min per seed on this Mac (60 s × 20 = 1200 s), or ~5 min per
  seed on Linux+OpenBLAS after § 3.
- ⏳ **Add Cpp row to [BENCHMARKS.md](BENCHMARKS.md)** side-by-side with the
  R and Py rows.
- ⏳ **Extend variability plot to 6 runs** (2× R, 2× Py, 2× Cpp) →
  6-choose-2 = 15 pairwise Jaccards. Update
  [`re-implementation/scripts/plot_variability.py`](re-implementation/scripts/plot_variability.py).

### On Linux (once § 3 lands)

Re-run all of the above inside the Docker container so the numbers are
reproducible. Replace the macOS-based tables in
[BENCHMARKS.md](BENCHMARKS.md) with Linux/OpenBLAS numbers, and keep the
macOS numbers in a smaller "developer machine reference" appendix.

---

## 8. Housekeeping

- Update [CPP_PLAN.md](CPP_PLAN.md) — mark stage 2 as complete; add the
  measured 3-way correlation numbers under a "Stage 2 status" section.
- Update [PYTHON_PLAN.md](PYTHON_PLAN.md) — the stage-P2 section refers to
  paths that will change after § 2 (`magical_core.{cpp,hpp}` moving out of
  `re-implementation/src/`). Re-anchor once the split lands.
- Consider **moving the C++ port out of `re-implementation/`** once § 2 and
  § 4 are done. Once the C++ core has both R and Python callers, it makes
  more sense at the repo root:
  ```
  src/
  ├── magical/           # Python package (unchanged)
  └── magical_cpp/       # C++ core + Rcpp adapter + pybind11 adapter
  R/                     # gets the modified MAGICAL_functions.R via a
                         # small overlay or symlink from re-implementation/R/
  ```
  Defer this until after § 5 — moving too early creates churn.
- Add a top-level [README.md](README.md) section that points at
  [TODO.md](TODO.md), [BENCHMARKS.md](BENCHMARKS.md), and the two plan docs.
  The upstream README currently only describes the paper pipeline.

---

## 9. Explicit non-goals (unless requested)

- Windows support
- GPU offload (nvBLAS, cuBLAS) — the problem sizes don't justify the
  transfer overhead
- Multi-chain parallel MCMC — the current single-chain design is what the
  paper describes; multi-chain would be a scientific extension, not a
  reimplementation task
- Rewriting `Data_loading` / `Candidate_circuits_construction_*` in C++ —
  they're one-off I/O; already sub-30 s on the demo, dominated by disk parsing
