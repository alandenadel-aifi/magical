# Environment & BLAS configuration

## Current machine

| Property | Value |
|---|---|
| Hostname | `gcp-hub-01` |
| OS | Ubuntu 22.04.5 LTS |
| CPU | Intel Xeon Platinum 8581C @ 2.10 GHz (Ice Lake SP) |
| Logical CPUs | 224 (2-socket NUMA) |
| AVX-512 flags | `avx512f avx512bw avx512cd avx512dq avx512vl avx512ifma avx512vbmi avx512vbmi2 avx512vnni avx512bitalg avx512vpopcntdq avx512_fp16 avx512_bf16` |
| R | 4.6.1 (2026-06-24) |
| Python | 3.12.12 (miniforge3) |
| uv | 0.9.27 at `~/.local/bin/uv` |
| Project venv | `/home/aland/magical/.venv` |

Verify AVX-512 support at any time:

```bash
grep -m1 'flags' /proc/cpuinfo | grep -o 'avx512[^ ]*' | sort
```

---

## Current BLAS situation (as benchmarked 2026-07-24)

### R (4.6.1, system install)

```
/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.10.0
```

This is the Debian/Ubuntu **reference BLAS** (`libblas3` package) — a portable scalar
C implementation with no SIMD and no threading. Single-threaded by design.

### C++ / Rcpp backend

Inherits R's BLAS at load time. The compiled `.so` is loaded into R's process, so
whatever `libblas.so.3` R resolves at startup is what Armadillo operations use. With
the reference BLAS this means the C++ backend also runs scalar, no SIMD.

### Python numpy (project venv)

```
scipy-openblas  0.3.33  (64-bit integer build, DYNAMIC_ARCH)
lib: scipy_openblas64/lib  (bundled inside the scipy wheel)
```

`DYNAMIC_ARCH` means OpenBLAS does a CPUID probe at first use and selects a
microarchitecture codepath. On this machine it selected **Haswell** (AVX2 only),
leaving AVX-512 unused. The CPU supports 13 `avx512*` flags including `avx512f`,
`avx512vnni`, and `avx512_bf16` — all unused by the current numpy configuration.

### Why this matters

All MAGICAL hot loops (B and L sampling) reduce to large matrix-multiply and
matrix-vector operations. At `n_iter = 1000`, estimation takes:

| Backend | BLAS | Estimation (mean 2 seeds) |
|---|---|---:|
| R | ref BLAS (scalar) | 2833 s |
| C++ / Rcpp | ref BLAS (scalar, inherited) | 1790 s |
| Python numpy | OpenBLAS Haswell (AVX2) | 990 s |

Python leads only because it has *any* SIMD BLAS. R and C++ are bottlenecked by the
scalar reference BLAS, not by algorithmic differences.

---

## Optimization options (no system-wide installs required)

None of the options below require `sudo`. All installs target `~/miniforge3` or the
project venv.

### Option 1 — env var only (Python, zero cost)

```bash
OPENBLAS_CORETYPE=SKYLAKEX uv run python re-implementation/scripts/bench_py_demo.py
```

`OPENBLAS_CORETYPE` overrides `DYNAMIC_ARCH`'s runtime detection and forces the
AVX-512 Skylake-X codepath. No installation needed. Verify it took effect:

```bash
OPENBLAS_CORETYPE=SKYLAKEX python3 -c \
  "import numpy as np; np.show_config()" | grep -i openblas
```

Expected speedup if successful: ~1.5–2× on estimation vs current Haswell path
(990 s → ~500–700 s). If it makes no difference, it suggests the bottleneck is
elsewhere (memory bandwidth, Python overhead) rather than FLOP throughput.

**Applies to**: Python only. R and C++ are unaffected.

---

### Option 2 — conda OpenBLAS for R + C++ (user-space)

conda-forge has OpenBLAS 0.3.34 available. Installing into a new conda environment
is entirely user-space (writes to `~/miniforge3`):

```bash
conda create -n magical-r -c conda-forge r-base "libblas=*=*openblas"
conda activate magical-r
Rscript re-implementation/scripts/bench_r_demo.R
```

Conda's R links to conda's OpenBLAS, which uses `DYNAMIC_ARCH` — same detection
issue as Python may apply. To force AVX-512:

```bash
OPENBLAS_CORETYPE=SKYLAKEX conda run -n magical-r \
  Rscript re-implementation/scripts/bench_r_demo.R
```

The Rcpp backend inherits the same OpenBLAS automatically — no code changes needed.

Expected speedup: ~3–4× on R estimation vs ref BLAS; ~6–9× on C++ estimation vs R
ref-BLAS baseline.

---

### Option 3 — conda MKL for R + C++ (recommended for Intel Xeon)

Intel MKL has no CPU-detection issues on Intel hardware — it correctly uses AVX-512
on Ice Lake SP without any env vars:

```bash
conda create -n magical-mkl -c conda-forge r-base "libblas=*=*mkl"
conda activate magical-mkl
Rscript re-implementation/scripts/bench_r_demo.R
```

MKL 2026.1.0 is available on conda-forge. Rcpp inherits MKL the same way as OpenBLAS.

---

### Option 4 — MKL for Python numpy (user-space)

```bash
# Inside the project venv
uv add mkl intel-openmp
python3 -c "import numpy as np; np.show_config()"
```

Or via conda into its own environment:

```bash
conda create -n magical-py -c conda-forge numpy "libblas=*=*mkl"
```

numpy auto-detects MKL when it is installed and on `LD_LIBRARY_PATH`. MKL will
correctly use AVX-512 on this CPU without `OPENBLAS_CORETYPE`.

---

## Projected timings after BLAS fix

The table below mixes **confirmed measurements** (2026-07-25) with estimates for
paths not yet tested (AVX-512, standalone C++).

| Configuration | 1000-iter estimation | vs R ref-BLAS | status |
|---|---:|---:|---|
| R ref BLAS | 2833 s | 1× | measured |
| Python scipy-openblas Haswell | 982 s | 2.88× | measured |
| R conda OpenBLAS | **2158 s** | **1.31×** | measured 2026-07-25 |
| **C++ / Rcpp conda OpenBLAS** | **624 s** | **4.54×** | **measured 2026-07-25 ✅ beats Python** |
| Python scipy-openblas AVX-512 | ~500–700 s | ~4–6× | not yet — scipy-openblas ignores `OPENBLAS_CORETYPE` |
| C++ / Rcpp OpenBLAS AVX-512 | ~300–450 s | ~6–9× | not yet — needs custom OpenBLAS build |
| Standalone C++ (no R overhead) | ~200–350 s | ~8–14× | not yet — see TODO.md § 2 |

The 10× project target is expected to be met by the Rcpp backend once R's BLAS is
switched (Option 2 or 3 above).

---

## Threading note

OpenBLAS and MKL default to using all available cores. With 224 logical CPUs this
can cause over-subscription if you run multiple MAGICAL chains. Limit BLAS threads:

```bash
export OPENBLAS_NUM_THREADS=1   # for OpenBLAS
export MKL_NUM_THREADS=1        # for MKL
export OMP_NUM_THREADS=1        # catches both via OpenMP
```

Set these before running any benchmarks or production MAGICAL jobs.

---

## Recommended next step

Option 3 (conda MKL for R + C++) is the most promising untested path.
Intel MKL correctly uses AVX-512 on Ice Lake SP without detection issues, which
could push C++ below ~400 s:

```bash
conda create -n magical-mkl -c conda-forge r-base "libblas=*=*mkl"
conda install -n magical-mkl -c conda-forge r-matrix r-dplyr r-rcpp r-rcpparmadillo
conda run -n magical-mkl Rscript re-implementation/scripts/bench_cpp_demo.R 20260723 100
```

**Note**: `OPENBLAS_CORETYPE=SKYLAKEX` is confirmed to have no effect on
`scipy-openblas64` (the bundled build that numpy uses in the uv project venv).
That env var is silently ignored. See [BENCHMARKS.md](BENCHMARKS.md) for the
full measured comparison.
