# Running the faster MAGICAL implementations

This guide is for users who have already run the original R version (`R/MAGICAL_functions.R`)
and want to use one of the two faster re-implementations:

| Implementation | Speedup vs R | Requires |
|---|---:|---|
| **Python (numpy)** | ~3× end-to-end | Python ≥ 3.12, `uv` |
| **C++ via Rcpp** | ~1.6× (ref BLAS) · **~4.5×** (OpenBLAS) | R + Rcpp + RcppArmadillo |

The input files are identical in both cases — no changes to your data are needed.

---

## Python (numpy) backend

### 1. Install

```bash
# From the repo root
pip install uv      # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync             # creates .venv and installs numpy / scipy / pandas
```

### 2. Run with your own data

```python
from magical.io import data_loading
from magical.circuits import candidate_circuits_construction_with_TAD
                           # or candidate_circuits_construction_without_TAD
from magical import magical_initialization, estimation, magical_circuits_output

# --- load data (same file paths as tutorial.R) ---
loaded = data_loading(
    "Cell type candidate genes.txt",
    "Cell type candidate peaks.txt",
    "Cell type scRNA read count.txt",
    "scRNA genes.txt",
    "Cell type scRNA cell meta.txt",
    "Cell type scATAC read count.txt",
    "scATAC peaks.txt",
    "Cell type scATAC cell meta.txt",
    "Motif mapping prior.txt",
    "Motifs.txt",
    "hg38_Refseq.txt",
)

# --- build candidate circuits (choose one) ---
circuits = candidate_circuits_construction_with_TAD(
    loaded, "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt"
)
# circuits = candidate_circuits_construction_without_TAD(loaded, distance_control=5e5)

# --- run estimation (equivalent to MAGICAL_estimation in R) ---
# Run multiple chains and average, same as tutorial.R
from concurrent.futures import ProcessPoolExecutor

def run_chain(seed):
    init = magical_initialization(loaded.as_loaded_data(), circuits)
    return estimation(loaded.as_loaded_data(), circuits, init,
                      iteration_num=1000, backend="numpy", seed=seed)

with ProcessPoolExecutor(max_workers=4) as pool:
    chain_results = list(pool.map(run_chain, [1001, 1002, 1003, 1004]))

import numpy as np
posterior = type(chain_results[0])(
    TF_Peak_Binding_prob  = np.mean([r.TF_Peak_Binding_prob  for r in chain_results], axis=0),
    Peak_Gene_Looping_prob = np.mean([r.Peak_Gene_Looping_prob for r in chain_results], axis=0),
    Noise_parameters      = np.mean([r.Noise_parameters      for r in chain_results], axis=0),
)

# --- write output (same format as MAGICAL_circuits_output in R) ---
magical_circuits_output("MAGICAL_selected_regulatory_circuits.txt", circuits, posterior)
```

Run directly from the terminal:

```bash
uv run python your_script.py
```

### 3. Verify the install

```bash
uv run pytest          # 43 fast tests; expect 0 failures
```

---

## C++ (Rcpp/Armadillo) backend

The C++ backend is a drop-in replacement for `MAGICAL_estimation`. Everything
else (data loading, circuit construction, initialization, output) is unchanged
from the original `R/MAGICAL_functions.R`.

### 1. Install R dependencies

```r
install.packages(c("Rcpp", "RcppArmadillo"))
```

### 2. Run — one-line change from tutorial.R

Replace:
```r
source('R/MAGICAL_functions.R')
```
with:
```r
source('re-implementation/R/MAGICAL_functions.R')
```

Then add `backend = "cpp"` to the estimation call:

```r
chain_init <- MAGICAL_initialization(loaded_data, Candidate_circuits)
result <- MAGICAL_estimation(loaded_data, Candidate_circuits, chain_init,
                             iteration_num = 1000,
                             backend = "cpp",   # <-- only change
                             seed = 42)
```

The first call compiles `re-implementation/src/magical.cpp` via
`Rcpp::sourceCpp()`. Compilation takes ~30 s. Subsequent calls in the same R
session reuse the compiled binary.

The return value is identical in structure to the original — `TF_Peak_Binding_prob`,
`Peak_Gene_Looping_prob`, and `Noise_parameters` — so `MAGICAL_circuits_output`
works unchanged.

### 3. Verify

```bash
Rscript re-implementation/tests/testthat.R
# expect: 40 tests passing (+ 1 more with MAGICAL_TEST_CPP_BUILD=1)
```

---

## Getting the most performance out of C++

By default R links to the reference BLAS (scalar, no SIMD). The C++ backend
inherits that BLAS, so on a plain R install both paths are bottlenecked. Use a
conda environment with OpenBLAS to unlock the full speedup:

```bash
conda create -n magical-r -c conda-forge r-base r-matrix r-dplyr \
      r-rcpp r-rcpparmadillo "libblas=*=*openblas"
conda activate magical-r
Rscript re-implementation/scripts/bench_cpp_demo.R 42 1000
```

Measured speedups at `n_iter = 1000` on Linux (Intel Xeon Platinum 8581C):

| Backend | BLAS | Estimation | vs R ref-BLAS |
|---|---|---:|---:|
| R | ref BLAS | 2833 s | 1.0× |
| Python numpy | OpenBLAS (Haswell) | 990 s | 2.9× |
| C++ / Rcpp | ref BLAS | 1790 s | 1.6× |
| **C++ / Rcpp** | **conda OpenBLAS** | **624 s** | **4.5×** |

On Intel hardware, substituting MKL for OpenBLAS (use `"libblas=*=*mkl"` in the
conda create command) avoids the CPUID-detection issue and is the recommended
production path.

**Threading note** — OpenBLAS and MKL default to all cores. If you run multiple
chains in parallel, set thread counts to 1 so inner BLAS threads don't fight
outer parallelism:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

---

## Statistical equivalence

All three backends produce statistically indistinguishable results. Across 6
independent runs (R×2, Python×2, C++×2 seeds, 1000 iterations, demo dataset):

| Pair type | Mean circuit Jaccard | B posterior r | L posterior r |
|---|:---:|:---:|:---:|
| R ↔ R (intra) | 0.901 | 0.99966 | 0.99988 |
| Py ↔ Py (intra) | 0.908 | 0.99966 | 0.99990 |
| C++ ↔ C++ (intra) | 0.906 | 0.99965 | 0.99992 |
| R ↔ Python (cross) | 0.909 | 0.99965 | 0.99990 |
| R ↔ C++ (cross) | 0.904 | 0.99965 | 0.99991 |
| Python ↔ C++ (cross) | 0.902 | 0.99965 | 0.99991 |

Cross-language agreement is within the intra-language MCMC noise floor — the
difference between backends is not distinguishable from using a different random
seed.

See [BENCHMARKS.md](BENCHMARKS.md) for the full numbers and
[re-implementation/benchmarks/plots/](re-implementation/benchmarks/plots/) for
the variability plots.
