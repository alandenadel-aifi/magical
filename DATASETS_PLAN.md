# Benchmark datasets — plan

Companion to [BENCHMARKS.md](BENCHMARKS.md), [CPP_PLAN.md](CPP_PLAN.md), [PYTHON_PLAN.md](PYTHON_PLAN.md), and [TODO.md](TODO.md).

Goal: move benchmarking beyond the shipped demo dataset so we can (a) measure how each backend (R, Python/numpy, C++/Rcpp) scales with problem size, and (b) confirm cross-backend parity holds on real data outside the demo's blood/PBMC context.

---

## Done so far — demo (tutorial) dataset

The shipped demo dataset is the current benchmark. Earlier results are in [BENCHMARKS.md](BENCHMARKS.md).

### 1. Get the data

The demo inputs are not in git (`Demo input files/` and `Demo input.zip` are gitignored). Download from the Google Drive link in [README.md](README.md) and unzip at the repo root:

```bash
curl -sSL -o "Demo input.zip" \
  "https://drive.usercontent.google.com/download?id=1CerwMHMnS1PNFNMy00OoHQjn6T30M1j4&export=download&confirm=t"
unzip -q -o "Demo input.zip" -x "__MACOSX/*"   # creates "Demo input files/" (~1.6 GB unpacked)
```

### 2. Prerequisites

- **R** with `Matrix`, `dplyr`, `Rcpp`, `RcppArmadillo` (the C++ backend is compiled on first use with `Rcpp::sourceCpp`)
- **uv** for the Python port (`uv run` resolves the project venv from `pyproject.toml` / `uv.lock`)

### 3. Run the tutorial

The original end-to-end tutorial (R backend, 4 parallel chains, writes `MAGICAL_selected_regulatory_circuits.txt`):

```bash
Rscript tutorial.R
```

### 4. Run the per-backend benchmarks

Each script times data loading, candidate circuits, initialization, and estimation separately. Arguments are seed then iterations:

```bash
# Python / numpy
uv run python re-implementation/scripts/bench_py_demo.py --iter 1000 --seed 42
# C++ / Rcpp
Rscript --vanilla re-implementation/scripts/bench_cpp_demo.R 42 1000
# original R
Rscript --vanilla re-implementation/scripts/bench_r_demo.R 42 1000
```

Or run all three backends × 2 seeds with `bash re-implementation/scripts/bench_all.sh`. On Linux with conda OpenBLAS, use `bench_all_openblas.sh` instead.

Side effects to be aware of:
- Timings are **appended** to the tracked TSVs in [`re-implementation/benchmarks/`](re-implementation/benchmarks/), which have no machine column. Don't commit rows from different machines without labelling them.
- Posterior snapshots go to `tests/snapshots_demo_<backend>_<N>iter_<seed>/` (gitignored). Use them for parity checks with `uv run python re-implementation/scripts/plot_variability.py`.

### 5. Results so far

**Apple M4 Pro (12 cores, 24 GB), macOS 15, R 4.5.1 with reference BLAS, numpy with Accelerate. 2026-10-29, seed 42.**

50-iteration smoke test (all three backends confirmed working):

| Backend | Load (s) | Circuits (s) | Init (s) | Estimation (s) | Total (s) | Est. vs R |
|---|---:|---:|---:|---:|---:|---:|
| R (ref BLAS) | 42.3 | 17.5 | 6.5 | 94.5 | 160.8 | 1.00× |
| C++ / Rcpp (ref BLAS) | 41.0 | 16.3 | 6.6 | 60.5 | 124.3 | 1.56× |
| Python / numpy (Accelerate) | 13.4 | 12.8 | 3.3 | 30.9 | 60.5 | 3.06× |

1000-iteration run: Python / numpy only, estimation 621.2 s (load 12.1 s, circuits 13.1 s, init 3.2 s). The C++ and R runs were stopped. The full 1000-iteration comparison will be run on HPC instead of a laptop.

Earlier 1000-iteration results (macOS M-series and Linux Xeon, ref BLAS and OpenBLAS) are in [BENCHMARKS.md](BENCHMARKS.md). In short: C++ with OpenBLAS was fastest on Linux (624 s, 4.5× vs R), and Python was about 3× faster than R on macOS.

---

## What a dataset must provide

MAGICAL models **sample-level** variation, so a usable dataset needs:

| Requirement | Notes |
|---|---|
| Sample-paired scRNA + scATAC (or single-cell multiome) | Same samples/donors in both modalities |
| Multiple samples per group | One pooled sample is not enough, however many cells it has |
| Condition / group labels | Case vs control, disease stage, or at minimum a donor/site grouping |
| Cell-type annotations | To pick one cell type's candidate genes / peaks |
| Raw read counts | Gene × cell and peak × cell |

The inputs are prepared with [`Multiomics_input_for_MAGICAL.R`](Multiomics_input_for_MAGICAL.R) (Seurat + ArchR/Signac). The motif-mapping prior must be regenerated against each dataset's peak set.

---

## Candidate datasets

> ⚠️ Accessions below were listed from memory. Verify each before starting a large download.

### Tier 1 — multi-sample with conditions (best fit)

| Dataset | Modality | Samples / conditions | Why it's useful |
|---|---|---|---|
| **MAGICAL paper — full *S. aureus* PBMC** (GSE220190; RData / ArchR links in [README.md](README.md)) | scRNA + scATAC | MRSA / MSSA / control | Same pipeline as the demo, but every cell type at full scale. Outputs can be checked against the paper. |
| **MAGICAL paper — COVID-19 PBMC** (ArchR link in [README.md](README.md)) | scRNA + scATAC | COVID / control | A second disease setting from the paper, with known expected circuits. |
| **SEA-AD** (Allen Institute; sea-ad.org / AWS Open Data) | snRNA + snATAC + multiome, MTG | 84 donors along an AD pathology continuum | Very large and in-house. Stress test for sample count and cell-type count. Needs a brain TAD prior. |
| **Morabito et al. 2021 — AD prefrontal cortex** (GSE174367) | snRNA + snATAC | ~20 donors, AD vs control | Mid-size case/control brain dataset. The step between the demo and SEA-AD. |
| **Wilson et al. 2022 — diabetic kidney disease** (GSE195460) | snRNA + snATAC | DKD vs control | New tissue with clear epithelial cell types. |
| **Kuppe et al. 2022 — myocardial infarction** (Zenodo / CELLxGENE) | snRNA + snATAC | ~30 samples, MI zones vs control | Many samples with several condition levels. |

### Tier 2 — scale tests (multi-donor, no disease contrast)

| Dataset | Notes |
|---|---|
| **NeurIPS 2021 multiome BMMC** (GSE194122) | 13 samples across donors/sites, true multiome. Use site or donor as the grouping. Good for timing as sample count grows; weak for biology. |
| **10x Genomics public Multiome** (PBMC 10k etc.) | Mostly single-sample. Only usable with pseudo-sample splits. Small extra test. |

### Tier 3 — controlled scaling (subsampled / synthetic)

Vary **one dimension at a time** on the demo or the full *S. aureus* data:

- number of candidate genes
- number of candidate peaks
- number of TFs (motif prior columns)
- number of samples
- cells per sample

Fit estimation runtime against each dimension, per backend. This shows what the hot loop scales with (peaks × genes, TFs × peaks, samples), which tells us what to optimize next. It may matter more for optimization than any single new dataset. Extend the existing synthetic fixture in [`helper-synthetic.R`](re-implementation/tests/testthat/helper-synthetic.R) for the purely synthetic end of the sweep.

---

## Practical considerations

- **TAD priors.** The demo TAD file is GM12878 (blood). For brain, kidney, or heart, use tissue-matched Hi-C (ENCODE / 3D Genome Browser), or run in distance mode with `Candidate_circuits_construction_without_TAD(loaded_data, distance_control)`.
- **Preprocessing cost.** Each new dataset needs a full Seurat + ArchR/Signac pass to produce MAGICAL inputs. The paper's own data is the cheapest to adopt because its integrated objects are already published.
- **Storage.** The full *S. aureus* objects are 15 GB (scRNA) + 34 GB (scATAC, with arrow files). SEA-AD is larger still. Keep raw downloads outside the repo, as already done for `Demo input files/` (see `.gitignore`).
- **Benchmark protocol.** Keep the same conventions as [BENCHMARKS.md](BENCHMARKS.md): `iteration_num = 1000`, single chain, 2 seeds per backend, per-stage timings, and record machine / OS / BLAS / commit hash. Reuse `plot_variability.py` for cross-backend parity on each new dataset.

---

## Proposed order

| Step | Dataset | Purpose | Status |
|---|---|---|---|
| 0 | Demo / tutorial dataset | Baseline timings and cross-backend parity | ✅ done on laptop (50-iter all backends; full 1000-iter comparison pending on HPC) |
| 1 | Full *S. aureus* PBMC (paper) | Real-scale run of the demo pipeline; validate against paper | ⏳ not started |
| 2 | Subsampled scaling sweep on step-1 data | Runtime vs genes / peaks / TFs / samples, per backend | ⏳ not started |
| 3 | SEA-AD | Large, real-world, non-blood stress test | ⏳ not started |
| 4 | Morabito AD / Wilson DKD / Kuppe MI | Breadth across tissues; parity on non-blood data | ⏳ optional |
| 5 | NeurIPS BMMC | Sample-count scaling with true multiome | ⏳ optional |
