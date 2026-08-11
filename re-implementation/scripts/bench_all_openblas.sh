#!/usr/bin/env bash
# Run R, C++, and Python benchmarks at 1000 iterations using the conda
# magical-r environment (OpenBLAS) for R and C++, and the uv project venv
# for Python.  Both seeds, same as bench_all.sh.
#
# Usage (from repo root):
#   bash re-implementation/scripts/bench_all_openblas.sh
#
# Results append to the same TSVs as bench_all.sh; the new "blas" column
# distinguishes openblas rows from ref_blas rows.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
CONDA_ENV="magical-r"
SEEDS=(20260723 42)
N_ITER=1000

log() { echo "[bench_all_openblas] $(date '+%H:%M:%S') $*"; }

log "=== OpenBLAS benchmark suite  n_iter=$N_ITER ==="
log "    R + C++ via conda env: $CONDA_ENV"
log "    Python via uv venv (scipy-openblas)"

# Python — no change from bench_all.sh (SKYLAKEX has no effect on scipy-openblas64)
for SEED in "${SEEDS[@]}"; do
  log "--- Python numpy  seed=$SEED ---"
  "$UV" run --project "$REPO" python \
    "$REPO/re-implementation/scripts/bench_py_demo.py" \
    --iter "$N_ITER" --seed "$SEED"
done

# R with conda OpenBLAS
for SEED in "${SEEDS[@]}"; do
  log "--- R (conda OpenBLAS)  seed=$SEED ---"
  conda run -n "$CONDA_ENV" Rscript \
    "$REPO/re-implementation/scripts/bench_r_demo.R" \
    "$SEED" "$N_ITER"
done

# C++ with conda OpenBLAS
for SEED in "${SEEDS[@]}"; do
  log "--- C++ (conda OpenBLAS)  seed=$SEED ---"
  conda run -n "$CONDA_ENV" Rscript \
    "$REPO/re-implementation/scripts/bench_cpp_demo.R" \
    "$SEED" "$N_ITER"
done

log "=== All benchmarks complete ==="
echo ""
echo "Results:"
echo "  re-implementation/benchmarks/py_demo_timings_${N_ITER}iter.tsv"
echo "  re-implementation/benchmarks/r_demo_timings.tsv"
echo "  re-implementation/benchmarks/cpp_demo_timings.tsv"
echo ""
echo "Compare old (ref_blas) vs new (openblas) rows with:"
echo "  grep -v 'ref_blas' re-implementation/benchmarks/r_demo_timings.tsv"
