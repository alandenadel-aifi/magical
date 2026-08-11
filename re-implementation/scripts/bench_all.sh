#!/usr/bin/env bash
# Run R, C++, and Python benchmarks at 1000 iterations with two seeds each.
# Each script writes timing TSVs AND posterior snapshots in one pass, so a
# separate snapshot-generation step is never needed.
#
# Usage: bash re-implementation/scripts/bench_all.sh
#   from the repo root.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
SEEDS=(20260723 42)
N_ITER=1000

log() { echo "[bench_all] $(date '+%H:%M:%S') $*"; }

log "=== Starting full benchmark suite  n_iter=$N_ITER ==="

for SEED in "${SEEDS[@]}"; do
  log "--- Python numpy  seed=$SEED ---"
  "$UV" run --project "$REPO" python \
    "$REPO/re-implementation/scripts/bench_py_demo.py" \
    --iter "$N_ITER" --seed "$SEED"
done

for SEED in "${SEEDS[@]}"; do
  log "--- R (ref BLAS)  seed=$SEED ---"
  Rscript --vanilla \
    "$REPO/re-implementation/scripts/bench_r_demo.R" \
    "$SEED" "$N_ITER"
done

for SEED in "${SEEDS[@]}"; do
  log "--- C++ (Rcpp, ref BLAS)  seed=$SEED ---"
  Rscript --vanilla \
    "$REPO/re-implementation/scripts/bench_cpp_demo.R" \
    "$SEED" "$N_ITER"
done

log "=== All benchmarks complete ==="
echo ""
echo "Results:"
echo "  re-implementation/benchmarks/py_demo_timings_${N_ITER}iter.tsv"
echo "  re-implementation/benchmarks/r_demo_timings.tsv"
echo "  re-implementation/benchmarks/cpp_demo_timings.tsv"
