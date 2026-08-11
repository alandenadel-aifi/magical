#!/usr/bin/env bash
# Generate R and Python snapshots for the variability analysis.
# C++ snapshots already exist from bench_cpp_demo.R.
#
# Runtime: ~34 min (Python x2) + ~100 min (R x2) = ~2.5 h total

set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

log() { echo "[gen_snapshots] $(date '+%H:%M:%S') $*"; }
log "=== Generating R + Python snapshots for variability analysis ==="

for SEED in 20260723 42; do
  log "--- Python  seed=$SEED ---"
  "$UV" run --project "$REPO" python \
    "$REPO/re-implementation/scripts/save_py_snapshots.py" \
    --seed "$SEED" --iter 1000 --out-suffix "Py_1000iter_${SEED}"
done

for SEED in 20260723 42; do
  log "--- R  seed=$SEED ---"
  Rscript --vanilla \
    "$REPO/re-implementation/scripts/gen_demo_parity_snapshots.R" \
    "$SEED" "R_1000iter_${SEED}" 1000
done

log "=== All snapshots done ==="
