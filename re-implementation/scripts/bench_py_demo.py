"""Clean per-stage timing of the Python pipeline on the demo dataset.

Writes ``re-implementation/benchmarks/py_demo_timings_<N>iter.tsv`` with
one row per stage.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run python \
        re-implementation/scripts/bench_py_demo.py [--iter N]

Default N is 50 to match the fast parity run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from magical import estimation, magical_initialization
from magical.circuits import candidate_circuits_construction_with_TAD
from magical.io import data_loading

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "Demo input files"
OUT_DIR = REPO_ROOT / "re-implementation" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()
    n_iter = args.iter
    out = OUT_DIR / f"py_demo_timings_{n_iter}iter.tsv"

    d = DEMO_DIR
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    t = time.perf_counter()
    loaded = data_loading(
        d / "Cell type candidate genes.txt",
        d / "Cell type candidate peaks.txt",
        d / "Cell type scRNA read count.txt",
        d / "scRNA genes.txt",
        d / "Cell type scRNA cell meta.txt",
        d / "Cell type scATAC read count.txt",
        d / "scATAC peaks.txt",
        d / "Cell type scATAC cell meta.txt",
        d / "Motif mapping prior.txt",
        d / "Motifs.txt",
        d / "hg38_Refseq.txt",
    )
    timings["data_loading"] = time.perf_counter() - t

    t = time.perf_counter()
    cand = candidate_circuits_construction_with_TAD(
        loaded, d / "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt",
    )
    timings["candidate_circuits"] = time.perf_counter() - t

    t = time.perf_counter()
    im = magical_initialization(loaded.as_loaded_data(), cand)
    timings["initialization"] = time.perf_counter() - t

    t = time.perf_counter()
    _ = estimation(
        loaded.as_loaded_data(), cand, im,
        iteration_num=n_iter, backend="numpy", seed=args.seed,
    )
    timings[f"estimation_{n_iter}iter"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0

    with out.open("w") as f:
        f.write("stage\twall_seconds\n")
        for k, v in timings.items():
            f.write(f"{k}\t{v:.4f}\n")

    print(f"Wrote {out}")
    for k, v in timings.items():
        print(f"  {k:22s} {v:8.2f} s")


if __name__ == "__main__":
    main()
