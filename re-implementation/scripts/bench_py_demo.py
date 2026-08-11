#!/usr/bin/env python3
"""Clean per-stage timing of the Python pipeline on the demo dataset.

Writes ``re-implementation/benchmarks/py_demo_timings_<N>iter.tsv`` (wide
format, one row per run, matching bench_r_demo.R / bench_cpp_demo.R) and
saves posterior snapshots to ``tests/snapshots_demo_Py_<N>iter_<seed>/``
so plot_variability.py can use them without a separate re-run.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run python \
        re-implementation/scripts/bench_py_demo.py [--iter N] [--seed S]

Default N=50, S=20260723.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

    print(f"[bench] seed={args.seed}  n_iter={n_iter}", flush=True)

    # detect BLAS (numpy 2.x changed the config API)
    try:
        import numpy as _np
        # numpy >= 2.0: use show_config(mode='dicts')
        _cfg = _np.show_config(mode="dicts")  # type: ignore[call-arg]
        _blas = _cfg.get("Build Dependencies", {}).get("blas", {})
        _name = _blas.get("name", "").lower()
    except Exception:
        try:
            # numpy < 2.0 fallback
            _info = _np.__config__.blas_opt_info  # type: ignore[attr-defined]
            _name = " ".join(str(v) for v in _info.values()).lower()
        except Exception:
            _name = ""
    if "openblas" in _name:
        blas_tag = "scipy_openblas"
    elif "mkl" in _name:
        blas_tag = "mkl"
    else:
        blas_tag = "scipy_openblas"  # scipy-openblas is bundled default
    _core = os.environ.get("OPENBLAS_CORETYPE", "")
    if _core:
        blas_tag = f"{blas_tag}_{_core.lower()}"
    print(f"[bench] blas={blas_tag}", flush=True)
    t0 = time.perf_counter()
    t = time.perf_counter()
    print("[bench] data_loading ...", flush=True)
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
    print(f"[bench]   data_loading          {timings['data_loading']:8.2f} s", flush=True)

    t = time.perf_counter()
    print("[bench] candidate_circuits ...", flush=True)
    cand = candidate_circuits_construction_with_TAD(
        loaded, d / "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt",
    )
    timings["candidate_circuits"] = time.perf_counter() - t
    print(f"[bench]   candidate_circuits     {timings['candidate_circuits']:8.2f} s", flush=True)

    t = time.perf_counter()
    print("[bench] initialization ...", flush=True)
    im = magical_initialization(loaded.as_loaded_data(), cand)
    timings["initialization"] = time.perf_counter() - t
    print(f"[bench]   initialization         {timings['initialization']:8.2f} s", flush=True)

    t = time.perf_counter()
    print(f"[bench] estimation ({n_iter} iter) ...", flush=True)
    result = estimation(
        loaded.as_loaded_data(), cand, im,
        iteration_num=n_iter, backend="numpy", seed=args.seed,
    )
    timings[f"estimation_{n_iter}iter"] = time.perf_counter() - t
    print(f"[bench]   estimation             {timings[f'estimation_{n_iter}iter']:8.2f} s", flush=True)

    timings["total"] = time.perf_counter() - t0
    print(f"[bench]   total                  {timings['total']:8.2f} s", flush=True)

    # ---- Write timing TSV (wide format, one row per run) -----------------
    timing_row = pd.DataFrame([{
        "seed":                      args.seed,
        "n_iter":                    n_iter,
        "blas":                      blas_tag,
        "data_loading_seconds":      round(timings["data_loading"], 3),
        "candidate_circuits_seconds": round(timings["candidate_circuits"], 3),
        "initialization_seconds":    round(timings["initialization"], 3),
        "estimation_seconds":        round(timings[f"estimation_{n_iter}iter"], 3),
        "total_seconds":             round(timings["total"], 3),
    }])
    append_mode = out.exists()
    timing_row.to_csv(out, sep="\t", index=False,
                      header=not append_mode, mode="a" if append_mode else "w")
    print(f"Wrote {out}")

    # ---- Save posterior snapshots ----------------------------------------
    snap_dir = REPO_ROOT / "tests" / f"snapshots_demo_Py_{n_iter}iter_{args.seed}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    def _write_df(df: pd.DataFrame, name: str) -> None:
        df.to_csv(snap_dir / f"{name}.tsv", sep="\t", index=False)

    def _write_mat(m: np.ndarray, name: str) -> None:
        np.savetxt(snap_dir / f"{name}.tsv", np.asarray(m), delimiter="\t")

    _write_df(cand.TFs,   "TFs")
    _write_df(cand.Peaks, "Peaks")
    _write_df(cand.Genes, "Genes")
    _write_mat(result.TF_Peak_Binding_prob,   "TF_Peak_Binding_prob")
    _write_mat(result.Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
    _write_mat(result.Noise_parameters,       "Noise_parameters")
    (snap_dir / "n_iter.txt").write_text(f"{n_iter}\n")
    (snap_dir / "seed.txt").write_text(f"{args.seed}\n")
    print(f"Wrote snapshots to {snap_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
