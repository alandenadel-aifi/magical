"""Run the Python MAGICAL pipeline on the demo data and dump the outputs.

The output layout matches the R snapshot script
(``re-implementation/scripts/gen_demo_parity_snapshots.R``) so that
``plot_variability.py`` (and other consumers) can load either language's
outputs interchangeably from disk.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run python \
        re-implementation/scripts/save_py_snapshots.py \
        --seed 20260723 --iter 1000 --out-suffix py_1000iter_20260723

Writes to ``tests/snapshots_demo_<out_suffix>/`` with these files
(subset of R's — only what plotting needs):

- ``TFs.tsv``, ``Peaks.tsv``, ``Genes.tsv`` — DataFrames with headers
- ``TF_Peak_Binding_prob.tsv`` — dense (P x M) posterior
- ``Peak_Gene_Looping_prob.tsv`` — dense (P x G) posterior
- ``Noise_parameters.tsv`` — (n_iter x 2) MCMC trace
- ``n_iter.tsv``, ``seed.tsv``, ``Common_samples.tsv`` — meta
- ``timings.tsv`` — per-stage wall time (data_loading, candidate_circuits, ...)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from magical import estimation, magical_initialization
from magical.circuits import candidate_circuits_construction_with_TAD
from magical.io import data_loading

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "Demo input files"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iter", type=int, required=True, dest="n_iter")
    ap.add_argument("--out-suffix", type=str, required=True)
    args = ap.parse_args()

    out_dir = REPO_ROOT / "tests" / f"snapshots_demo_{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Python snapshot run with seed={args.seed}, n_iter={args.n_iter}, "
          f"out_dir={out_dir}")

    d = DEMO_DIR
    timings: dict[str, float] = {}

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
    result = estimation(
        loaded.as_loaded_data(), cand, im,
        iteration_num=args.n_iter, backend="numpy", seed=args.seed,
    )
    timings[f"estimation_{args.n_iter}iter"] = time.perf_counter() - t
    timings["total"] = sum(timings.values())

    for k, v in timings.items():
        print(f"  {k:22s} {v:8.2f} s")

    def write_df(df: pd.DataFrame, name: str) -> None:
        df.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)

    def write_mat(m: np.ndarray, name: str) -> None:
        np.savetxt(out_dir / f"{name}.tsv", np.asarray(m), delimiter="\t")

    write_df(cand.TFs, "TFs")
    write_df(cand.Peaks, "Peaks")
    write_df(cand.Genes, "Genes")
    write_mat(result.TF_Peak_Binding_prob, "TF_Peak_Binding_prob")
    write_mat(result.Peak_Gene_Looping_prob, "Peak_Gene_Looping_prob")
    write_mat(result.Noise_parameters, "Noise_parameters")

    (out_dir / "n_iter.tsv").write_text(f"{args.n_iter}\n")
    (out_dir / "seed.tsv").write_text(f"{args.seed}\n")
    (out_dir / "Common_samples.tsv").write_text(
        "\n".join(loaded.common_samples) + "\n"
    )

    pd.DataFrame(
        {"stage": list(timings.keys()), "wall_seconds": list(timings.values())}
    ).to_csv(out_dir / "timings.tsv", sep="\t", index=False,
             float_format="%.4f")

    print(f"Wrote Python snapshots to {out_dir}")


if __name__ == "__main__":
    main()
