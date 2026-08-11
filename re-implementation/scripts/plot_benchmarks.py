"""Timing comparison figures for R / C++ / Python numpy backends.

Reads TSVs produced by bench_r_demo.R, bench_cpp_demo.R, and bench_py_demo.py,
then writes two plots to re-implementation/benchmarks/plots/:

- benchmarks_estimation.png   – estimation wall time per method × seed (bar)
- benchmarks_breakdown.png    – stacked bar: setup stages + estimation per method

Usage::

    uv run python re-implementation/scripts/plot_benchmarks.py

Run from the repo root, or let uv handle it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "re-implementation" / "benchmarks"
OUT_DIR = BENCH_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Load timing files                                                            #
# --------------------------------------------------------------------------- #

def _load_r() -> pd.DataFrame:
    p = BENCH_DIR / "r_demo_timings.tsv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t")
    n_iter = df["n_iter"].max()
    return df[df["n_iter"] == n_iter].copy()


def _load_cpp() -> pd.DataFrame:
    p = BENCH_DIR / "cpp_demo_timings.tsv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t")
    # keep only rows that match our n_iter (in case 50-iter warm-up rows exist)
    n_iter = df["n_iter"].max()
    return df[df["n_iter"] == n_iter].copy()


def _load_py(n_iter: int = 1000) -> pd.DataFrame:
    p = BENCH_DIR / f"py_demo_timings_{n_iter}iter.tsv"
    if not p.exists():
        candidates = sorted(BENCH_DIR.glob("py_demo_timings_*iter.tsv"))
        if not candidates:
            raise FileNotFoundError(f"No py_demo_timings TSV found in {BENCH_DIR}")
        p = candidates[-1]
    df = pd.read_csv(p, sep="\t")
    # Accept either wide format (seed, n_iter, *_seconds) or
    # legacy tall format (stage, wall_seconds).
    if "estimation_seconds" in df.columns:
        return df  # already wide
    # Tall format: one row per stage — convert to a single wide row
    stages = {r["stage"]: r["wall_seconds"] for _, r in df.iterrows()}
    est_key = next((k for k in stages if k.startswith("estimation")), None)
    if est_key is None:
        raise ValueError(f"No estimation row in {p}")
    ni = int(est_key.replace("estimation_", "").replace("iter", ""))
    return pd.DataFrame([{
        "seed": "unknown",
        "n_iter": ni,
        "data_loading_seconds": stages.get("data_loading", np.nan),
        "candidate_circuits_seconds": stages.get("candidate_circuits", np.nan),
        "initialization_seconds": stages.get("initialization", np.nan),
        "estimation_seconds": stages[est_key],
        "total_seconds": stages.get("total", np.nan),
    }])


def _method_label(backend: str, blas: str) -> str:
    """Human-readable method label derived from backend + blas column."""
    blas_str = str(blas) if pd.notna(blas) else "unknown"
    if "openblas" in blas_str.lower() and blas_str != "ref_blas":
        blas_label = "OpenBLAS"
    elif blas_str == "ref_blas":
        blas_label = "ref BLAS"
    elif "mkl" in blas_str.lower():
        blas_label = "MKL"
    else:
        blas_label = blas_str
    return f"{backend}\n({blas_label})"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    r_df   = _load_r()
    cpp_df = _load_cpp()

    n_iter = int(r_df["n_iter"].iloc[0])

    # Try to load matching Python TSV; fall back gracefully
    try:
        py_df = _load_py(n_iter)
        has_py = True
    except FileNotFoundError:
        has_py = False
        py_df = pd.DataFrame()

    print(f"Loaded R ({len(r_df)} runs), C++ ({len(cpp_df)} runs), "
          f"Python ({len(py_df)} runs if has_py={has_py})")

    METHOD_COLORS = {
        "R\n(ref BLAS)": "#E24A33",
        "R\n(OpenBLAS)": "#8B1A0E",
        "C++\n(ref BLAS)": "#F7941E",
        "C++\n(OpenBLAS)": "#B05A00",
        "Python\n(OpenBLAS)": "#348ABD",
        "Python\n(MKL)": "#1A5276",
    }

    # ------------------------------------------------------------------ #
    # PLOT 1: estimation time, one bar per run coloured by method         #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bar_data: list[dict] = []
    for _, row in r_df.iterrows():
        blas = row.get("blas", "ref_blas")
        method = _method_label("R", blas)
        bar_data.append({"label": f"R\nseed {row['seed']}",
                         "method": method,
                         "est": row["estimation_seconds"]})
    for _, row in cpp_df.iterrows():
        blas = row.get("blas", "ref_blas")
        method = _method_label("C++", blas)
        bar_data.append({"label": f"C++\nseed {row['seed']}",
                         "method": method,
                         "est": row["estimation_seconds"]})
    if has_py:
        for _, row in py_df.iterrows():
            blas = row.get("blas", "scipy_openblas")
            method = _method_label("Python", blas)
            bar_data.append({"label": f"Python\nseed {row['seed']}",
                             "method": method,
                             "est": row["estimation_seconds"]})

    xs = np.arange(len(bar_data))
    heights = [d["est"] for d in bar_data]
    colors  = [METHOD_COLORS.get(d["method"], "#888888") for d in bar_data]
    labels  = [d["label"] for d in bar_data]

    bars = ax.bar(xs, heights, color=colors, edgecolor="white", linewidth=0.8)
    for bar_, h in zip(bars, heights):
        ax.text(bar_.get_x() + bar_.get_width() / 2, h + 20,
                f"{h/60:.1f} min", ha="center", va="bottom", fontsize=9)

    # speedup relative to mean ref-BLAS R
    ref_r_rows = [d for d in bar_data if d["method"] == "R\n(ref BLAS)"]
    mean_r_est = (
        np.mean([d["est"] for d in ref_r_rows]) if ref_r_rows
        else r_df["estimation_seconds"].mean()
    )
    for i, d in enumerate(bar_data):
        sx = mean_r_est / d["est"]
        if abs(sx - 1.0) > 0.05:
            ax.text(xs[i], d["est"] / 2,
                    f"{sx:.2f}×", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    # legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=m)
                      for m, c in METHOD_COLORS.items()
                      if any(d["method"] == m for d in bar_data)]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wall time (seconds)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_title(
        f"MAGICAL estimation wall time by backend (n_iter={n_iter})\n"
        f"Machine: Intel Xeon Platinum 8581C @ 2.10 GHz — speedup labels vs mean R"
    )
    fig.tight_layout()
    out1 = OUT_DIR / "benchmarks_estimation.png"
    fig.savefig(out1, dpi=150)
    print(f"Wrote {out1.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------ #
    # PLOT 2: stacked bar — setup stages + estimation                     #
    # ------------------------------------------------------------------ #
    method_rows: dict[str, list[dict]] = {}
    for _, row in r_df.iterrows():
        blas = row.get("blas", "ref_blas")
        key = _method_label("R", blas).replace("\n", "\n")
        method_rows.setdefault(key, []).append(row.to_dict())
    for _, row in cpp_df.iterrows():
        blas = row.get("blas", "ref_blas")
        key = _method_label("C++", blas)
        method_rows.setdefault(key, []).append(row.to_dict())
    if has_py:
        for _, row in py_df.iterrows():
            blas = row.get("blas", "scipy_openblas")
            key = _method_label("Python", blas)
            method_rows.setdefault(key, []).append(row.to_dict())

    method_names = list(method_rows.keys())
    stages = ["data_loading_seconds", "candidate_circuits_seconds",
              "initialization_seconds", "estimation_seconds"]
    stage_labels = ["Data loading", "Candidate circuits", "Initialization", "Estimation"]
    stage_colors = ["#aec6cf", "#b5ead7", "#ffdac1", "#ff9999"]

    # Average across seeds per method
    means: dict[str, dict[str, float]] = {}
    errs:  dict[str, dict[str, float]] = {}
    for mname, rows in method_rows.items():
        means[mname] = {}
        errs[mname]  = {}
        for s in stages:
            vals = [r[s] for r in rows if not np.isnan(r.get(s, np.nan))]
            means[mname][s] = float(np.mean(vals)) if vals else 0.0
            errs[mname][s]  = float(np.std(vals))  if len(vals) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs2 = np.arange(len(method_names))
    bottoms = np.zeros(len(method_names))
    for s, slabel, scolor in zip(stages, stage_labels, stage_colors):
        heights2 = [means[m][s] for m in method_names]
        err2     = [errs[m][s]  for m in method_names]
        yerr_arr = [e if e > 0 else 0 for e in err2]
        ax.bar(xs2, heights2, bottom=bottoms, label=slabel,
               color=scolor, edgecolor="white", linewidth=0.5)
        # error bars only on estimation (last stage)
        if s == "estimation_seconds":
            ax.errorbar(xs2, bottoms + np.array(heights2),
                        yerr=yerr_arr, fmt="none", color="black",
                        capsize=4, linewidth=1.5)
        bottoms += np.array(heights2)

    for i, (m, total_b) in enumerate(zip(method_names, bottoms)):
        ax.text(xs2[i], total_b + 30, f"{total_b/60:.0f} min",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs2)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylabel("Wall time (seconds)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"MAGICAL wall time breakdown by stage (n_iter={n_iter}, mean ± SD over seeds)\n"
        f"Error bars on estimation bar show seed-to-seed variation"
    )
    fig.tight_layout()
    out2 = OUT_DIR / "benchmarks_breakdown.png"
    fig.savefig(out2, dpi=150)
    print(f"Wrote {out2.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    print(f"\nEstimation summary (n_iter={n_iter}):")
    print(f"{'Method':<30} {'seed':>10} {'estimation (s)':>16} {'vs R ref-BLAS':>14}")
    print("-" * 74)
    for df_, backend in [(r_df, "R"), (cpp_df, "C++")]:
        for _, row in df_.iterrows():
            blas = row.get("blas", "ref_blas")
            label = _method_label(backend, blas).replace("\n", " ")
            sx = mean_r_est / row["estimation_seconds"]
            print(f"{label:<30} {str(row['seed']):>10} {row['estimation_seconds']:>16.1f} {sx:>13.2f}×")
    if has_py:
        for _, row in py_df.iterrows():
            blas = row.get("blas", "scipy_openblas")
            label = _method_label("Python", blas).replace("\n", " ")
            sx = mean_r_est / row["estimation_seconds"]
            print(f"{label:<30} {str(row['seed']):>10} {row['estimation_seconds']:>16.1f} {sx:>13.2f}×")

    print("\nDone.")


if __name__ == "__main__":
    main()
