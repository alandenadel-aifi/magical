"""Intra- vs cross-language variability of MAGICAL selected circuits.

Loads posterior snapshots from four runs (2 R seeds x 2 Python seeds) and
computes all 4-choose-2 = 6 pairwise agreements at the R paper defaults
(iteration_num=1000, thresholds TF-Peak > 0.8 and Peak-Gene > 0.95).

Snapshots are produced by:

  Rscript re-implementation/scripts/gen_demo_parity_snapshots.R \
      20260723 R_1000iter_20260723 1000
  Rscript re-implementation/scripts/gen_demo_parity_snapshots.R \
      42 R_1000iter_42 1000
  uv run python re-implementation/scripts/save_py_snapshots.py \
      --seed 20260723 --iter 1000 --out-suffix Py_1000iter_20260723
  uv run python re-implementation/scripts/save_py_snapshots.py \
      --seed 42 --iter 1000 --out-suffix Py_1000iter_42
  Rscript re-implementation/scripts/bench_cpp_demo.R 20260723 1000
  Rscript re-implementation/scripts/bench_cpp_demo.R 42 1000

Renders to ``re-implementation/benchmarks/plots/``:
- ``variability_jaccard_matrix.png``   6x6 heatmap (15 unique pairs)
- ``variability_same_vs_cross.png``    intra R/Py/C++ vs cross-method bars
- ``variability_components.png``       per-component Jaccards by pair type
- ``variability_posterior_corr.png``   Pearson-r of raw posteriors, 6x6
Writes ``re-implementation/benchmarks/variability_metrics.tsv``.

Interpretation: if intra-language Jaccard is close to cross-language Jaccard,
the disagreement is intrinsic MCMC noise, not an implementation bug.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run python \
        re-implementation/scripts/plot_variability.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "re-implementation" / "benchmarks" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_TSV = REPO_ROOT / "re-implementation" / "benchmarks" / "variability_metrics.tsv"

PROB_TF_PEAK = 0.8
PROB_PEAK_GENE = 0.95

# All six runs to compare (1000-iter, two seeds per backend)
RUNS: dict[str, Path] = {
    "R  seed 20260723":   REPO_ROOT / "tests" / "snapshots_demo_R_1000iter_20260723",
    "R  seed 42":         REPO_ROOT / "tests" / "snapshots_demo_R_1000iter_42",
    "Py seed 20260723":   REPO_ROOT / "tests" / "snapshots_demo_Py_1000iter_20260723",
    "Py seed 42":         REPO_ROOT / "tests" / "snapshots_demo_Py_1000iter_42",
    "C++ seed 20260723":  REPO_ROOT / "tests" / "snapshots_demo_Cpp_1000iter_20260723",
    "C++ seed 42":        REPO_ROOT / "tests" / "snapshots_demo_Cpp_1000iter_42",
}

# Language group for intra vs cross classification
RUN_GROUP: dict[str, str] = {
    "R  seed 20260723":   "R",
    "R  seed 42":         "R",
    "Py seed 20260723":   "Py",
    "Py seed 42":         "Py",
    "C++ seed 20260723":  "C++",
    "C++ seed 42":        "C++",
}

GROUP_COLORS: dict[str, str] = {
    "R":   "#E24A33",
    "Py":  "#348ABD",
    "C++": "#F7941E",
}

# Colors for cross-method pair types (sorted pair name → color)
CROSS_COLORS: dict[str, str] = {
    "C++↔Py": "#9467BD",   # purple
    "C++↔R":  "#17BECF",   # teal
    "Py↔R":   "#2CA02C",   # dark green
}


def _mat(snap_dir: Path, name: str) -> np.ndarray:
    return np.loadtxt(snap_dir / f"{name}.tsv", delimiter="\t")


def _df(snap_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(snap_dir / f"{name}.tsv", sep="\t")


def _read_col(snap_dir: Path, tsv_name: str, *col_candidates: str) -> list:
    """Read a column from a TSV, trying multiple candidate column names."""
    df = _df(snap_dir, tsv_name)
    for col in col_candidates:
        if col in df.columns:
            return df[col].tolist()
    raise KeyError(
        f"None of {col_candidates} found in {tsv_name}.tsv; "
        f"available: {df.columns.tolist()}"
    )


def _read_meta_int(snap_dir: Path, stem: str) -> int:
    """Read an integer from <stem>.tsv or <stem>.txt."""
    for ext in ("tsv", "txt"):
        p = snap_dir / f"{stem}.{ext}"
        if p.exists():
            return int(p.read_text().strip())
    raise FileNotFoundError(f"{stem}.tsv/.txt not found in {snap_dir}")


def _align_axes(mat, src_rows, src_cols, dst_rows, dst_cols):
    """Reorder ``mat`` (rows=src_rows, cols=src_cols) to (dst_rows, dst_cols)."""
    if src_rows == dst_rows and src_cols == dst_cols:
        return mat
    row_map = {n: i for i, n in enumerate(src_rows)}
    col_map = {n: i for i, n in enumerate(src_cols)}
    r_idx = np.array([row_map[n] for n in dst_rows])
    c_idx = np.array([col_map[n] for n in dst_cols])
    return mat[r_idx, :][:, c_idx]


def _triples(B_prob, L_prob, tf_names, gene_names, peak_ids, tau_b, tau_l):
    triples: set[tuple[str, int, str]] = set()
    for peak_i, gene_i in zip(*np.nonzero(L_prob > tau_l)):
        tf_idx = np.where(B_prob[peak_i, :] > tau_b)[0]
        if tf_idx.size == 0:
            continue
        pid = int(peak_ids[peak_i])
        g = gene_names[gene_i]
        for j in tf_idx:
            triples.add((g, pid, tf_names[j]))
    return triples


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main() -> None:
    missing = [(n, p) for n, p in RUNS.items()
               if not p.exists() or not any(p.glob("*.tsv"))]
    if missing:
        print("WARNING: missing snapshots for the following runs — they will be skipped:")
        for n, p in missing:
            print(f"  {n!r}: {p}")

    active_runs = {n: p for n, p in RUNS.items()
                   if p.exists() and any(p.glob("*.tsv"))}
    if len(active_runs) < 2:
        raise SystemExit("Need at least 2 snapshot directories to compare.")

    print(f"Loading snapshots from {len(active_runs)} runs...")

    # First pass: load all index sets and find the common intersection.
    all_peaks: dict[str, list] = {}
    all_genes: dict[str, list] = {}
    all_tfs:   dict[str, list] = {}
    n_iters: set[int] = set()
    for name, path in active_runs.items():
        all_peaks[name] = [int(x) for x in _read_col(path, "Peaks", "Peak_index", "Peak")]
        all_genes[name] = _read_col(path, "Genes", "Gene_symbols", "Gene")
        all_tfs[name]   = _read_col(path, "TFs",   "name", "TF")
        n_iters.add(_read_meta_int(path, "n_iter"))

    # Use intersection so a TF/peak/gene absent in any run is excluded.
    ref_peaks = sorted(set.intersection(*[set(v) for v in all_peaks.values()]),
                       key=lambda x: all_peaks[next(iter(active_runs))].index(x)
                       if x in all_peaks[next(iter(active_runs))] else 0)
    ref_genes = sorted(set.intersection(*[set(v) for v in all_genes.values()]),
                       key=lambda x: all_genes[next(iter(active_runs))].index(x)
                       if x in all_genes[next(iter(active_runs))] else 0)
    ref_tfs   = sorted(set.intersection(*[set(v) for v in all_tfs.values()]),
                       key=lambda x: all_tfs[next(iter(active_runs))].index(x)
                       if x in all_tfs[next(iter(active_runs))] else 0)

    print(f"  Common peaks={len(ref_peaks)}, genes={len(ref_genes)}, TFs={len(ref_tfs)}")
    for name in active_runs:
        only = set(all_tfs[name]) - set(ref_tfs)
        if only:
            print(f"  {name}: {len(only)} TFs not in intersection: {sorted(only)[:5]}...")

    # Second pass: load and align posteriors to the common index sets.
    posteriors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    noise: dict[str, np.ndarray] = {}
    for name, path in active_runs.items():
        B = _align_axes(_mat(path, "TF_Peak_Binding_prob"),
                        all_peaks[name], all_tfs[name], ref_peaks, ref_tfs)
        L = _align_axes(_mat(path, "Peak_Gene_Looping_prob"),
                        all_peaks[name], all_genes[name], ref_peaks, ref_genes)
        posteriors[name] = (B, L)
        noise[name] = _mat(path, "Noise_parameters")

    assert len(n_iters) == 1, f"Snapshots have mixed n_iter: {n_iters}"
    n_iter = next(iter(n_iters))
    print(f"  All {len(active_runs)} runs used n_iter = {n_iter}")

    # ---- Compute triples for each run ----
    triples_per_run: dict[str, set] = {}
    for name, (B, L) in posteriors.items():
        triples_per_run[name] = _triples(
            B, L, ref_tfs, ref_genes, ref_peaks, PROB_TF_PEAK, PROB_PEAK_GENE
        )
        print(f"  {name:22s}  |circuits| = {len(triples_per_run[name]):6d}")

    # ---- Jaccard matrix ----
    names = list(active_runs.keys())
    n = len(names)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = _jacc(triples_per_run[names[i]], triples_per_run[names[j]])
            J[i, j] = J[j, i] = v

    print(f"\nTriple-Jaccard matrix (n_iter={n_iter}, "
          f"thresholds {PROB_TF_PEAK}/{PROB_PEAK_GENE}):")
    print(pd.DataFrame(J, index=names, columns=names).round(4).to_string())

    # ---- Posterior Pearson-r matrices ----
    Pr_B = np.eye(n); Pr_L = np.eye(n)
    for i in range(n):
        Bi, Li = posteriors[names[i]]
        for j in range(i + 1, n):
            Bj, Lj = posteriors[names[j]]
            Pr_B[i, j] = Pr_B[j, i] = float(np.corrcoef(Bi.ravel(), Bj.ravel())[0, 1])
            Pr_L[i, j] = Pr_L[j, i] = float(np.corrcoef(Li.ravel(), Lj.ravel())[0, 1])

    # ---- PLOT 1: triple-Jaccard heatmap ----
    fs = max(7, 10 - n)  # shrink font for larger matrices
    fig_sz = max(7.5, 1.2 * n)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz - 0.5))
    im_ = ax.imshow(J, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=fs)
    ax.set_yticklabels(names, fontsize=fs)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{J[i, j]:.3f}", ha="center", va="center",
                    color="white" if J[i, j] < 0.85 else "black", fontsize=fs)
    fig.colorbar(im_, ax=ax, label="Jaccard on TF-Peak-Gene triples")
    npairs = n * (n - 1) // 2
    ax.set_title(
        f"Selected-circuit agreement across {n} runs (n_iter={n_iter})\n"
        f"{npairs} unique pairs; same-method pairs measure intrinsic MCMC noise"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_jaccard_matrix.png", dpi=150)
    print("  wrote plots/variability_jaccard_matrix.png")

    # ---- PLOT 2: intra vs cross bars grouped by method pair ----
    # Build group-level summaries
    groups = sorted(set(RUN_GROUP.get(nm, nm) for nm in names))
    intra: dict[str, list[float]] = {g: [] for g in groups}
    cross_pairs: dict[tuple[str, str], list[float]] = {}
    pair_indices_all = list(combinations(range(n), 2))
    for a, b in pair_indices_all:
        ga, gb = RUN_GROUP.get(names[a], names[a]), RUN_GROUP.get(names[b], names[b])
        if ga == gb:
            intra[ga].append(J[a, b])
        else:
            key = (min(ga, gb), max(ga, gb))
            cross_pairs.setdefault(key, []).append(J[a, b])

    bar_positions, bar_heights, bar_colors, bar_xlabels, bar_errs = [], [], [], [], []
    # intra bars
    for g in groups:
        vals_g = intra.get(g, [])
        if vals_g:
            bar_positions.append(len(bar_positions))
            bar_heights.append(float(np.mean(vals_g)))
            bar_errs.append(float(np.std(vals_g)) if len(vals_g) > 1 else 0)
            bar_colors.append(GROUP_COLORS.get(g, "#999999"))
            bar_xlabels.append(f"{g} vs {g}\n(intra,\ndiff seeds)")
    # separator gap
    gap = len(bar_positions)
    # cross bars
    for (ga, gb), vals_c in sorted(cross_pairs.items()):
        bar_positions.append(len(bar_positions) + 0.5)
        bar_heights.append(float(np.mean(vals_c)))
        bar_errs.append(float(np.std(vals_c)) if len(vals_c) > 1 else 0)
        # blend: mix of colors (use lighter)
        bar_colors.append("#7FA453")
        bar_xlabels.append(f"{ga} vs {gb}\n(cross)")

    xs_arr = np.array(bar_positions, dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(bar_positions)), 5.5))
    bars2 = ax.bar(xs_arr, bar_heights, color=bar_colors, width=0.7,
                   yerr=bar_errs, capsize=4, error_kw={"linewidth": 1.5})
    for xi, v, e in zip(xs_arr, bar_heights, bar_errs):
        label = f"{v:.3f}"
        if e > 0:
            label += f"\n±{e:.3f}"
        ax.text(xi, v + e + 0.008, label, ha="center", va="bottom", fontsize=10)
    # scatter individual cross-pair values
    for k_i, ((ga, gb), vals_c) in enumerate(sorted(cross_pairs.items())):
        xi = xs_arr[gap + k_i]
        jitter = np.linspace(-0.15, 0.15, len(vals_c))
        ax.scatter(xi + jitter, vals_c, color="k", zorder=5, s=25)
    ax.set_xticks(xs_arr)
    ax.set_xticklabels(bar_xlabels, fontsize=9)
    ax.set_ylabel("Triple Jaccard")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"Intra- vs cross-method circuit agreement (n_iter={n_iter})\n"
        f"Error bars = SD over pairs; if cross ≈ intra, disagreement is MCMC noise"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_same_vs_cross.png", dpi=150)
    print("  wrote plots/variability_same_vs_cross.png")

    # ---- PLOT 3: per-component agreement, averaged by pair type ----
    def _components(trip: set) -> dict[str, set]:
        return {
            "TFs":       {t[2] for t in trip},
            "Peaks":     {t[1] for t in trip},
            "Genes":     {t[0] for t in trip},
            "Peak-Gene": {(t[0], t[1]) for t in trip},
            "TF-Peak":   {(t[1], t[2]) for t in trip},
            "Triples":   trip,
        }
    comps_per_run = {n_: _components(t) for n_, t in triples_per_run.items()}
    keys = ["TFs", "Peaks", "Genes", "Peak-Gene", "TF-Peak", "Triples"]

    pair_indices = list(combinations(range(n), 2))

    # Group pairs by type: intra-R, intra-Py, intra-C++, R-Py, R-C++, Py-C++
    def _pair_type(a: int, b: int) -> str:
        ga = RUN_GROUP.get(names[a], names[a])
        gb = RUN_GROUP.get(names[b], names[b])
        if ga == gb:
            return f"intra-{ga}"
        return "\u2194".join(sorted([ga, gb]))

    pair_type_vals: dict[str, dict[str, list[float]]] = {}
    for a, b in pair_indices:
        pt = _pair_type(a, b)
        for k in keys:
            pair_type_vals.setdefault(pt, {}).setdefault(k, []).append(
                _jacc(comps_per_run[names[a]][k], comps_per_run[names[b]][k])
            )

    # sort: intra first, then cross
    intra_pts  = sorted(pt for pt in pair_type_vals if pt.startswith("intra"))
    cross_pts  = sorted(pt for pt in pair_type_vals if not pt.startswith("intra"))
    pair_types = intra_pts + cross_pts

    xs3 = np.arange(len(keys))
    w3 = 0.8 / len(pair_types)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for pt_i, pt in enumerate(pair_types):
        is_intra = pt.startswith("intra")
        grp = pt.replace("intra-", "")
        if is_intra:
            color = GROUP_COLORS.get(grp, "#888888")
        else:
            color = CROSS_COLORS.get(pt, "#888888")
        vals3  = [float(np.mean(pair_type_vals[pt][k])) for k in keys]
        offset = (pt_i - (len(pair_types) - 1) / 2) * w3
        ax.bar(xs3 + offset, vals3, w3, label=pt, color=color,
               alpha=0.85,
               hatch="///" if not is_intra else "",
               edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs3)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylabel("Mean Jaccard (over pairs in group)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-component agreement by pair type (n_iter={n_iter})\n"
        f"Solid = intra-method (MCMC noise floor); striped = cross-method"
    )
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_components.png", dpi=150)
    print("  wrote plots/variability_components.png")

    # ---- PLOT 4: posterior Pearson-r heatmap (2 panels: B and L) ----
    fs4 = max(6, 9 - n)
    fig_sz4 = max(13, 2 * n)
    fig, axs = plt.subplots(1, 2, figsize=(fig_sz4, fig_sz4 // 2 + 1))
    for ax, M, title in [
        (axs[0], Pr_B, "TF-Peak posterior (B) Pearson r"),
        (axs[1], Pr_L, "Peak-Gene posterior (L) Pearson r"),
    ]:
        im_ = ax.imshow(M, vmin=0.9, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=fs4)
        ax.set_yticklabels(names, fontsize=fs4)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.97 else "black", fontsize=fs4)
        fig.colorbar(im_, ax=ax, label="Pearson r")
        ax.set_title(title)
    fig.suptitle(
        f"Raw posterior agreement across {n} runs (n_iter={n_iter}) — "
        f"before thresholding into circuits"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_posterior_corr.png", dpi=150)
    print("  wrote plots/variability_posterior_corr.png")

    # ---- Metrics TSV ----
    rows = []
    for a, b in pair_indices:
        pair = f"{names[a]}  <->  {names[b]}"
        kind = _pair_type(a, b)
        row = {
            "pair": pair, "kind": kind,
            "triple_jaccard": J[a, b],
            "triples_a": len(triples_per_run[names[a]]),
            "triples_b": len(triples_per_run[names[b]]),
            "triples_shared": len(triples_per_run[names[a]] & triples_per_run[names[b]]),
            "B_pearson": Pr_B[a, b],
            "L_pearson": Pr_L[a, b],
        }
        for k in keys:
            row[f"jacc_{k.replace('-','_')}"] = _jacc(
                comps_per_run[names[a]][k], comps_per_run[names[b]][k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_TSV, sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote {METRICS_TSV.relative_to(REPO_ROOT)}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))

    # ---- Summary ----
    print(f"\nSummary (n_iter={n_iter}):")
    for pt in intra_pts:
        js  = [J[a, b]    for a, b in pair_indices if _pair_type(a, b) == pt]
        brs = [Pr_B[a, b] for a, b in pair_indices if _pair_type(a, b) == pt]
        lrs = [Pr_L[a, b] for a, b in pair_indices if _pair_type(a, b) == pt]
        print(f"  {pt:18s} triple J: {float(np.mean(js)):.4f}  "
              f"B r={float(np.mean(brs)):.4f}  L r={float(np.mean(lrs)):.4f}")
    for pt in cross_pts:
        js = [J[a, b] for a, b in pair_indices if _pair_type(a, b) == pt]
        gs = [g.strip() for g in pt.split("\u2194")]
        intra_js = []
        for g in gs:
            g_js = [J[a, b] for a, b in pair_indices
                    if _pair_type(a, b) == f"intra-{g}"]
            if g_js:
                intra_js.append(float(np.mean(g_js)))
        delta = float(np.mean(js)) - (float(np.mean(intra_js)) if intra_js else 0.0)
        print(f"  {pt:18s} triple J: {float(np.mean(js)):.4f}  "
              f"(range {min(js):.4f}-{max(js):.4f})  delta={delta:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
