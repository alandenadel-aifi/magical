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

Renders to ``re-implementation/benchmarks/plots/``:
- ``variability_jaccard_matrix.png``   full 4x4 heatmap (6 unique pairs)
- ``variability_same_vs_cross.png``    intra R vs intra Py vs cross bars
- ``variability_components.png``       per-component Jaccards per pair
- ``variability_posterior_corr.png``   Pearson-r of raw posteriors, 4x4
Writes ``re-implementation/benchmarks/variability_metrics.tsv``.

Interpretation: if intra-language Jaccard (R vs R, Py vs Py) is close to
cross-language Jaccard (R vs Py), the disagreement is intrinsic MCMC noise,
not an implementation bug.

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

# The four runs we compare (default to the 1000-iter runs)
RUNS: dict[str, Path] = {
    "R  seed 20260723": REPO_ROOT / "tests" / "snapshots_demo_R_1000iter_20260723",
    "R  seed 42":       REPO_ROOT / "tests" / "snapshots_demo_R_1000iter_42",
    "Py seed 20260723": REPO_ROOT / "tests" / "snapshots_demo_Py_1000iter_20260723",
    "Py seed 42":       REPO_ROOT / "tests" / "snapshots_demo_Py_1000iter_42",
}


def _mat(snap_dir: Path, name: str) -> np.ndarray:
    return np.loadtxt(snap_dir / f"{name}.tsv", delimiter="\t")


def _df(snap_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(snap_dir / f"{name}.tsv", sep="\t")


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
    for name, path in RUNS.items():
        if not path.exists() or not any(path.glob("*.tsv")):
            raise SystemExit(f"Missing snapshots for {name!r}: {path}")

    print("Loading snapshots from all 4 runs...")
    # Use R seed 20260723 as canonical row/col order; align everything to it
    canonical = RUNS["R  seed 20260723"]
    ref_peaks = _df(canonical, "Peaks")["Peak_index"].astype(int).tolist()
    ref_genes = _df(canonical, "Genes")["Gene_symbols"].tolist()
    ref_tfs = _df(canonical, "TFs")["name"].tolist()

    n_iters: set[int] = set()
    posteriors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    noise: dict[str, np.ndarray] = {}
    for name, path in RUNS.items():
        peaks = _df(path, "Peaks")["Peak_index"].astype(int).tolist()
        genes = _df(path, "Genes")["Gene_symbols"].tolist()
        tfs = _df(path, "TFs")["name"].tolist()
        B = _align_axes(_mat(path, "TF_Peak_Binding_prob"),
                        peaks, tfs, ref_peaks, ref_tfs)
        L = _align_axes(_mat(path, "Peak_Gene_Looping_prob"),
                        peaks, genes, ref_peaks, ref_genes)
        posteriors[name] = (B, L)
        noise[name] = _mat(path, "Noise_parameters")
        n_iters.add(int((path / "n_iter.tsv").read_text().strip()))

    assert len(n_iters) == 1, f"Snapshots have mixed n_iter: {n_iters}"
    n_iter = next(iter(n_iters))
    print(f"  All four runs used n_iter = {n_iter}")

    # ---- Compute triples for each run ----
    triples_per_run: dict[str, set] = {}
    for name, (B, L) in posteriors.items():
        triples_per_run[name] = _triples(
            B, L, ref_tfs, ref_genes, ref_peaks, PROB_TF_PEAK, PROB_PEAK_GENE
        )
        print(f"  {name:20s}  |circuits| = {len(triples_per_run[name]):6d}")

    # ---- 4x4 triple-Jaccard matrix ----
    names = list(RUNS.keys())
    n = len(names)
    J = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = _jacc(triples_per_run[names[i]], triples_per_run[names[j]])
            J[i, j] = J[j, i] = v

    print(f"\nTriple-Jaccard matrix (n_iter={n_iter}, "
          f"thresholds {PROB_TF_PEAK}/{PROB_PEAK_GENE}):")
    print(pd.DataFrame(J, index=names, columns=names).round(4).to_string())

    # ---- 4x4 posterior Pearson-r matrix (mean of B and L flat) ----
    Pr_B = np.eye(n); Pr_L = np.eye(n)
    for i in range(n):
        Bi, Li = posteriors[names[i]]
        for j in range(i + 1, n):
            Bj, Lj = posteriors[names[j]]
            Pr_B[i, j] = Pr_B[j, i] = float(np.corrcoef(Bi.ravel(), Bj.ravel())[0, 1])
            Pr_L[i, j] = Pr_L[j, i] = float(np.corrcoef(Li.ravel(), Lj.ravel())[0, 1])

    # ---- PLOT 1: triple-Jaccard heatmap ----
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im_ = ax.imshow(J, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{J[i, j]:.3f}", ha="center", va="center",
                    color="white" if J[i, j] < 0.85 else "black", fontsize=10)
    fig.colorbar(im_, ax=ax, label="Jaccard on TF-Peak-Gene triples")
    ax.set_title(
        f"Selected-circuit agreement across 4 runs (n_iter={n_iter})\n"
        f"6 unique pairs; same-language pairs measure intrinsic MCMC noise"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_jaccard_matrix.png", dpi=150)
    print("  wrote plots/variability_jaccard_matrix.png")

    # ---- PLOT 2: same-language vs cross-language bars ----
    same_r = J[0, 1]      # R1 vs R2
    same_py = J[2, 3]     # Py1 vs Py2
    cross = [J[0, 2], J[0, 3], J[1, 2], J[1, 3]]
    cross_mean = float(np.mean(cross))
    cross_labels = [f"{names[a]} \u2194 {names[b]}"
                    for a, b in [(0, 2), (0, 3), (1, 2), (1, 3)]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bar_labels = ["R vs R\n(same code,\ndifferent seeds)",
                  "Py vs Py\n(same code,\ndifferent seeds)",
                  "R vs Py\n(mean over 4 pairs)"]
    vals = [same_r, same_py, cross_mean]
    colors = ["#E24A33", "#348ABD", "#7FA453"]
    ax.bar(bar_labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    # scatter individual cross pairs
    xs = np.full(len(cross), 2.0) + np.linspace(-0.15, 0.15, len(cross))
    ax.scatter(xs, cross, color="k", zorder=5, s=30,
               label="individual cross-language pairs")
    for xi, ci, lbl in zip(xs, cross, cross_labels):
        ax.text(xi, ci + 0.005, f"{ci:.3f}", ha="center", fontsize=7, rotation=0)
    ax.set_ylabel("Triple Jaccard")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Intra- vs cross-language circuit agreement (n_iter={n_iter})\n"
        f"If green \u2248 red/blue, disagreement is MCMC noise, not a bug"
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_same_vs_cross.png", dpi=150)
    print("  wrote plots/variability_same_vs_cross.png")

    # ---- PLOT 3: per-component agreement across the 6 pairs ----
    def _components(trip: set) -> dict[str, set]:
        return {
            "TFs":       {t[2] for t in trip},
            "Peaks":     {t[1] for t in trip},
            "Genes":     {t[0] for t in trip},
            "Peak-Gene": {(t[0], t[1]) for t in trip},
            "TF-Peak":   {(t[1], t[2]) for t in trip},
            "triples":   trip,
        }
    comps_per_run = {n_: _components(t) for n_, t in triples_per_run.items()}
    keys = ["TFs", "Peaks", "Genes", "Peak-Gene", "TF-Peak", "triples"]

    pair_indices = list(combinations(range(n), 2))
    pair_labels = [f"{names[a].split()[0]}{a}\n\u2194\n{names[b].split()[0]}{b}"
                   for a, b in pair_indices]
    same_lang_mask = [(a < 2) == (b < 2) for a, b in pair_indices]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = np.arange(len(keys))
    w = 0.12
    for k_i, (a, b) in enumerate(pair_indices):
        vals = [_jacc(comps_per_run[names[a]][k], comps_per_run[names[b]][k]) for k in keys]
        color = "#E24A33" if same_lang_mask[k_i] and a == 0 else \
                "#348ABD" if same_lang_mask[k_i] and a == 2 else "#7FA453"
        offset = (k_i - (len(pair_indices) - 1) / 2) * w
        ax.bar(xs + offset, vals, w,
               label=f"{names[a]} \u2194 {names[b]}", color=color, alpha=0.85,
               edgecolor="black" if same_lang_mask[k_i] else "none",
               linewidth=1.2 if same_lang_mask[k_i] else 0)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-component agreement across all 6 run pairs (n_iter={n_iter})\n"
        f"Black-edged bars = same-language pairs (intra-MCMC noise); "
        f"green = cross-language"
    )
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_components.png", dpi=150)
    print("  wrote plots/variability_components.png")

    # ---- PLOT 4: posterior Pearson-r heatmap (2 panels: B and L) ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, M, title in [
        (axs[0], Pr_B, "TF-Peak posterior (B) Pearson r"),
        (axs[1], Pr_L, "Peak-Gene posterior (L) Pearson r"),
    ]:
        im_ = ax.imshow(M, vmin=0.9, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.97 else "black", fontsize=8)
        fig.colorbar(im_, ax=ax, label="Pearson r")
        ax.set_title(title)
    fig.suptitle(
        f"Raw posterior agreement across 4 runs (n_iter={n_iter}) — "
        f"before thresholding into circuits"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "variability_posterior_corr.png", dpi=150)
    print("  wrote plots/variability_posterior_corr.png")

    # ---- Metrics TSV ----
    rows = []
    for a, b in pair_indices:
        pair = f"{names[a]}  <->  {names[b]}"
        kind = "intra" if same_lang_mask[pair_indices.index((a, b))] else "cross"
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
            row[f"jacc_{k}"] = _jacc(comps_per_run[names[a]][k],
                                     comps_per_run[names[b]][k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_TSV, sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote {METRICS_TSV.relative_to(REPO_ROOT)}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))

    # ---- Summary ----
    print(f"\nSummary (n_iter={n_iter}):")
    print(f"  R  vs R  (intra) triple J:  {same_r:.4f}   B r={Pr_B[0,1]:.4f}   L r={Pr_L[0,1]:.4f}")
    print(f"  Py vs Py (intra) triple J:  {same_py:.4f}   B r={Pr_B[2,3]:.4f}   L r={Pr_L[2,3]:.4f}")
    print(f"  R  vs Py (cross) triple J:  {cross_mean:.4f}   "
          f"(range {min(cross):.4f}-{max(cross):.4f})")
    delta = cross_mean - (same_r + same_py) / 2
    print(f"  Delta (cross - mean intra): {delta:+.4f}  "
          f"({'\u2248 pure MCMC noise' if abs(delta) < 0.02 else 'suggests residual bias'})")

    print("\nDone.")


if __name__ == "__main__":
    main()
