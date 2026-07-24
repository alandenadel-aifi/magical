"""Parity plots + metrics comparing Python vs R MAGICAL outputs on the demo.

Compares all four MAGICAL outputs qualitatively (plots) and quantitatively
(metrics printed + emitted to a metrics TSV):

1. ``TF_Peak_Binding_prob``   (P x M)     posterior probs, RNG-dependent
2. ``Peak_Gene_Looping_prob`` (P x G)     posterior probs, RNG-dependent
3. ``Noise_parameters``       (n_iter x 2) MCMC traces
4. Selected regulatory circuits           TF-Peak-Gene triples above thresholds

The circuits comparison includes:
- overall triple-set Jaccard / precision / recall
- per-component sets: TFs, peaks, genes selected by each side (Jaccard bar)
- degree-distribution histograms (# TFs per gene, # peaks per gene)
- top-N TF ranking overlap (by how many circuits each TF participates in)

Reads R snapshots from ``tests/snapshots_demo/`` (produced by
``re-implementation/scripts/gen_demo_parity_snapshots.R``), re-runs the
Python pipeline at ``n_iter`` from the snapshot, and writes PNGs + metrics
to ``re-implementation/benchmarks/plots/``.

Usage::

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run python \
        re-implementation/scripts/plot_parity.py
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magical import estimation, magical_initialization
from magical.circuits import candidate_circuits_construction_with_TAD
from magical.io import data_loading

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = REPO_ROOT / "Demo input files"
SNAP = REPO_ROOT / "tests" / "snapshots_demo"
OUT_DIR = REPO_ROOT / "re-implementation" / "benchmarks" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_TSV = REPO_ROOT / "re-implementation" / "benchmarks" / "parity_metrics.tsv"

PROB_TF_PEAK = 0.8
PROB_PEAK_GENE = 0.95


# ------------------------- IO helpers -------------------------

def _mat(name: str) -> np.ndarray:
    return np.loadtxt(SNAP / f"{name}.tsv", delimiter="\t")


def _df(name: str) -> pd.DataFrame:
    return pd.read_csv(SNAP / f"{name}.tsv", sep="\t")


def _align(py_mat, py_row_names, py_col_names, r_row_names, r_col_names):
    row_map = {n: i for i, n in enumerate(py_row_names)}
    col_map = {n: i for i, n in enumerate(py_col_names)}
    row_idx = np.array([row_map[n] for n in r_row_names])
    col_idx = np.array([col_map[n] for n in r_col_names])
    return py_mat[row_idx, :][:, col_idx]


# ------------------------- Plot primitives -------------------------

def _scatter(ax, x, y, title, xlabel="R", ylabel="Python (numpy)"):
    ax.scatter(x.ravel(), y.ravel(), s=2, alpha=0.25, edgecolors="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "r-", lw=1, label="y = x")
    corr = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    ax.set_title(f"{title}\nPearson r = {corr:.4f}, n = {x.size:,}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)


def _hist_overlay(ax, r_vals, py_vals, title, bins=50):
    ax.hist(r_vals.ravel(), bins=bins, alpha=0.5, label="R", color="#E24A33")
    ax.hist(py_vals.ravel(), bins=bins, alpha=0.5, label="Python", color="#348ABD")
    ax.set_yscale("log")
    ax.set_xlabel("posterior probability")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    ax.legend(fontsize=8)


# ------------------------- Circuits helpers -------------------------

def _triples(B_prob, L_prob, tf_names, gene_names, peak_ids, tau_b, tau_l):
    """Return set of (gene, peak_id, tf) triples selected at these thresholds."""
    triples: set[tuple[str, int, str]] = set()
    peak_gene_mask = L_prob > tau_l
    for peak_i, gene_i in zip(*np.nonzero(peak_gene_mask)):
        tf_idx = np.where(B_prob[peak_i, :] > tau_b)[0]
        if tf_idx.size == 0:
            continue
        peak_id = peak_ids[peak_i]
        gene = gene_names[gene_i]
        for j in tf_idx:
            triples.add((gene, int(peak_id), tf_names[j]))
    return triples


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _run_python_pipeline(n_iter, seed=20260723):
    d = DEMO_DIR
    times: dict[str, float] = {}
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
    times["data_loading"] = time.perf_counter() - t
    t = time.perf_counter()
    cand = candidate_circuits_construction_with_TAD(
        loaded, d / "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt",
    )
    times["candidate_circuits"] = time.perf_counter() - t
    t = time.perf_counter()
    im = magical_initialization(loaded.as_loaded_data(), cand)
    times["initialization"] = time.perf_counter() - t
    t = time.perf_counter()
    result = estimation(
        loaded.as_loaded_data(), cand, im,
        iteration_num=n_iter, backend="numpy", seed=seed,
    )
    times[f"estimation_{n_iter}iter"] = time.perf_counter() - t
    return loaded, cand, im, result, times


# ------------------------- Main -------------------------

def main() -> None:
    if not any(SNAP.glob("*.tsv")):
        raise SystemExit(
            "No R snapshots found. Run "
            "`Rscript re-implementation/scripts/gen_demo_parity_snapshots.R` first."
        )

    print("Loading R snapshots...")
    r_peaks = _df("Peaks")["Peak_index"].astype(int).tolist()
    r_genes = _df("Genes")["Gene_symbols"].tolist()
    r_tfs = _df("TFs")["name"].tolist()
    r_B_prior = _mat("B_prior")
    r_L_prior = _mat("L_prior")
    r_B_post = _mat("TF_Peak_Binding_prob")
    r_L_post = _mat("Peak_Gene_Looping_prob")
    r_noise = _mat("Noise_parameters")
    n_iter = int((SNAP / "n_iter.tsv").read_text().strip())

    print(f"Running Python pipeline @ {n_iter} iterations...")
    _loaded, cand, im, result, py_times = _run_python_pipeline(n_iter)
    for k, v in py_times.items():
        print(f"  {k:22s} {v:8.2f} s")

    py_peaks = cand.Peaks["Peak_index"].astype(int).tolist()
    py_genes = cand.Genes["Gene_symbols"].tolist()
    py_tfs = cand.TFs["name"].tolist()

    py_B_prior = _align(im.B_prior, py_peaks, py_tfs, r_peaks, r_tfs)
    py_L_prior = _align(im.L_prior, py_peaks, py_genes, r_peaks, r_genes)
    py_B_post = _align(result.TF_Peak_Binding_prob, py_peaks, py_tfs, r_peaks, r_tfs)
    py_L_post = _align(result.Peak_Gene_Looping_prob, py_peaks, py_genes, r_peaks, r_genes)
    py_noise = result.Noise_parameters

    # ---- Plot 1: init parity (deterministic sanity check) ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    _scatter(axs[0], r_B_prior, py_B_prior, "B_prior (TF-Peak, deterministic)")
    _scatter(axs[1], r_L_prior, py_L_prior, "L_prior (Peak-Gene, deterministic)")
    fig.suptitle("Initialization parity — Python numpy vs R (demo dataset)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_initialization.png", dpi=150)
    print("  wrote plots/parity_initialization.png")

    # ---- Plot 2: posteriors scatter + histograms ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    _scatter(axs[0, 0], r_B_post, py_B_post,
             f"TF-Peak posterior (Gibbs, {n_iter} iter)")
    _scatter(axs[0, 1], r_L_post, py_L_post,
             f"Peak-Gene posterior (Gibbs, {n_iter} iter)")
    _hist_overlay(axs[1, 0], r_B_post, py_B_post, "TF-Peak posterior distribution")
    _hist_overlay(axs[1, 1], r_L_post, py_L_post, "Peak-Gene posterior distribution")
    fig.suptitle("Posterior parity — Python numpy vs R (seed 20260723)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_posteriors.png", dpi=150)
    print("  wrote plots/parity_posteriors.png")

    # ---- Plot 3: noise-parameter traces ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(1, n_iter + 1)
    axs[0].plot(xs, r_noise[:, 0], label="R", color="#E24A33", lw=1)
    axs[0].plot(xs, py_noise[:, 0], label="Python", color="#348ABD", lw=1, alpha=0.7)
    axs[0].set_title(r"$\sigma^2_{A,\ noise}$ trace")
    axs[0].set_xlabel("Gibbs iteration"); axs[0].set_ylabel("noise variance"); axs[0].legend()
    axs[1].plot(xs, r_noise[:, 1], label="R", color="#E24A33", lw=1)
    axs[1].plot(xs, py_noise[:, 1], label="Python", color="#348ABD", lw=1, alpha=0.7)
    axs[1].set_title(r"$\sigma^2_{R,\ noise}$ trace")
    axs[1].set_xlabel("Gibbs iteration"); axs[1].set_ylabel("noise variance"); axs[1].legend()
    fig.suptitle("Noise-parameter MCMC traces")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_noise_trace.png", dpi=150)
    print("  wrote plots/parity_noise_trace.png")

    # ---- Circuits comparison ----
    r_trip = _triples(r_B_post, r_L_post, r_tfs, r_genes, r_peaks,
                      PROB_TF_PEAK, PROB_PEAK_GENE)
    py_trip = _triples(py_B_post, py_L_post, r_tfs, r_genes, r_peaks,
                       PROB_TF_PEAK, PROB_PEAK_GENE)
    inter = r_trip & py_trip
    union = r_trip | py_trip
    j_trip = len(inter) / len(union) if union else 1.0
    prec = len(inter) / len(py_trip) if py_trip else float("nan")
    rec = len(inter) / len(r_trip) if r_trip else float("nan")

    # Per-component sets
    r_tf_set = {t[2] for t in r_trip}
    py_tf_set = {t[2] for t in py_trip}
    r_peak_set = {t[1] for t in r_trip}
    py_peak_set = {t[1] for t in py_trip}
    r_gene_set = {t[0] for t in r_trip}
    py_gene_set = {t[0] for t in py_trip}
    r_pg_pairs = {(t[0], t[1]) for t in r_trip}
    py_pg_pairs = {(t[0], t[1]) for t in py_trip}
    r_tp_pairs = {(t[1], t[2]) for t in r_trip}
    py_tp_pairs = {(t[1], t[2]) for t in py_trip}

    print(f"\nSelected circuits @ thresholds "
          f"(TF-Peak>{PROB_TF_PEAK}, Peak-Gene>{PROB_PEAK_GENE}):")
    print(f"  R triples:      {len(r_trip):6d}")
    print(f"  Python triples: {len(py_trip):6d}")
    print(f"  intersection:   {len(inter):6d}")
    print(f"  Jaccard {j_trip:.4f}  precision {prec:.4f}  recall {rec:.4f}")
    print(f"  Component Jaccards: TFs={_jaccard(r_tf_set, py_tf_set):.3f}  "
          f"Peaks={_jaccard(r_peak_set, py_peak_set):.3f}  "
          f"Genes={_jaccard(r_gene_set, py_gene_set):.3f}  "
          f"Peak-Gene pairs={_jaccard(r_pg_pairs, py_pg_pairs):.3f}  "
          f"TF-Peak pairs={_jaccard(r_tp_pairs, py_tp_pairs):.3f}")

    # ---- Plot 4a: circuit overlap (triples + components) ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(
        ["R only", "shared", "Python only"],
        [len(r_trip - py_trip), len(inter), len(py_trip - r_trip)],
        color=["#E24A33", "#7FA453", "#348ABD"],
    )
    for i, c in enumerate([len(r_trip - py_trip), len(inter), len(py_trip - r_trip)]):
        axs[0].text(i, c, str(c), ha="center", va="bottom", fontsize=10)
    axs[0].set_ylabel("# TF-Peak-Gene triples")
    axs[0].set_title(
        f"Selected circuits (triples)\n"
        f"Jaccard = {j_trip:.3f}, precision = {prec:.3f}, recall = {rec:.3f}"
    )

    components = [
        ("TFs", r_tf_set, py_tf_set),
        ("Peaks", r_peak_set, py_peak_set),
        ("Genes", r_gene_set, py_gene_set),
        ("Peak-Gene\npairs", r_pg_pairs, py_pg_pairs),
        ("TF-Peak\npairs", r_tp_pairs, py_tp_pairs),
    ]
    x = np.arange(len(components))
    w = 0.28
    r_counts = [len(r) for _, r, _ in components]
    py_counts = [len(p) for _, _, p in components]
    inter_counts = [len(r & p) for _, r, p in components]
    axs[1].bar(x - w, r_counts, w, label="R", color="#E24A33")
    axs[1].bar(x, py_counts, w, label="Python", color="#348ABD")
    axs[1].bar(x + w, inter_counts, w, label="shared", color="#7FA453")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([c[0] for c in components], fontsize=9)
    axs[1].set_ylabel("count")
    axs[1].set_title("Selected-circuit components (unique element counts + overlap)")
    for xi, (r, p, i_) in enumerate(zip(r_counts, py_counts, inter_counts)):
        j = i_ / (len(components[xi][1] | components[xi][2])) if (components[xi][1] | components[xi][2]) else 1.0
        axs[1].text(xi, max(r, p) * 1.02, f"J={j:.2f}", ha="center", fontsize=8, color="gray")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_circuits.png", dpi=150)
    print("  wrote plots/parity_circuits.png")

    # ---- Plot 4b: circuit degree distributions ----
    def _degree(trip, key_i):
        c = Counter()
        for t in trip:
            c[t[key_i]] += 1
        return list(c.values())

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key_i, singular, plural in [
        (axs[0], 2, "TF", "TFs"),
        (axs[1], 0, "gene", "genes"),
        (axs[2], 1, "peak", "peaks"),
    ]:
        r_deg = _degree(r_trip, key_i)
        py_deg = _degree(py_trip, key_i)
        all_deg = r_deg + py_deg
        if all_deg:
            bins = np.arange(1, max(all_deg) + 2) - 0.5
        else:
            bins = 10
        r_mean = float(np.mean(r_deg)) if r_deg else 0.0
        py_mean = float(np.mean(py_deg)) if py_deg else 0.0
        ax.hist(
            r_deg, bins=bins, alpha=0.5,
            label=f"R  ({len(r_deg)} {plural}, mean {r_mean:.1f})", color="#E24A33",
        )
        ax.hist(
            py_deg, bins=bins, alpha=0.5,
            label=f"Python  ({len(py_deg)} {plural}, mean {py_mean:.1f})",
            color="#348ABD",
        )
        ax.set_yscale("log")
        ax.set_xlabel(f"# circuits containing this {singular}")
        ax.set_ylabel(f"# {plural} (log)")
        ax.set_title(f"How many circuits does each {singular} participate in?")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Circuit-participation distributions\n"
        "Each bar = # of TFs/genes/peaks that appear in exactly that many circuits"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_circuit_degrees.png", dpi=150)
    print("  wrote plots/parity_circuit_degrees.png")

    # ---- Plot 4c: top-30 TF ranking by # participating circuits ----
    r_tf_deg = Counter(t[2] for t in r_trip)
    py_tf_deg = Counter(t[2] for t in py_trip)
    all_tfs = sorted(set(r_tf_deg) | set(py_tf_deg),
                     key=lambda t: -(r_tf_deg[t] + py_tf_deg[t]))

    # Rank correlation on the union (needed for the annotation)
    r_vec = np.array([r_tf_deg[t] for t in all_tfs])
    p_vec = np.array([py_tf_deg[t] for t in all_tfs])
    from scipy.stats import spearmanr, pearsonr
    sp = spearmanr(r_vec, p_vec).statistic
    pr = pearsonr(r_vec, p_vec).statistic

    top = all_tfs[:30]
    top_r_vals = np.array([r_tf_deg[t] for t in top])
    top_py_vals = np.array([py_tf_deg[t] for t in top])
    top_diff = np.abs(top_r_vals - top_py_vals)
    top_max_diff = int(top_diff.max()) if len(top_diff) else 0
    top_median_diff = float(np.median(top_diff)) if len(top_diff) else 0.0
    # % relative agreement on the top-30
    top_rel_err = top_diff / np.maximum(np.maximum(top_r_vals, top_py_vals), 1)
    top_mean_rel = float(top_rel_err.mean())

    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(top))
    w = 0.4
    ax.bar(idx - w / 2, top_r_vals, w, label="R", color="#E24A33")
    ax.bar(idx + w / 2, top_py_vals, w, label="Python", color="#348ABD")
    ax.set_xticks(idx)
    ax.set_xticklabels(top, rotation=75, fontsize=8, ha="right")
    ax.set_ylabel("# circuits containing TF")
    ax.set_title(
        f"Top-30 most-participating TFs (ranked by R+Python total)\n"
        f"Full-vector Spearman \u03c1 = {sp:.3f}  \u2022  "
        f"Pearson r = {pr:.3f}  \u2022  "
        f"n = {len(all_tfs)} TFs total"
    )
    # inline stats box
    ax.text(
        0.98, 0.95,
        f"Top-30 median |R \u2212 Py| = {top_median_diff:.1f}\n"
        f"Top-30 max |R \u2212 Py| = {top_max_diff}\n"
        f"Top-30 mean rel. error = {top_mean_rel:.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.75))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_top_TFs.png", dpi=150)
    print("  wrote plots/parity_top_TFs.png")
    print(f"  TF-participation vector: Spearman \u03c1={sp:.3f}, Pearson r={pr:.3f}")

    # ---- Emit metrics TSV ----
    def _row(name, r, p, tau=None):
        d = {
            "output": name,
            "pearson_r": float(np.corrcoef(r.ravel(), p.ravel())[0, 1]),
            "mae": float(np.mean(np.abs(r - p))),
            "max_abs_err": float(np.max(np.abs(r - p))),
            "n_elements": int(r.size),
        }
        if tau is not None:
            rb = (r > tau).ravel(); pb = (p > tau).ravel()
            inter_ = int(np.logical_and(rb, pb).sum())
            uni_ = int(np.logical_or(rb, pb).sum())
            d.update({
                f"thresh_{tau}_jaccard":
                    inter_ / uni_ if uni_ > 0 else 1.0,
                f"thresh_{tau}_r_positive": int(rb.sum()),
                f"thresh_{tau}_py_positive": int(pb.sum()),
                f"thresh_{tau}_shared": inter_,
            })
        return d

    rows = [
        _row("B_prior", r_B_prior, py_B_prior),
        _row("L_prior", r_L_prior, py_L_prior),
        _row("TF_Peak_Binding_prob", r_B_post, py_B_post, PROB_TF_PEAK),
        _row("Peak_Gene_Looping_prob", r_L_post, py_L_post, PROB_PEAK_GENE),
        {"output": "Noise_parameters sigma_A",
         "pearson_r": float(np.corrcoef(r_noise[:, 0], py_noise[:, 0])[0, 1]),
         "mae": float(np.mean(np.abs(r_noise[:, 0] - py_noise[:, 0]))),
         "max_abs_err": float(np.max(np.abs(r_noise[:, 0] - py_noise[:, 0]))),
         "n_elements": int(r_noise.shape[0])},
        {"output": "Noise_parameters sigma_R",
         "pearson_r": float(np.corrcoef(r_noise[:, 1], py_noise[:, 1])[0, 1]),
         "mae": float(np.mean(np.abs(r_noise[:, 1] - py_noise[:, 1]))),
         "max_abs_err": float(np.max(np.abs(r_noise[:, 1] - py_noise[:, 1]))),
         "n_elements": int(r_noise.shape[0])},
        {"output": "Circuits (triples)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(union),
         "circuits_jaccard": j_trip,
         "circuits_precision": prec,
         "circuits_recall": rec,
         "circuits_r_only": len(r_trip - py_trip),
         "circuits_py_only": len(py_trip - r_trip),
         "circuits_shared": len(inter)},
        {"output": "Circuits (TF set)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(r_tf_set | py_tf_set),
         "circuits_jaccard": _jaccard(r_tf_set, py_tf_set),
         "circuits_r_only": len(r_tf_set - py_tf_set),
         "circuits_py_only": len(py_tf_set - r_tf_set),
         "circuits_shared": len(r_tf_set & py_tf_set),
         "tf_participation_spearman": float(sp),
         "tf_participation_pearson": float(pr)},
        {"output": "Circuits (Peak set)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(r_peak_set | py_peak_set),
         "circuits_jaccard": _jaccard(r_peak_set, py_peak_set),
         "circuits_r_only": len(r_peak_set - py_peak_set),
         "circuits_py_only": len(py_peak_set - r_peak_set),
         "circuits_shared": len(r_peak_set & py_peak_set)},
        {"output": "Circuits (Gene set)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(r_gene_set | py_gene_set),
         "circuits_jaccard": _jaccard(r_gene_set, py_gene_set),
         "circuits_r_only": len(r_gene_set - py_gene_set),
         "circuits_py_only": len(py_gene_set - r_gene_set),
         "circuits_shared": len(r_gene_set & py_gene_set)},
        {"output": "Circuits (Peak-Gene pairs)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(r_pg_pairs | py_pg_pairs),
         "circuits_jaccard": _jaccard(r_pg_pairs, py_pg_pairs),
         "circuits_r_only": len(r_pg_pairs - py_pg_pairs),
         "circuits_py_only": len(py_pg_pairs - r_pg_pairs),
         "circuits_shared": len(r_pg_pairs & py_pg_pairs)},
        {"output": "Circuits (TF-Peak pairs)",
         "pearson_r": float("nan"), "mae": float("nan"), "max_abs_err": float("nan"),
         "n_elements": len(r_tp_pairs | py_tp_pairs),
         "circuits_jaccard": _jaccard(r_tp_pairs, py_tp_pairs),
         "circuits_r_only": len(r_tp_pairs - py_tp_pairs),
         "circuits_py_only": len(py_tp_pairs - r_tp_pairs),
         "circuits_shared": len(r_tp_pairs & py_tp_pairs)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_TSV, sep="\t", index=False, float_format="%.6g")
    print(f"\nWrote metrics table: {METRICS_TSV.relative_to(REPO_ROOT)}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
