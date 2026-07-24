"""Demo-scale parity tests: numpy port vs original R implementation on the
shipped ``Demo input files/`` dataset.

Opt-in via ``MAGICAL_RUN_SLOW_TESTS=1`` because the R snapshot generation
takes about 3 minutes and the Python re-run adds ~1-2 more.

R snapshots are produced by::

    Rscript re-implementation/scripts/gen_demo_parity_snapshots.R

Everything up through candidate-circuit construction is deterministic, so
Python must match R element-wise on shapes and (for the log2 count matrices,
after row/col ordering is aligned) on values. Estimation is RNG-dependent
so we assert Pearson correlation >= 0.90 on the posterior probability maps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magical import estimation, magical_initialization
from magical.circuits import candidate_circuits_construction_with_TAD
from magical.io import data_loading


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "Demo input files"
SNAP = REPO_ROOT / "tests" / "snapshots_demo"


slow_reason = "set MAGICAL_RUN_SLOW_TESTS=1 to enable"
slow_skip = pytest.mark.skipif(
    os.environ.get("MAGICAL_RUN_SLOW_TESTS") != "1", reason=slow_reason
)


def _load_tsv_mat(name: str) -> np.ndarray:
    return np.loadtxt(SNAP / f"{name}.tsv", delimiter="\t")


def _load_tsv_df(name: str) -> pd.DataFrame:
    return pd.read_csv(SNAP / f"{name}.tsv", sep="\t")


@pytest.fixture(scope="module")
def demo_snapshots():
    if not SNAP.exists() or not any(SNAP.glob("*.tsv")):
        pytest.skip(
            "demo snapshots missing; run "
            "`Rscript re-implementation/scripts/gen_demo_parity_snapshots.R`"
        )
    if not DEMO_DIR.exists():
        pytest.skip("Demo input files/ not found")
    return {
        "TFs": _load_tsv_df("TFs"),
        "Peaks": _load_tsv_df("Peaks"),
        "Genes": _load_tsv_df("Genes"),
        "TF_log2Count": _load_tsv_mat("TF_log2Count"),
        "Peak_log2Count": _load_tsv_mat("Peak_log2Count"),
        "Gene_log2Count": _load_tsv_mat("Gene_log2Count"),
        "TF_Peak_Binding": _load_tsv_mat("TF_Peak_Binding").astype(np.int8),
        "Peak_Gene_looping": _load_tsv_mat("Peak_Gene_looping").astype(np.int8),
        "B_prior": _load_tsv_mat("B_prior"),
        "B_prob": _load_tsv_mat("B_prob"),
        "L_prior": _load_tsv_mat("L_prior"),
        "L_prob": _load_tsv_mat("L_prob"),
        "T_var": _load_tsv_mat("T_var"),
        "L_var": float(Path(SNAP / "L_var.tsv").read_text().strip()),
        "TF_Peak_Binding_prob": _load_tsv_mat("TF_Peak_Binding_prob"),
        "Peak_Gene_Looping_prob": _load_tsv_mat("Peak_Gene_Looping_prob"),
        "n_iter": int(Path(SNAP / "n_iter.tsv").read_text().strip()),
        "common_samples": Path(SNAP / "Common_samples.tsv").read_text().splitlines(),
    }


@pytest.fixture(scope="module")
def py_pipeline(demo_snapshots):
    d = DEMO_DIR
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
    cand = candidate_circuits_construction_with_TAD(
        loaded, d / "RaoGM12878_40kb_TopDomTADs_filtered_hg38.txt",
    )
    im = magical_initialization(loaded.as_loaded_data(), cand)
    return loaded, cand, im


def _align_rows_cols(py_mat, py_row_names, py_col_names, r_row_names, r_col_names):
    """Reorder ``py_mat`` so its rows/cols match R's ordering."""
    row_map = {n: i for i, n in enumerate(py_row_names)}
    col_map = {n: i for i, n in enumerate(py_col_names)}
    row_idx = np.array([row_map[n] for n in r_row_names])
    col_idx = np.array([col_map[n] for n in r_col_names])
    return py_mat[row_idx, :][:, col_idx]


# ---- Loader parity --------------------------------------------------------

@slow_skip
def test_common_samples_match(demo_snapshots, py_pipeline) -> None:
    loaded, _, _ = py_pipeline
    assert list(loaded.common_samples) == list(demo_snapshots["common_samples"])


@slow_skip
def test_candidate_circuit_shapes_match(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    assert cand.Peaks.shape[0] == demo_snapshots["Peaks"].shape[0]
    assert cand.Genes.shape[0] == demo_snapshots["Genes"].shape[0]
    assert cand.TFs.shape[0] == demo_snapshots["TFs"].shape[0]
    assert cand.TF_Peak_Binding.shape == demo_snapshots["TF_Peak_Binding"].shape
    assert cand.Peak_Gene_looping.shape == demo_snapshots["Peak_Gene_looping"].shape


@slow_skip
def test_candidate_membership_matches(demo_snapshots, py_pipeline) -> None:
    """Row/col identity sets must agree (order may differ due to dplyr vs
    pandas merge conventions)."""
    _, cand, _ = py_pipeline
    r_peaks = set(demo_snapshots["Peaks"]["Peak_index"].astype(int))
    py_peaks = set(cand.Peaks["Peak_index"].astype(int))
    assert py_peaks == r_peaks

    r_genes = set(demo_snapshots["Genes"]["Gene_symbols"])
    py_genes = set(cand.Genes["Gene_symbols"])
    assert py_genes == r_genes

    r_tfs = set(demo_snapshots["TFs"]["name"])
    py_tfs = set(cand.TFs["name"])
    assert py_tfs == r_tfs


@slow_skip
def test_TF_Peak_Binding_matches_after_reordering(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    py_mat = np.asarray(cand.TF_Peak_Binding.todense())
    aligned = _align_rows_cols(
        py_mat,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.TFs["name"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["TFs"]["name"].tolist(),
    )
    np.testing.assert_array_equal(aligned, demo_snapshots["TF_Peak_Binding"])


@slow_skip
def test_Peak_Gene_looping_matches_after_reordering(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    py_mat = np.asarray(cand.Peak_Gene_looping.todense())
    aligned = _align_rows_cols(
        py_mat,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.Genes["Gene_symbols"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["Genes"]["Gene_symbols"].tolist(),
    )
    np.testing.assert_array_equal(aligned, demo_snapshots["Peak_Gene_looping"])


@slow_skip
def test_Peak_log2Count_matches_R(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    # rows aligned by Peak_index; columns are already common_samples in order
    peak_map = {int(p): i for i, p in enumerate(cand.Peaks["Peak_index"].values)}
    row_idx = np.array(
        [peak_map[int(p)] for p in demo_snapshots["Peaks"]["Peak_index"]]
    )
    aligned = cand.Peak_log2Count[row_idx, :]
    np.testing.assert_allclose(aligned, demo_snapshots["Peak_log2Count"], atol=1e-8)


@slow_skip
def test_Gene_log2Count_matches_R(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    gene_map = {g: i for i, g in enumerate(cand.Genes["Gene_symbols"].values)}
    row_idx = np.array([gene_map[g] for g in demo_snapshots["Genes"]["Gene_symbols"]])
    aligned = cand.Gene_log2Count[row_idx, :]
    np.testing.assert_allclose(aligned, demo_snapshots["Gene_log2Count"], atol=1e-8)


@slow_skip
def test_TF_log2Count_matches_R(demo_snapshots, py_pipeline) -> None:
    _, cand, _ = py_pipeline
    tf_map = {t: i for i, t in enumerate(cand.TFs["name"].values)}
    row_idx = np.array([tf_map[t] for t in demo_snapshots["TFs"]["name"]])
    aligned = cand.TF_log2Count[row_idx, :]
    np.testing.assert_allclose(aligned, demo_snapshots["TF_log2Count"], atol=1e-8)


# ---- Initialization parity -----------------------------------------------

@slow_skip
def test_initialization_B_prior_matches_R(demo_snapshots, py_pipeline) -> None:
    _, cand, im = py_pipeline
    aligned = _align_rows_cols(
        im.B_prior,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.TFs["name"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["TFs"]["name"].tolist(),
    )
    np.testing.assert_allclose(aligned, demo_snapshots["B_prior"], atol=1e-8)


@slow_skip
def test_initialization_L_prior_matches_R(demo_snapshots, py_pipeline) -> None:
    _, cand, im = py_pipeline
    aligned = _align_rows_cols(
        im.L_prior,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.Genes["Gene_symbols"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["Genes"]["Gene_symbols"].tolist(),
    )
    np.testing.assert_allclose(aligned, demo_snapshots["L_prior"], atol=1e-8)


@slow_skip
def test_initialization_L_var_matches_R(demo_snapshots, py_pipeline) -> None:
    _, _, im = py_pipeline
    assert abs(im.L_var - demo_snapshots["L_var"]) < 1e-6


# ---- Estimation parity (RNG-dependent) -----------------------------------

@slow_skip
def test_estimation_posteriors_correlate_with_R(demo_snapshots, py_pipeline) -> None:
    loaded, cand, im = py_pipeline
    result = estimation(
        loaded.as_loaded_data(), cand, im,
        iteration_num=demo_snapshots["n_iter"], backend="numpy", seed=20260723,
    )
    # Align Python posterior to R ordering
    py_b = _align_rows_cols(
        result.TF_Peak_Binding_prob,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.TFs["name"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["TFs"]["name"].tolist(),
    )
    py_l = _align_rows_cols(
        result.Peak_Gene_Looping_prob,
        cand.Peaks["Peak_index"].astype(int).tolist(),
        cand.Genes["Gene_symbols"].tolist(),
        demo_snapshots["Peaks"]["Peak_index"].astype(int).tolist(),
        demo_snapshots["Genes"]["Gene_symbols"].tolist(),
    )
    corr_b = float(np.corrcoef(py_b.ravel(), demo_snapshots["TF_Peak_Binding_prob"].ravel())[0, 1])
    corr_l = float(np.corrcoef(py_l.ravel(), demo_snapshots["Peak_Gene_Looping_prob"].ravel())[0, 1])
    assert corr_b >= 0.90, f"demo TF-Peak posterior correlation too low: {corr_b:.3f}"
    assert corr_l >= 0.90, f"demo Peak-Gene posterior correlation too low: {corr_l:.3f}"
