"""Cross-language parity tests: numpy port vs original R implementation.

R writes snapshots via ``re-implementation/scripts/gen_parity_snapshots.R``.
Python reads those snapshots, rebuilds identical inputs, and asserts:

* ``magical_initialization`` output is *bit-for-bit* equivalent to R
  (no RNG involved).
* ``estimation(backend="numpy")`` output correlates >= 0.85 with the R
  posterior (RNG streams differ; statistical equivalence is the goal).

If the snapshots aren't present, the tests are skipped with instructions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from magical import estimation, magical_initialization
from magical.types import CandidateCircuits, LoadedData


REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP = REPO_ROOT / "tests" / "snapshots"


def _load(name: str) -> np.ndarray:
    p = SNAP / f"{name}.tsv"
    return np.loadtxt(p, delimiter="\t")


@pytest.fixture(scope="module")
def r_snapshots():
    if not SNAP.exists() or not any(SNAP.glob("*.tsv")):
        pytest.skip(
            "R snapshots missing. Regenerate with "
            "`Rscript re-implementation/scripts/gen_parity_snapshots.R`"
        )
    n_iter = int(Path(SNAP / "n_iter.tsv").read_text().strip())
    L_var = float(Path(SNAP / "L_var.tsv").read_text().strip())
    return {
        "TF_log2Count": _load("TF_log2Count"),
        "Peak_log2Count": _load("Peak_log2Count"),
        "Gene_log2Count": _load("Gene_log2Count"),
        "TF_Peak_Binding": _load("TF_Peak_Binding").astype(np.int8),
        "Peak_Gene_looping": _load("Peak_Gene_looping").astype(np.int8),
        "T_prior": _load("T_prior"),
        "T_mean": _load("T_mean"),
        "T_var": _load("T_var"),
        "B_prior": _load("B_prior"),
        "B_mean": _load("B_mean"),
        "B_var": _load("B_var").reshape(1, -1),
        "B_prob": _load("B_prob"),
        "L_prior": _load("L_prior"),
        "L_mean": _load("L_mean"),
        "L_var": L_var,
        "L_prob": _load("L_prob"),
        "TF_Peak_Binding_prob": _load("TF_Peak_Binding_prob"),
        "Peak_Gene_Looping_prob": _load("Peak_Gene_Looping_prob"),
        "Noise_parameters": _load("Noise_parameters"),
        "n_iter": n_iter,
    }


def _build_inputs(snap):
    """Reconstruct LoadedData + CandidateCircuits from raw R snapshot matrices."""
    tf_log2 = snap["TF_log2Count"]
    peak_log2 = snap["Peak_log2Count"]
    gene_log2 = snap["Gene_log2Count"]
    tf_peak = snap["TF_Peak_Binding"]
    peak_gene = snap["Peak_Gene_looping"]

    M, S = tf_log2.shape
    P = peak_log2.shape[0]
    G = gene_log2.shape[0]

    sample_names = [f"S{i + 1}" for i in range(S)]
    tf_names = [f"TF{i + 1}" for i in range(M)]
    peak_ids = [f"peak{i + 1}" for i in range(P)]
    gene_names = [f"gene{i + 1}" for i in range(G)]

    # Cell metadata (structure matters for estimation but not for
    # initialization). Use 8 cells per sample per modality (matches the R
    # snapshot script).
    cps = 8
    def cellmeta(prefix):
        rows = [
            {
                "cell_index": s * cps + k + 1,
                "cell_barcode": f"{prefix}_{sample_names[s]}_{k + 1}",
                "cell_type": "T",
                "subject_ID": sample_names[s],
            }
            for s in range(S)
            for k in range(cps)
        ]
        return pd.DataFrame(rows)

    scATAC_cells = cellmeta("atac")
    scRNA_cells = cellmeta("rna")
    scRNA_cells["condition"] = "cond"

    rng = np.random.default_rng(0)
    scATAC_read = sparse.csr_matrix(
        rng.poisson(lam=2.0, size=(P, scATAC_cells.shape[0])).astype(float)
    )
    rna_features = list(dict.fromkeys([*gene_names, *tf_names]))
    scRNA_read = sparse.csr_matrix(
        rng.poisson(lam=3.0, size=(len(rna_features), scRNA_cells.shape[0])).astype(float)
    )

    loaded_data = LoadedData(
        common_samples=sample_names,
        scATAC_Peaks=pd.DataFrame(
            {
                "Peak_index": peak_ids,
                "chr": "chr1",
                "point1": (np.arange(P) + 1) * 1000,
                "point2": (np.arange(P) + 1) * 1000 + 500,
            }
        ),
        scATAC_cells=scATAC_cells,
        scATAC_read_count_matrix=scATAC_read,
        scRNA_Genes=pd.DataFrame(
            {"Gene_index": np.arange(1, len(rna_features) + 1), "Gene_symbols": rna_features}
        ),
        scRNA_cells=scRNA_cells,
        scRNA_read_count_matrix=scRNA_read,
    )

    candidate = CandidateCircuits(
        TFs=pd.DataFrame({"TF_id": tf_names, "TF_name": tf_names}),
        TF_log2Count=tf_log2,
        Peaks=pd.DataFrame(
            {
                "Peak_index": peak_ids,
                "chr": "chr1",
                "point1": (np.arange(P) + 1) * 1000,
                "point2": (np.arange(P) + 1) * 1000 + 500,
            }
        ),
        Peak_log2Count=peak_log2,
        Genes=pd.DataFrame(
            {"Gene_symbols": gene_names, "chr": "chr1", "TSS": (np.arange(G) + 1) * 5000}
        ),
        Gene_log2Count=gene_log2,
        TF_Peak_Binding=sparse.csr_matrix(tf_peak),
        Peak_Gene_looping=sparse.csr_matrix(peak_gene),
    )
    return loaded_data, candidate


# --- Initialization parity (deterministic, must match exactly) ------------

@pytest.mark.parametrize(
    "field, atol",
    [
        ("T_prior", 1e-10),
        ("T_mean", 1e-10),
        ("T_var", 1e-10),
        ("B_prior", 1e-10),
        ("B_mean", 1e-10),
        ("B_var", 1e-10),
        ("B_prob", 1e-10),
        ("L_prior", 1e-10),
        ("L_mean", 1e-10),
        ("L_prob", 1e-10),
    ],
)
def test_initialization_matches_r_exactly(r_snapshots, field, atol) -> None:
    loaded, candidate = _build_inputs(r_snapshots)
    im = magical_initialization(loaded, candidate)
    py_val = getattr(im, field)
    r_val = r_snapshots[field]
    np.testing.assert_allclose(
        np.asarray(py_val), np.asarray(r_val), atol=atol,
        err_msg=f"{field} mismatch (Python vs R)",
    )


def test_initialization_L_var_matches_r_exactly(r_snapshots) -> None:
    loaded, candidate = _build_inputs(r_snapshots)
    im = magical_initialization(loaded, candidate)
    assert abs(im.L_var - r_snapshots["L_var"]) < 1e-10


# --- Estimation parity (statistical: correlation, means) -------------------

def _corr(a, b):
    x = np.asarray(a).ravel()
    y = np.asarray(b).ravel()
    if np.std(x) == 0 and np.std(y) == 0:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def test_estimation_TF_Peak_prob_correlates_with_r(r_snapshots) -> None:
    loaded, candidate = _build_inputs(r_snapshots)
    im = magical_initialization(loaded, candidate)
    result = estimation(loaded, candidate, im,
                        iteration_num=r_snapshots["n_iter"],
                        backend="numpy", seed=20260723)
    corr = _corr(result.TF_Peak_Binding_prob, r_snapshots["TF_Peak_Binding_prob"])
    assert corr >= 0.90, f"TF-Peak posterior correlation too low: {corr:.3f}"


def test_estimation_Peak_Gene_prob_correlates_with_r(r_snapshots) -> None:
    loaded, candidate = _build_inputs(r_snapshots)
    im = magical_initialization(loaded, candidate)
    result = estimation(loaded, candidate, im,
                        iteration_num=r_snapshots["n_iter"],
                        backend="numpy", seed=20260723)
    corr = _corr(result.Peak_Gene_Looping_prob, r_snapshots["Peak_Gene_Looping_prob"])
    assert corr >= 0.90, f"Peak-Gene posterior correlation too low: {corr:.3f}"


def test_estimation_posterior_masks_agree(r_snapshots) -> None:
    """Wherever R's posterior is exactly zero (no candidate edge), Python's
    must be too — and vice versa."""
    loaded, candidate = _build_inputs(r_snapshots)
    im = magical_initialization(loaded, candidate)
    result = estimation(loaded, candidate, im,
                        iteration_num=r_snapshots["n_iter"],
                        backend="numpy", seed=1)
    r_b_zero = r_snapshots["TF_Peak_Binding_prob"] == 0
    py_b_zero = result.TF_Peak_Binding_prob == 0
    np.testing.assert_array_equal(r_b_zero, py_b_zero)
    r_l_zero = r_snapshots["Peak_Gene_Looping_prob"] == 0
    py_l_zero = result.Peak_Gene_Looping_prob == 0
    np.testing.assert_array_equal(r_l_zero, py_l_zero)
