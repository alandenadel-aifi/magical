"""Port of ``MAGICAL_initialization`` (R/MAGICAL_functions.R lines 474-556).

Builds prior distributions from the candidate multi-omics summaries.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse, stats

from .types import CandidateCircuits, InitialModel, LoadedData


def magical_initialization(
    loaded_data: LoadedData, candidate_circuits: CandidateCircuits
) -> InitialModel:
    """Numpy port of ``MAGICAL_initialization``.

    Mirrors R semantics closely; only cosmetic differences (0-based indexing).
    """
    TFs = candidate_circuits.TFs
    tf_log2 = np.asarray(candidate_circuits.TF_log2Count, dtype=float)

    peak_log2 = np.asarray(candidate_circuits.Peak_log2Count, dtype=float)
    gene_log2 = np.asarray(candidate_circuits.Gene_log2Count, dtype=float)

    tf_peak = _to_dense_binary(candidate_circuits.TF_Peak_Binding)
    peak_gene = _to_dense_binary(candidate_circuits.Peak_Gene_looping)

    S = len(loaded_data.common_samples)
    M = TFs.shape[0]
    P = candidate_circuits.Peaks.shape[0]
    G = candidate_circuits.Genes.shape[0]

    # ---- TF activity prior --------------------------------------------------
    T_prior = tf_log2.copy()
    T_mean = tf_log2.copy()
    # R: var() with default ddof=1 (sample variance)
    T_var = np.tile(
        _row_variance(tf_log2, ddof=1).reshape(M, 1), (1, S)
    )

    # ---- TF-Peak binding prior via OLS slope ------------------------------
    B_prior = tf_peak.astype(float).copy()
    B_prob = tf_peak.astype(float).copy()

    # `which(mat > 0, arr.ind = TRUE)` returns (row, col) 1-indexed pairs
    # ordered column-major. Match that traversal for parity.
    tf_peak_rows, tf_peak_cols = _which_col_major(tf_peak > 0)
    for r, c in zip(tf_peak_rows, tf_peak_cols):
        slope, pval = _ols_slope_pvalue(
            x=tf_log2[c, :], y=peak_log2[r, :]
        )
        B_prior[r, c] = slope
        B_prob[r, c] = 1.0 - pval
    B_mean = B_prior.copy()

    B_var = np.full((1, M), 0.5, dtype=float)
    for m in range(M):
        col_mask = tf_peak[:, m] > 0
        if col_mask.sum() > 1:
            B_var[0, m] = float(np.var(B_prior[col_mask, m], ddof=1))
        else:
            B_var[0, m] = 0.5

    # ---- Peak-Gene looping prior ------------------------------------------
    L_prior = peak_gene.astype(float).copy()
    L_prob = peak_gene.astype(float).copy()

    pg_rows, pg_cols = _which_col_major(peak_gene > 0)
    for r, c in zip(pg_rows, pg_cols):
        slope, pval = _ols_slope_pvalue(
            x=peak_log2[r, :], y=gene_log2[c, :]
        )
        L_prior[r, c] = slope
        L_prob[r, c] = 1.0 - pval
    L_mean = L_prior.copy()
    # R: L_var = var(L_prior[which(L_prob > 0)]) -- scalar
    lvar_mask = L_prob > 0
    L_var = float(np.var(L_prior[lvar_mask], ddof=1))

    return InitialModel(
        T_prior=T_prior,
        T_mean=T_mean,
        T_var=T_var,
        B_prior=B_prior,
        B_mean=B_mean,
        B_var=B_var,
        B_prob=B_prob,
        L_prior=L_prior,
        L_mean=L_mean,
        L_var=L_var,
        L_prob=L_prob,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense_binary(mat: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(mat):
        return np.asarray(mat.todense())
    return np.asarray(mat)


def _row_variance(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Per-row variance with R's default ``ddof = 1``."""
    return np.var(x, axis=1, ddof=ddof)


def _which_col_major(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of True entries in column-major (R) order."""
    # np.nonzero on the transposed matrix walks columns of the original
    cols, rows = np.nonzero(mask.T)
    return rows, cols


def _ols_slope_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope and two-sided p-value for the slope coefficient.

    Matches ``summary(lm(y ~ x))$coefficients[2, c(1, 4)]`` in R.
    """
    # scipy.stats.linregress uses the two-sided t-test for the slope,
    # identical to lm() in R.
    res = stats.linregress(x=x, y=y)
    return float(res.slope), float(res.pvalue)
