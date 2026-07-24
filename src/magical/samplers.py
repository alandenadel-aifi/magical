"""Numpy ports of the five Gibbs samplers in ``R/MAGICAL_functions.R``.

Each function is a line-by-line translation of the corresponding R function;
naming and signatures follow R conventions so cross-referencing stays easy.
Every draw uses the supplied ``numpy.random.Generator`` (no global state).
"""

from __future__ import annotations

import numpy as np

from ._rng import truncated_standard_normal
from .types import TFA


# ---------------------------------------------------------------------------
# Step 1 — TF activity sampling
# ---------------------------------------------------------------------------

def tf_activity_sampling(
    A: np.ndarray,                # unused (parity)
    A_sample: np.ndarray,         # (P, S)
    ATAC_Cell_Sample_vector: np.ndarray,  # (N_atac,)
    R: np.ndarray,                # unused (parity)
    R_sample: np.ndarray,         # unused (parity)
    RNA_Cell_Sample_vector: np.ndarray,   # (N_rna,)
    tfa: TFA,
    T_prior_mean: np.ndarray,     # (M, S)
    T_prior_var: np.ndarray,      # (M, S)
    B: np.ndarray,                # (P, M)
    B_state: np.ndarray,          # (P, M)
    sigma_A_noise: float,
    P: int,
    G: int,
    M: int,
    S: int,
    rng: np.random.Generator,
) -> TFA:
    """Port of ``TF_activity_T_sampling``."""
    del A, R, R_sample  # unused in original R too

    TF_index = rng.permutation(M)
    for m in TF_index:
        if B_state[:, m].sum() <= 0:
            continue
        bm = B[:, m]
        bm_sq = float((bm ** 2).sum())
        bstate_sum = float(B_state[:, m].sum())

        temp_var = bm_sq * T_prior_var[m, :] / bstate_sum + sigma_A_noise
        # A_sample - B %*% T_sample + B[,m] %*% t(T_sample[m,]) has shape (P, S)
        resid = A_sample - B @ tfa.T_sample + np.outer(bm, tfa.T_sample[m, :])
        mean_T = (
            (bm @ resid) / bstate_sum
            + T_prior_mean[m, :] * sigma_A_noise
        ) / temp_var
        variance_T = T_prior_var[m, :] * sigma_A_noise / temp_var

        for s in range(S):
            aa = float(truncated_standard_normal(rng, size=1)[0])
            tfa.T_sample[m, s] = aa * np.sqrt(abs(variance_T[s])) + mean_T[s]

            atac_idx = np.flatnonzero(ATAC_Cell_Sample_vector == (s + 1))
            if atac_idx.size > 0:
                z = rng.standard_normal(size=atac_idx.size)
                tfa.T_A[m, atac_idx] = z * np.sqrt(abs(variance_T[s])) + tfa.T_sample[m, s]

            rna_idx = np.flatnonzero(RNA_Cell_Sample_vector == (s + 1))
            if rna_idx.size > 0:
                z = rng.standard_normal(size=rna_idx.size)
                tfa.T_R[m, rna_idx] = z * np.sqrt(abs(variance_T[s])) + tfa.T_sample[m, s]
    return tfa


# ---------------------------------------------------------------------------
# Step 2 — TF-peak binding weight sampling
# ---------------------------------------------------------------------------

def tf_peak_binding_sampling(
    A: np.ndarray,
    A_sample: np.ndarray,
    ATAC_Cell_Sample_vector: np.ndarray,
    tfa: TFA,
    B: np.ndarray,
    B_state: np.ndarray,
    B_prior_mean: np.ndarray,
    B_prior_var: np.ndarray,  # (1, M)
    sigma_A_noise: float,
    P: int,
    G: int,
    M: int,
    S: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Port of ``TF_peak_binding_B_sampling``."""
    del A, ATAC_Cell_Sample_vector  # T_sample matrix built in R is dead code

    B = B.copy()
    B_prior_var_flat = np.asarray(B_prior_var).ravel()

    TF_index = rng.permutation(M)
    for m in TF_index:
        Tm = tfa.T_sample[m, :]
        temp_var = (Tm ** 2).sum() * B_prior_var_flat[m] / S + sigma_A_noise
        resid = A_sample - B @ tfa.T_sample + np.outer(B[:, m], Tm)
        mean_B = (
            (resid @ Tm) * B_prior_var_flat[m] / S
            + B_prior_mean[:, m] * sigma_A_noise
        ) / temp_var
        variance_B = B_prior_var_flat[m] * sigma_A_noise / temp_var

        bb = truncated_standard_normal(rng, size=P)
        B[:, m] = (bb * np.sqrt(abs(variance_B)) + mean_B) * B_state[:, m]
    return B


# ---------------------------------------------------------------------------
# Step 3 — TF-peak binding STATE update
# ---------------------------------------------------------------------------

def tf_peak_binary_binding_sampling(
    A: np.ndarray,
    A_sample: np.ndarray,
    ATAC_Cell_Sample_vector: np.ndarray,
    tfa: TFA,
    B: np.ndarray,
    B_state: np.ndarray,
    B_prior_mean: np.ndarray,
    B_prior_var: np.ndarray,   # (1, M)
    B_prior_prob: np.ndarray,  # (P, M)
    sigma_A_noise: float,
    P: int,
    G: int,
    M: int,
    S: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``TF_peak_binary_binding_B_state_sampling``."""
    del A, ATAC_Cell_Sample_vector

    B = B.copy()
    B_state = B_state.copy()
    B_prior_var_flat = np.asarray(B_prior_var).ravel()

    TF_index = rng.permutation(M)
    Peak_index = rng.permutation(P)

    for f in Peak_index:
        temp = B[f, :] @ tfa.T_sample  # (S,)
        for m in TF_index:
            Tm = tfa.T_sample[m, :]
            temp_var = (Tm ** 2).sum() * B_prior_var_flat[m] / S + sigma_A_noise
            variance_B = B_prior_var_flat[m] * sigma_A_noise / temp_var

            if B_state[f, m] > 0:
                mean_B = (
                    ((A_sample[f, :] - temp + B[f, m] * Tm) * Tm).sum()
                    * B_prior_var_flat[m] / S
                    + B_prior_mean[f, m] * sigma_A_noise
                ) / temp_var

                post_b1 = np.exp(-((B[f, m] * 1 - mean_B) ** 2) / (2 * variance_B)) * (
                    B_prior_prob[f, m] + 0.25
                ) + 1e-6
                post_b0 = np.exp(-((B[f, m] * 0 - mean_B) ** 2) / (2 * variance_B)) * (
                    1 - B_prior_prob[f, m] + 0.25
                ) + 1e-6
                P1 = post_b1 / (post_b1 + post_b0)
                if not np.isfinite(P1):
                    P1 = 0.5
                if P1 < rng.random():
                    B[f, m] = 0.0
                    B_state[f, m] = 0
                # else: leave B[f,m] alone, B_state[f,m] = 1 already

            if B_state[f, m] == 0 and B_prior_prob[f, m] > 0:
                mean_B = (
                    ((A_sample[f, :] - temp) * Tm).sum() * B_prior_var_flat[m] / S
                    + B_prior_mean[f, m] * sigma_A_noise
                ) / temp_var

                bb = float(truncated_standard_normal(rng, size=1)[0])
                B_temp = bb * np.sqrt(abs(variance_B)) + mean_B

                post_b1 = np.exp(-(bb ** 2) / 2) * (B_prior_prob[f, m] + 0.25) + 1e-6
                post_b0 = np.exp(-(mean_B ** 2) / (2 * variance_B)) * (
                    1 - B_prior_prob[f, m] + 0.25
                ) + 1e-6
                P1 = post_b1 / (post_b1 + post_b0)
                if not np.isfinite(P1):
                    P1 = 0.5
                if P1 < rng.random():
                    B[f, m] = 0.0
                    B_state[f, m] = 0
                else:
                    B[f, m] = B_temp
                    B_state[f, m] = 1
    return B, B_state


# ---------------------------------------------------------------------------
# Step 4 — Peak-gene looping weight sampling
# ---------------------------------------------------------------------------

def peak_gene_looping_sampling(
    R: np.ndarray,
    R_sample: np.ndarray,
    RNA_Cell_Sample_vector: np.ndarray,
    tfa: TFA,
    B: np.ndarray,
    L: np.ndarray,
    L_state: np.ndarray,
    L_prior_mean: np.ndarray,
    L_prior_var: float,
    sigma_R_noise: float,
    P: int,
    G: int,
    M: int,
    S: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Port of ``Peak_gene_looping_L_samping``."""
    del R, RNA_Cell_Sample_vector  # T_sample block in R is unused

    L = L.copy()
    A_estimate = B @ tfa.T_sample  # (P, S)

    Peak_index = rng.permutation(P)
    for f in Peak_index:
        af = A_estimate[f, :]
        temp_var = (af ** 2).sum() * L_prior_var / S + sigma_R_noise
        resid = R_sample - L.T @ A_estimate + np.outer(L[f, :], af)  # (G, S)
        mean_L = (
            (resid @ af) * L_prior_var / S
            + L_prior_mean[f, :] * sigma_R_noise
        ) / temp_var
        variance_L = L_prior_var * sigma_R_noise / temp_var

        ll = truncated_standard_normal(rng, size=G)
        # R uses sqrt(vairance_L) without abs here (differs from the state
        # update version). Match that behaviour.
        L[f, :] = (ll * np.sqrt(variance_L) + mean_L) * L_state[f, :]
    return L


# ---------------------------------------------------------------------------
# Step 5 — Peak-gene looping STATE update
# ---------------------------------------------------------------------------

def peak_gene_binary_looping_sampling(
    R: np.ndarray,
    R_sample: np.ndarray,
    RNA_Cell_Sample_vector: np.ndarray,
    tfa: TFA,
    B: np.ndarray,
    L: np.ndarray,
    L_state: np.ndarray,
    L_prior_mean: np.ndarray,
    L_prior_var: float,
    L_prior_prob: np.ndarray,
    sigma_R_noise: float,
    P: int,
    G: int,
    M: int,
    S: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``Peak_gene_binary_looping_L_state_samping``."""
    del R, RNA_Cell_Sample_vector

    L = L.copy()
    L_state = L_state.copy()

    Peak_index = rng.permutation(P)
    Gene_index = rng.permutation(G)

    A_estimate = B @ tfa.T_sample  # (P, S)

    for g in Gene_index:
        temp = L[:, g] @ A_estimate  # (S,)
        for f in Peak_index:
            af = A_estimate[f, :]
            temp_var = (af ** 2).sum() * L_prior_var / S + sigma_R_noise
            variance_L = L_prior_var * sigma_R_noise / temp_var

            if L_state[f, g] > 0:
                mean_L = (
                    ((R_sample[g, :] - temp + L[f, g] * af) * af).sum()
                    * L_prior_var / S
                    + L_prior_mean[f, g] * sigma_R_noise
                ) / temp_var

                post_l1 = np.exp(-((L[f, g] * 1 - mean_L) ** 2) / (2 * variance_L)) * (
                    L_prior_prob[f, g] + 0.25
                ) + 1e-6
                post_l0 = np.exp(-((L[f, g] * 0 - mean_L) ** 2) / (2 * variance_L)) * (
                    1 - L_prior_prob[f, g] + 0.25
                ) + 1e-6
                P1 = post_l1 / (post_l1 + post_l0)
                if not np.isfinite(P1):
                    P1 = 0.5
                if P1 < rng.random():
                    L[f, g] = 0.0
                    L_state[f, g] = 0

            if L_state[f, g] == 0 and L_prior_prob[f, g] > 0:
                mean_L = (
                    ((R_sample[g, :] - temp) * af).sum() * L_prior_var / S
                    + L_prior_mean[f, g] * sigma_R_noise
                ) / temp_var

                ll = float(truncated_standard_normal(rng, size=1)[0])
                L_temp = ll * np.sqrt(abs(variance_L)) + mean_L

                # Note the +0.1 (not 0.25!) here — matches R state-update block.
                post_l1 = np.exp(-(ll ** 2) / 2) * (L_prior_prob[f, g] + 0.1) + 1e-6
                post_l0 = np.exp(-(mean_L ** 2) / (2 * variance_L)) * (
                    1 - L_prior_prob[f, g] + 0.1
                ) + 1e-6
                P1 = post_l1 / (post_l1 + post_l0)
                if not np.isfinite(P1):
                    P1 = 0.5
                if P1 < rng.random():
                    L[f, g] = 0.0
                    L_state[f, g] = 0
                else:
                    L[f, g] = L_temp
                    L_state[f, g] = 1
    return L, L_state
