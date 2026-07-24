"""Numpy port of ``MAGICAL_estimation`` (outer Gibbs loop)."""

from __future__ import annotations

import numpy as np

from ._rng import inv_gamma
from .samplers import (
    peak_gene_binary_looping_sampling,
    peak_gene_looping_sampling,
    tf_activity_sampling,
    tf_peak_binary_binding_sampling,
    tf_peak_binding_sampling,
)
from .types import (
    CandidateCircuits,
    EstimationResult,
    InitialModel,
    LoadedData,
    TFA,
)


def magical_estimation(
    loaded_data: LoadedData,
    candidate_circuits: CandidateCircuits,
    initial_model: InitialModel,
    iteration_num: int,
    *,
    seed: int | None = None,
    verbose: bool = False,
) -> EstimationResult:
    """Port of ``MAGICAL_estimation`` (R/MAGICAL_functions.R lines 864-1023)."""
    rng = np.random.default_rng(seed)

    common_samples = loaded_data.common_samples
    S = len(common_samples)

    # --- ATAC data setup ---------------------------------------------------
    peak_ids = candidate_circuits.Peaks["Peak_index"].tolist()
    peak_index = np.array(
        [loaded_data.scATAC_Peaks["Peak_index"].tolist().index(p) for p in peak_ids]
    )
    A = loaded_data.scATAC_read_count_matrix[peak_index, :].toarray()
    atac_vec = np.zeros(loaded_data.scATAC_cells.shape[0], dtype=int)
    for s, name in enumerate(common_samples, start=1):
        atac_vec[loaded_data.scATAC_cells["subject_ID"].values == name] = s
    A_sample = np.asarray(candidate_circuits.Peak_log2Count, dtype=float)
    P = peak_index.size

    # --- RNA data setup ----------------------------------------------------
    gene_syms = candidate_circuits.Genes["Gene_symbols"].tolist()
    gene_index = np.array(
        [loaded_data.scRNA_Genes["Gene_symbols"].tolist().index(g) for g in gene_syms]
    )
    R = loaded_data.scRNA_read_count_matrix[gene_index, :].toarray()
    rna_vec = np.zeros(loaded_data.scRNA_cells.shape[0], dtype=int)
    for s, name in enumerate(common_samples, start=1):
        rna_vec[loaded_data.scRNA_cells["subject_ID"].values == name] = s
    R_sample = np.asarray(candidate_circuits.Gene_log2Count, dtype=float)
    G = gene_index.size

    # --- Priors and initial state -----------------------------------------
    B_prior_mean = initial_model.B_mean.copy()
    B_prior_var = initial_model.B_var.copy()
    B_prior_prob = initial_model.B_prob.copy()
    B = initial_model.B_prior.copy()
    B_state = np.asarray(candidate_circuits.TF_Peak_Binding.todense(), dtype=float)

    L_prior_mean = initial_model.L_mean.copy()
    L_prior_var = float(initial_model.L_var)
    L_prior_prob = initial_model.L_prob.copy()
    L = initial_model.L_prior.copy()
    L_state = np.asarray(candidate_circuits.Peak_Gene_looping.todense(), dtype=float)

    T_prior_mean = initial_model.T_mean.copy()
    T_prior_var = initial_model.T_var.copy()
    T_sample = T_prior_mean.copy()
    M = T_sample.shape[0]

    T_A = np.zeros((M, loaded_data.scATAC_cells.shape[0]))
    for s in range(1, S + 1):
        idx = np.flatnonzero(atac_vec == s)
        for m in range(M):
            T_A[m, idx] = rng.normal(
                loc=T_prior_mean[m, s - 1],
                scale=np.sqrt(abs(T_prior_var[m, s - 1])),
                size=idx.size,
            )
    T_R = np.zeros((M, loaded_data.scRNA_cells.shape[0]))
    for s in range(1, S + 1):
        idx = np.flatnonzero(rna_vec == s)
        for m in range(M):
            T_R[m, idx] = rng.normal(
                loc=T_prior_mean[m, s - 1],
                scale=np.sqrt(abs(T_prior_var[m, s - 1])),
                size=idx.size,
            )
    tfa = TFA(T_A=T_A, T_R=T_R, T_sample=T_sample)

    # --- Noise initial values ---------------------------------------------
    alpha_A = 1.0
    beta_A = 1.0
    sigma_A_noise = inv_gamma(
        rng,
        shape=alpha_A + 0.5,
        rate=(beta_A + float(((A_sample - B @ tfa.T_sample) ** 2).sum())) / (2 * P * S),
    )
    alpha_R = 1.0
    beta_R = 1.0
    sigma_R_noise = inv_gamma(
        rng,
        shape=alpha_R + 0.5,
        rate=(beta_R + float(((R_sample - L.T @ (B @ tfa.T_sample)) ** 2).sum())) / (2 * G * S),
    )

    # --- Gibbs loop --------------------------------------------------------
    # Latent-bug fix: original R divides by 0 when iteration_num < 6.
    iteration_seg = max(1, round(iteration_num / 10))
    B_state_frq = B_state.copy()
    L_state_frq = L_state.copy()

    if verbose:
        print("MAGICAL integration starts ...")

    Noise_parameters = np.zeros((iteration_num, 2))
    for i in range(1, iteration_num + 1):
        # Step 1
        tfa = tf_activity_sampling(
            A, A_sample, atac_vec, R, R_sample, rna_vec,
            tfa, T_prior_mean, T_prior_var, B, B_state, sigma_A_noise,
            P, G, M, S, rng,
        )
        # Step 2
        B = tf_peak_binding_sampling(
            A, A_sample, atac_vec, tfa,
            B, B_state, B_prior_mean, B_prior_var, sigma_A_noise,
            P, G, M, S, rng,
        )
        # Step 3
        B, B_state = tf_peak_binary_binding_sampling(
            A, A_sample, atac_vec, tfa,
            B, B_state, B_prior_mean, B_prior_var, B_prior_prob,
            sigma_A_noise, P, G, M, S, rng,
        )
        B_state_frq = B_state_frq + B_state

        # Step 4
        L = peak_gene_looping_sampling(
            R, R_sample, rna_vec, tfa,
            B, L, L_state, L_prior_mean, L_prior_var, sigma_R_noise,
            P, G, M, S, rng,
        )
        # Step 5
        L, L_state = peak_gene_binary_looping_sampling(
            R, R_sample, rna_vec, tfa,
            B, L, L_state, L_prior_mean, L_prior_var, L_prior_prob,
            sigma_R_noise, P, G, M, S, rng,
        )
        L_state_frq = L_state_frq + L_state

        # Step 6: noise variance updates
        sigma_A_noise = inv_gamma(
            rng, shape=alpha_A + 0.5,
            rate=(beta_A + float(((A_sample - B @ tfa.T_sample) ** 2).sum())) / (2 * P * S),
        )
        sigma_R_noise = inv_gamma(
            rng, shape=alpha_R + 0.5,
            rate=(beta_R + float(((R_sample - L.T @ (B @ tfa.T_sample)) ** 2).sum())) / (2 * G * S),
        )
        Noise_parameters[i - 1, 0] = sigma_A_noise
        Noise_parameters[i - 1, 1] = sigma_R_noise

        if verbose and (i % iteration_seg == 0):
            print(f"MAGICAL finished {10 * i // iteration_seg} percent")

    return EstimationResult(
        TF_Peak_Binding_prob=B_state_frq / (iteration_num + 1),
        Peak_Gene_Looping_prob=L_state_frq / (iteration_num + 1),
        Noise_parameters=Noise_parameters,
    )
