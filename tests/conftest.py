"""Shared pytest fixtures.

Mirrors [re-implementation/tests/testthat/helper-synthetic.R](../re-implementation/tests/testthat/helper-synthetic.R).
Same P/G/M/S so scale expectations line up 1:1 with the R suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from magical._rng import inv_gamma
from magical.initialization import magical_initialization
from magical.types import CandidateCircuits, InitialModel, LoadedData, TFA


@dataclass
class Fixture:
    loaded_data: LoadedData
    candidate_circuits: CandidateCircuits
    initial_model: InitialModel


def make_synthetic_fixture(
    seed: int = 1,
    n_samples: int = 6,
    n_tfs: int = 5,
    n_peaks: int = 20,
    n_genes: int = 10,
    cells_per_sample_atac: int = 8,
    cells_per_sample_rna: int = 8,
    binding_density: float = 0.35,
    looping_density: float = 0.30,
) -> Fixture:
    rng = np.random.default_rng(seed)

    S = n_samples
    M = n_tfs
    P = n_peaks
    G = n_genes

    sample_names = [f"S{i + 1}" for i in range(S)]
    tf_names = [f"TF{i + 1}" for i in range(M)]
    peak_ids = [f"peak{i + 1}" for i in range(P)]
    gene_names = [f"gene{i + 1}" for i in range(G)]

    TF_log2Count = rng.normal(loc=5.0, scale=1.0, size=(M, S))
    Peak_log2Count = rng.normal(loc=3.0, scale=1.0, size=(P, S))
    Gene_log2Count = rng.normal(loc=4.0, scale=1.0, size=(G, S))

    tf_peak = (rng.random((P, M)) < binding_density).astype(np.int8)
    for m in range(M):
        if tf_peak[:, m].sum() == 0:
            tf_peak[0, m] = 1
    TF_Peak_Binding = sparse.csr_matrix(tf_peak)

    peak_gene = (rng.random((P, G)) < looping_density).astype(np.int8)
    for g in range(G):
        if peak_gene[:, g].sum() == 0:
            peak_gene[0, g] = 1
    Peak_Gene_looping = sparse.csr_matrix(peak_gene)

    scATAC_Peaks = pd.DataFrame(
        {
            "Peak_index": peak_ids,
            "chr": "chr1",
            "point1": (np.arange(P) + 1) * 1000,
            "point2": (np.arange(P) + 1) * 1000 + 500,
        }
    )

    def build_cellmeta(prefix: str, cells_per_sample: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for s_idx, name in enumerate(sample_names):
            for k in range(cells_per_sample):
                rows.append(
                    {
                        "cell_index": -1,  # filled below
                        "cell_barcode": f"{prefix}_{name}_{k + 1}",
                        "cell_type": "T",
                        "subject_ID": name,
                    }
                )
        meta = pd.DataFrame(rows)
        meta["cell_index"] = np.arange(1, len(meta) + 1)
        return meta

    scATAC_cells = build_cellmeta("atac", cells_per_sample_atac)
    scRNA_cells = build_cellmeta("rna", cells_per_sample_rna)
    scRNA_cells["condition"] = "cond"

    def build_readcount(n_features: int, cells: pd.DataFrame, mean_count: float) -> sparse.spmatrix:
        counts = rng.poisson(lam=mean_count, size=(n_features, cells.shape[0])).astype(float)
        return sparse.csr_matrix(counts)

    scATAC_read_count_matrix = build_readcount(P, scATAC_cells, mean_count=2.0)

    rna_feature_names = list(dict.fromkeys([*gene_names, *tf_names]))
    scRNA_Genes = pd.DataFrame(
        {
            "Gene_index": np.arange(1, len(rna_feature_names) + 1),
            "Gene_symbols": rna_feature_names,
        }
    )
    scRNA_read_count_matrix = build_readcount(len(rna_feature_names), scRNA_cells, mean_count=3.0)

    loaded_data = LoadedData(
        common_samples=sample_names,
        scATAC_Peaks=scATAC_Peaks,
        scATAC_cells=scATAC_cells,
        scATAC_read_count_matrix=scATAC_read_count_matrix,
        scRNA_Genes=scRNA_Genes,
        scRNA_cells=scRNA_cells,
        scRNA_read_count_matrix=scRNA_read_count_matrix,
    )

    candidate_circuits = CandidateCircuits(
        TFs=pd.DataFrame({"TF_id": tf_names, "TF_name": tf_names}),
        TF_log2Count=TF_log2Count,
        Peaks=pd.DataFrame(
            {
                "Peak_index": peak_ids,
                "chr": "chr1",
                "point1": (np.arange(P) + 1) * 1000,
                "point2": (np.arange(P) + 1) * 1000 + 500,
            }
        ),
        Peak_log2Count=Peak_log2Count,
        Genes=pd.DataFrame(
            {
                "Gene_symbols": gene_names,
                "chr": "chr1",
                "TSS": (np.arange(G) + 1) * 5000,
            }
        ),
        Gene_log2Count=Gene_log2Count,
        TF_Peak_Binding=TF_Peak_Binding,
        Peak_Gene_looping=Peak_Gene_looping,
    )

    initial_model = magical_initialization(loaded_data, candidate_circuits)

    return Fixture(
        loaded_data=loaded_data,
        candidate_circuits=candidate_circuits,
        initial_model=initial_model,
    )


@pytest.fixture()
def fixture() -> Fixture:
    """Default fixture used by most tests."""
    return make_synthetic_fixture(seed=1)


@pytest.fixture()
def make_fixture():
    """Factory in case a test needs a differently-seeded fixture."""
    return make_synthetic_fixture


# ---------------------------------------------------------------------------
# Sampler-input bundle (mirror of R's ``make_sampler_inputs``)
# ---------------------------------------------------------------------------

@dataclass
class SamplerInputs:
    A: np.ndarray
    A_sample: np.ndarray
    ATAC_Cell_Sample_vector: np.ndarray
    R: np.ndarray
    R_sample: np.ndarray
    RNA_Cell_Sample_vector: np.ndarray
    tfa: TFA
    T_prior_mean: np.ndarray
    T_prior_var: np.ndarray
    B: np.ndarray
    B_state: np.ndarray
    B_prior_mean: np.ndarray
    B_prior_var: np.ndarray
    B_prior_prob: np.ndarray
    L: np.ndarray
    L_state: np.ndarray
    L_prior_mean: np.ndarray
    L_prior_var: float
    L_prior_prob: np.ndarray
    sigma_A_noise: float
    sigma_R_noise: float
    P: int
    G: int
    M: int
    S: int


def make_sampler_inputs(fx: Fixture, seed: int = 123) -> SamplerInputs:
    """Build a valid argument bundle for the individual samplers.

    Mirrors the setup block at the top of ``MAGICAL_estimation`` in R so tests
    can exercise a single Gibbs step in isolation.
    """
    rng = np.random.default_rng(seed)

    ld = fx.loaded_data
    cc = fx.candidate_circuits
    im = fx.initial_model

    S = len(ld.common_samples)

    # ATAC data
    peak_ids = cc.Peaks["Peak_index"].tolist()
    peak_idx = np.array(
        [ld.scATAC_Peaks["Peak_index"].tolist().index(p) for p in peak_ids]
    )
    A = ld.scATAC_read_count_matrix[peak_idx, :].toarray()
    atac_vec = np.zeros(ld.scATAC_cells.shape[0], dtype=int)
    for s, name in enumerate(ld.common_samples, start=1):
        atac_vec[ld.scATAC_cells["subject_ID"].values == name] = s
    A_sample = np.asarray(cc.Peak_log2Count, dtype=float)
    P = peak_idx.size

    # RNA data
    gene_syms = cc.Genes["Gene_symbols"].tolist()
    gene_idx = np.array(
        [ld.scRNA_Genes["Gene_symbols"].tolist().index(g) for g in gene_syms]
    )
    R = ld.scRNA_read_count_matrix[gene_idx, :].toarray()
    rna_vec = np.zeros(ld.scRNA_cells.shape[0], dtype=int)
    for s, name in enumerate(ld.common_samples, start=1):
        rna_vec[ld.scRNA_cells["subject_ID"].values == name] = s
    R_sample = np.asarray(cc.Gene_log2Count, dtype=float)
    G = gene_idx.size

    B_prior_mean = im.B_mean.copy()
    B_prior_var = im.B_var.copy()
    B_prior_prob = im.B_prob.copy()
    B = im.B_prior.copy()
    B_state = np.asarray(cc.TF_Peak_Binding.todense(), dtype=float)

    L_prior_mean = im.L_mean.copy()
    L_prior_var = float(im.L_var)
    L_prior_prob = im.L_prob.copy()
    L = im.L_prior.copy()
    L_state = np.asarray(cc.Peak_Gene_looping.todense(), dtype=float)

    T_prior_mean = im.T_mean.copy()
    T_prior_var = im.T_var.copy()
    T_sample = T_prior_mean.copy()
    M = T_sample.shape[0]

    T_A = np.zeros((M, ld.scATAC_cells.shape[0]))
    for s in range(1, S + 1):
        idx = np.flatnonzero(atac_vec == s)
        for m in range(M):
            T_A[m, idx] = rng.normal(
                loc=T_prior_mean[m, s - 1],
                scale=np.sqrt(abs(T_prior_var[m, s - 1])),
                size=idx.size,
            )
    T_R = np.zeros((M, ld.scRNA_cells.shape[0]))
    for s in range(1, S + 1):
        idx = np.flatnonzero(rna_vec == s)
        for m in range(M):
            T_R[m, idx] = rng.normal(
                loc=T_prior_mean[m, s - 1],
                scale=np.sqrt(abs(T_prior_var[m, s - 1])),
                size=idx.size,
            )
    tfa = TFA(T_A=T_A, T_R=T_R, T_sample=T_sample)

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

    return SamplerInputs(
        A=A, A_sample=A_sample, ATAC_Cell_Sample_vector=atac_vec,
        R=R, R_sample=R_sample, RNA_Cell_Sample_vector=rna_vec,
        tfa=tfa,
        T_prior_mean=T_prior_mean, T_prior_var=T_prior_var,
        B=B, B_state=B_state,
        B_prior_mean=B_prior_mean, B_prior_var=B_prior_var,
        B_prior_prob=B_prior_prob,
        L=L, L_state=L_state,
        L_prior_mean=L_prior_mean, L_prior_var=L_prior_var,
        L_prior_prob=L_prior_prob,
        sigma_A_noise=sigma_A_noise, sigma_R_noise=sigma_R_noise,
        P=P, G=G, M=M, S=S,
    )


@pytest.fixture()
def sampler_inputs(fixture) -> SamplerInputs:
    return make_sampler_inputs(fixture, seed=123)
