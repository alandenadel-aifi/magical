"""Dataclass mirrors of the R list structures used by MAGICAL.

These mirror the return shapes of the R functions in
``R/MAGICAL_functions.R`` so the numpy port can consume the same fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Loaded data (R: return value of ``Data_loading``)
# ---------------------------------------------------------------------------

@dataclass
class LoadedData:
    """Multi-omics inputs after ``Data_loading``."""

    common_samples: list[str]
    scATAC_Peaks: pd.DataFrame            # noqa: N815 - R name preserved
    scATAC_cells: pd.DataFrame            # noqa: N815
    scATAC_read_count_matrix: sparse.spmatrix  # noqa: N815
    scRNA_Genes: pd.DataFrame             # noqa: N815
    scRNA_cells: pd.DataFrame             # noqa: N815
    scRNA_read_count_matrix: sparse.spmatrix  # noqa: N815


# ---------------------------------------------------------------------------
# Candidate circuits (R: return value of ``Candidate_circuits_construction_*``)
# ---------------------------------------------------------------------------

@dataclass
class CandidateCircuits:
    """Candidate TFs / peaks / genes and their per-sample summaries."""

    TFs: pd.DataFrame                     # noqa: N815 - R name preserved
    TF_log2Count: np.ndarray              # noqa: N815 (M, S)
    Peaks: pd.DataFrame                   # noqa: N815
    Peak_log2Count: np.ndarray            # noqa: N815 (P, S)
    Genes: pd.DataFrame                   # noqa: N815
    Gene_log2Count: np.ndarray            # noqa: N815 (G, S)
    TF_Peak_Binding: sparse.spmatrix      # noqa: N815 (P, M)  binary 0/1
    Peak_Gene_looping: sparse.spmatrix    # noqa: N815 (P, G)  binary 0/1


# ---------------------------------------------------------------------------
# Initial model (R: return value of ``MAGICAL_initialization``)
# ---------------------------------------------------------------------------

@dataclass
class InitialModel:
    """Prior distributions for the Gibbs sampler."""

    T_prior: np.ndarray                   # noqa: N815 (M, S)
    T_mean: np.ndarray                    # noqa: N815 (M, S)
    T_var: np.ndarray                     # noqa: N815 (M, S)
    B_prior: np.ndarray                   # noqa: N815 (P, M) dense
    B_mean: np.ndarray                    # noqa: N815 (P, M) dense
    B_var: np.ndarray                     # noqa: N815 (1, M) row-vector of variances
    B_prob: np.ndarray                    # noqa: N815 (P, M) dense
    L_prior: np.ndarray                   # noqa: N815 (P, G) dense
    L_mean: np.ndarray                    # noqa: N815 (P, G) dense
    L_var: float                          # noqa: N815 - scalar
    L_prob: np.ndarray                    # noqa: N815 (P, G) dense


# ---------------------------------------------------------------------------
# Sampler state (R: the ``TFA`` list of three matrices)
# ---------------------------------------------------------------------------

@dataclass
class TFA:
    """Hidden TF activity state passed between Gibbs steps."""

    T_A: np.ndarray                       # noqa: N815 (M, N_atac_cells)
    T_R: np.ndarray                       # noqa: N815 (M, N_rna_cells)
    T_sample: np.ndarray                  # noqa: N815 (M, S)


# ---------------------------------------------------------------------------
# Estimation result (R: return value of ``MAGICAL_estimation``)
# ---------------------------------------------------------------------------

@dataclass
class EstimationResult:
    """Posterior summaries returned by ``magical.estimation``."""

    TF_Peak_Binding_prob: np.ndarray      # noqa: N815 (P, M) in [0, 1]
    Peak_Gene_Looping_prob: np.ndarray    # noqa: N815 (P, G) in [0, 1]
    Noise_parameters: np.ndarray          # noqa: N815 (iteration_num, 2)

    def as_dict(self) -> dict[str, Any]:
        """R-style dict view (parity with the R return list)."""
        return {
            "TF_Peak_Binding_prob": self.TF_Peak_Binding_prob,
            "Peak_Gene_Looping_prob": self.Peak_Gene_Looping_prob,
            "Noise_parameters": self.Noise_parameters,
        }
