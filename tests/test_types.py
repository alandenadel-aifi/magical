"""Round-trip tests for the dataclass mirrors in ``magical.types``."""

from __future__ import annotations

import numpy as np

from magical.types import EstimationResult, InitialModel, TFA


def test_estimation_result_as_dict_has_r_style_names() -> None:
    result = EstimationResult(
        TF_Peak_Binding_prob=np.zeros((3, 2)),
        Peak_Gene_Looping_prob=np.zeros((3, 4)),
        Noise_parameters=np.zeros((5, 2)),
    )
    d = result.as_dict()
    assert set(d) == {
        "TF_Peak_Binding_prob",
        "Peak_Gene_Looping_prob",
        "Noise_parameters",
    }


def test_tfa_holds_three_matrices() -> None:
    tfa = TFA(
        T_A=np.zeros((2, 5)),
        T_R=np.zeros((2, 5)),
        T_sample=np.zeros((2, 3)),
    )
    assert tfa.T_A.shape == (2, 5)
    assert tfa.T_R.shape == (2, 5)
    assert tfa.T_sample.shape == (2, 3)


def test_initial_model_fields_are_declared() -> None:
    im = InitialModel(
        T_prior=np.zeros((2, 3)),
        T_mean=np.zeros((2, 3)),
        T_var=np.zeros((2, 3)),
        B_prior=np.zeros((4, 2)),
        B_mean=np.zeros((4, 2)),
        B_var=np.zeros((1, 2)),
        B_prob=np.zeros((4, 2)),
        L_prior=np.zeros((4, 5)),
        L_mean=np.zeros((4, 5)),
        L_var=0.5,
        L_prob=np.zeros((4, 5)),
    )
    assert im.L_var == 0.5
