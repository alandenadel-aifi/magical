"""API contract tests mirroring ``test-api-contract.R``."""

from __future__ import annotations

import numpy as np

from magical import estimation
from magical.types import EstimationResult


def _run(fixture, iteration_num=8, seed=1):
    return estimation(
        fixture.loaded_data,
        fixture.candidate_circuits,
        fixture.initial_model,
        iteration_num=iteration_num,
        backend="numpy",
        seed=seed,
    )


def test_returns_expected_shape_and_names(fixture) -> None:
    result = _run(fixture)
    assert isinstance(result, EstimationResult)
    d = result.as_dict()
    assert set(d) == {
        "TF_Peak_Binding_prob",
        "Peak_Gene_Looping_prob",
        "Noise_parameters",
    }
    P, M = fixture.initial_model.B_prior.shape
    P2, G = fixture.initial_model.L_prior.shape
    assert P == P2
    assert result.TF_Peak_Binding_prob.shape == (P, M)
    assert result.Peak_Gene_Looping_prob.shape == (P, G)
    assert result.Noise_parameters.shape == (8, 2)


def test_probabilities_lie_in_zero_one(fixture) -> None:
    result = _run(fixture)
    assert result.TF_Peak_Binding_prob.min() >= 0.0
    assert result.TF_Peak_Binding_prob.max() <= 1.0
    assert result.Peak_Gene_Looping_prob.min() >= 0.0
    assert result.Peak_Gene_Looping_prob.max() <= 1.0


def test_noise_variances_are_positive_and_finite(fixture) -> None:
    result = _run(fixture)
    assert np.all(result.Noise_parameters > 0)
    assert np.all(np.isfinite(result.Noise_parameters))


def test_zero_prior_positions_stay_zero_in_posterior(fixture) -> None:
    result = _run(fixture)
    b_mask = np.asarray(fixture.candidate_circuits.TF_Peak_Binding.todense()) == 0
    l_mask = np.asarray(fixture.candidate_circuits.Peak_Gene_looping.todense()) == 0
    assert np.all(result.TF_Peak_Binding_prob[b_mask] == 0.0)
    assert np.all(result.Peak_Gene_Looping_prob[l_mask] == 0.0)
