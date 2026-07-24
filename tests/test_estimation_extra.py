"""Additional estimation tests mirroring ``test-estimation-extra.R``."""

from __future__ import annotations

import numpy as np
import pytest

from magical import estimation


def test_small_iteration_count_still_returns(fixture) -> None:
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=1, backend="numpy", seed=0,
    )
    assert result.Noise_parameters.shape == (1, 2)


def test_estimation_is_deterministic_under_same_seed(fixture) -> None:
    a = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=42,
    )
    b = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=42,
    )
    np.testing.assert_allclose(a.TF_Peak_Binding_prob, b.TF_Peak_Binding_prob)
    np.testing.assert_allclose(a.Peak_Gene_Looping_prob, b.Peak_Gene_Looping_prob)
    np.testing.assert_allclose(a.Noise_parameters, b.Noise_parameters)


def test_noise_parameters_row_count_matches_iterations(fixture) -> None:
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=13, backend="numpy", seed=0,
    )
    assert result.Noise_parameters.shape[0] == 13


def test_different_seed_produces_different_noise_trajectory(fixture) -> None:
    a = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=1,
    )
    b = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=6, backend="numpy", seed=2,
    )
    assert not np.allclose(a.Noise_parameters, b.Noise_parameters)


def test_cpp_backend_raises_not_implemented(fixture) -> None:
    with pytest.raises(NotImplementedError, match="cpp"):
        estimation(
            fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
            iteration_num=1, backend="cpp", seed=0,
        )


def test_binding_frequency_bounded_by_iteration_count(fixture) -> None:
    """Posterior prob is (initial + sum of per-iter states) / (iters + 1);
    all values must be in [0, 1] even at very high iteration counts."""
    result = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=20, backend="numpy", seed=0,
    )
    b_mask = np.asarray(fixture.candidate_circuits.TF_Peak_Binding.todense()) > 0
    vals = result.TF_Peak_Binding_prob[b_mask]
    # frequency count / (iters+1) is bounded by (iters+1)/(iters+1) = 1
    assert vals.max() <= 1.0 + 1e-12
    assert vals.min() >= 0.0
