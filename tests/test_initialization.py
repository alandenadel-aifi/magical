"""Tests for ``magical.initialization`` mirroring the R suite."""

from __future__ import annotations

import numpy as np


def test_returns_initial_model_with_all_expected_fields(fixture) -> None:
    im = fixture.initial_model
    for name in (
        "T_prior",
        "T_mean",
        "T_var",
        "B_prior",
        "B_mean",
        "B_var",
        "B_prob",
        "L_prior",
        "L_mean",
        "L_var",
        "L_prob",
    ):
        assert hasattr(im, name), f"missing field {name}"


def test_T_prior_and_T_mean_match_input_tf_log2count(fixture) -> None:
    tf_log2 = fixture.candidate_circuits.TF_log2Count
    np.testing.assert_allclose(fixture.initial_model.T_prior, tf_log2)
    np.testing.assert_allclose(fixture.initial_model.T_mean, tf_log2)


def test_T_var_is_broadcast_across_samples(fixture) -> None:
    im = fixture.initial_model
    T_var = im.T_var
    # Every column of T_var should equal the row-variance of TF_log2Count
    for s in range(T_var.shape[1]):
        np.testing.assert_allclose(T_var[:, 0], T_var[:, s])


def test_B_prior_zero_outside_candidate_binding(fixture) -> None:
    im = fixture.initial_model
    mask = np.asarray(fixture.candidate_circuits.TF_Peak_Binding.todense()) > 0
    assert np.all(im.B_prior[~mask] == 0.0)


def test_B_prob_is_in_zero_one_at_candidate_positions(fixture) -> None:
    im = fixture.initial_model
    mask = np.asarray(fixture.candidate_circuits.TF_Peak_Binding.todense()) > 0
    vals = im.B_prob[mask]
    # B_prob = 1 - p-value, so must lie in [0, 1]
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0


def test_B_var_is_row_shape_and_positive(fixture) -> None:
    im = fixture.initial_model
    M = fixture.candidate_circuits.TFs.shape[0]
    assert im.B_var.shape == (1, M)
    assert np.all(im.B_var > 0)


def test_L_prior_zero_outside_candidate_looping(fixture) -> None:
    im = fixture.initial_model
    mask = np.asarray(fixture.candidate_circuits.Peak_Gene_looping.todense()) > 0
    assert np.all(im.L_prior[~mask] == 0.0)


def test_L_prob_is_in_zero_one_at_candidate_positions(fixture) -> None:
    im = fixture.initial_model
    mask = np.asarray(fixture.candidate_circuits.Peak_Gene_looping.todense()) > 0
    vals = im.L_prob[mask]
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0


def test_L_var_is_positive_scalar(fixture) -> None:
    L_var = fixture.initial_model.L_var
    assert isinstance(L_var, float)
    assert L_var > 0


def test_dimensions_match_candidate_shapes(fixture) -> None:
    im = fixture.initial_model
    cc = fixture.candidate_circuits
    M = cc.TFs.shape[0]
    P = cc.Peaks.shape[0]
    G = cc.Genes.shape[0]
    S = len(fixture.loaded_data.common_samples)
    assert im.T_prior.shape == (M, S)
    assert im.T_var.shape == (M, S)
    assert im.B_prior.shape == (P, M)
    assert im.B_prob.shape == (P, M)
    assert im.L_prior.shape == (P, G)
    assert im.L_prob.shape == (P, G)


def test_initialization_is_deterministic(make_fixture) -> None:
    a = make_fixture(seed=42)
    b = make_fixture(seed=42)
    np.testing.assert_allclose(a.initial_model.B_prior, b.initial_model.B_prior)
    np.testing.assert_allclose(a.initial_model.L_prior, b.initial_model.L_prior)
    np.testing.assert_allclose(a.initial_model.T_var, b.initial_model.T_var)
