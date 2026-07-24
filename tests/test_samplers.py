"""Sampler unit tests mirroring ``re-implementation/tests/testthat/test-samplers.R``."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from magical.samplers import (
    peak_gene_binary_looping_sampling,
    peak_gene_looping_sampling,
    tf_activity_sampling,
    tf_peak_binary_binding_sampling,
    tf_peak_binding_sampling,
)


def _call_tfa(inputs, seed):
    tfa = deepcopy(inputs.tfa)
    rng = np.random.default_rng(seed)
    out = tf_activity_sampling(
        A=inputs.A, A_sample=inputs.A_sample,
        ATAC_Cell_Sample_vector=inputs.ATAC_Cell_Sample_vector,
        R=inputs.R, R_sample=inputs.R_sample,
        RNA_Cell_Sample_vector=inputs.RNA_Cell_Sample_vector,
        tfa=tfa,
        T_prior_mean=inputs.T_prior_mean, T_prior_var=inputs.T_prior_var,
        B=inputs.B, B_state=inputs.B_state,
        sigma_A_noise=inputs.sigma_A_noise,
        P=inputs.P, G=inputs.G, M=inputs.M, S=inputs.S,
        rng=rng,
    )
    return out


# --- Step 1: TF activity ---------------------------------------------------

def test_tf_activity_returns_correct_component_shapes(sampler_inputs) -> None:
    out = _call_tfa(sampler_inputs, seed=1)
    assert out.T_sample.shape == sampler_inputs.tfa.T_sample.shape
    assert out.T_A.shape == sampler_inputs.tfa.T_A.shape
    assert out.T_R.shape == sampler_inputs.tfa.T_R.shape


def test_tf_activity_output_is_finite(sampler_inputs) -> None:
    out = _call_tfa(sampler_inputs, seed=1)
    assert np.all(np.isfinite(out.T_sample))
    assert np.all(np.isfinite(out.T_A))
    assert np.all(np.isfinite(out.T_R))


def test_tf_activity_is_deterministic_under_same_seed(sampler_inputs) -> None:
    a = _call_tfa(sampler_inputs, seed=99)
    b = _call_tfa(sampler_inputs, seed=99)
    np.testing.assert_allclose(a.T_sample, b.T_sample)
    np.testing.assert_allclose(a.T_A, b.T_A)
    np.testing.assert_allclose(a.T_R, b.T_R)


# --- Step 2: TF-peak binding (continuous) ---------------------------------

def _call_binding(inputs, seed):
    rng = np.random.default_rng(seed)
    return tf_peak_binding_sampling(
        A=inputs.A, A_sample=inputs.A_sample,
        ATAC_Cell_Sample_vector=inputs.ATAC_Cell_Sample_vector,
        tfa=inputs.tfa,
        B=inputs.B, B_state=inputs.B_state,
        B_prior_mean=inputs.B_prior_mean, B_prior_var=inputs.B_prior_var,
        sigma_A_noise=inputs.sigma_A_noise,
        P=inputs.P, G=inputs.G, M=inputs.M, S=inputs.S,
        rng=rng,
    )


def test_binding_returns_same_shape_as_B(sampler_inputs) -> None:
    B_out = _call_binding(sampler_inputs, seed=1)
    assert B_out.shape == sampler_inputs.B.shape


def test_binding_is_zero_where_state_is_zero(sampler_inputs) -> None:
    B_out = _call_binding(sampler_inputs, seed=1)
    mask = sampler_inputs.B_state == 0
    assert np.all(B_out[mask] == 0.0)


def test_binding_is_deterministic_under_same_seed(sampler_inputs) -> None:
    a = _call_binding(sampler_inputs, seed=7)
    b = _call_binding(sampler_inputs, seed=7)
    np.testing.assert_allclose(a, b)


# --- Step 3: TF-peak binding STATE ----------------------------------------

def _call_binary_binding(inputs, seed):
    rng = np.random.default_rng(seed)
    return tf_peak_binary_binding_sampling(
        A=inputs.A, A_sample=inputs.A_sample,
        ATAC_Cell_Sample_vector=inputs.ATAC_Cell_Sample_vector,
        tfa=inputs.tfa,
        B=inputs.B, B_state=inputs.B_state,
        B_prior_mean=inputs.B_prior_mean, B_prior_var=inputs.B_prior_var,
        B_prior_prob=inputs.B_prior_prob,
        sigma_A_noise=inputs.sigma_A_noise,
        P=inputs.P, G=inputs.G, M=inputs.M, S=inputs.S,
        rng=rng,
    )


def test_binary_binding_returns_B_and_state_same_shape(sampler_inputs) -> None:
    B, B_state = _call_binary_binding(sampler_inputs, seed=1)
    assert B.shape == sampler_inputs.B.shape
    assert B_state.shape == sampler_inputs.B_state.shape


def test_binary_binding_state_values_are_binary(sampler_inputs) -> None:
    _, B_state = _call_binary_binding(sampler_inputs, seed=1)
    unique = np.unique(B_state)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_binary_binding_state_stays_zero_where_prior_prob_is_zero(sampler_inputs) -> None:
    _, B_state = _call_binary_binding(sampler_inputs, seed=1)
    mask = sampler_inputs.B_prior_prob == 0
    assert np.all(B_state[mask] == 0.0)


# --- Step 4: Peak-gene looping (continuous) -------------------------------

def _call_looping(inputs, seed):
    rng = np.random.default_rng(seed)
    return peak_gene_looping_sampling(
        R=inputs.R, R_sample=inputs.R_sample,
        RNA_Cell_Sample_vector=inputs.RNA_Cell_Sample_vector,
        tfa=inputs.tfa,
        B=inputs.B, L=inputs.L, L_state=inputs.L_state,
        L_prior_mean=inputs.L_prior_mean, L_prior_var=inputs.L_prior_var,
        sigma_R_noise=inputs.sigma_R_noise,
        P=inputs.P, G=inputs.G, M=inputs.M, S=inputs.S,
        rng=rng,
    )


def test_looping_returns_same_shape_as_L(sampler_inputs) -> None:
    L_out = _call_looping(sampler_inputs, seed=1)
    assert L_out.shape == sampler_inputs.L.shape


def test_looping_is_zero_where_state_is_zero(sampler_inputs) -> None:
    L_out = _call_looping(sampler_inputs, seed=1)
    mask = sampler_inputs.L_state == 0
    assert np.all(L_out[mask] == 0.0)


def test_looping_is_deterministic_under_same_seed(sampler_inputs) -> None:
    a = _call_looping(sampler_inputs, seed=13)
    b = _call_looping(sampler_inputs, seed=13)
    np.testing.assert_allclose(a, b)


# --- Step 5: Peak-gene looping STATE --------------------------------------

def _call_binary_looping(inputs, seed):
    rng = np.random.default_rng(seed)
    return peak_gene_binary_looping_sampling(
        R=inputs.R, R_sample=inputs.R_sample,
        RNA_Cell_Sample_vector=inputs.RNA_Cell_Sample_vector,
        tfa=inputs.tfa,
        B=inputs.B, L=inputs.L, L_state=inputs.L_state,
        L_prior_mean=inputs.L_prior_mean, L_prior_var=inputs.L_prior_var,
        L_prior_prob=inputs.L_prior_prob,
        sigma_R_noise=inputs.sigma_R_noise,
        P=inputs.P, G=inputs.G, M=inputs.M, S=inputs.S,
        rng=rng,
    )


def test_binary_looping_returns_L_and_state_same_shape(sampler_inputs) -> None:
    L, L_state = _call_binary_looping(sampler_inputs, seed=1)
    assert L.shape == sampler_inputs.L.shape
    assert L_state.shape == sampler_inputs.L_state.shape


def test_binary_looping_state_values_are_binary(sampler_inputs) -> None:
    _, L_state = _call_binary_looping(sampler_inputs, seed=1)
    unique = np.unique(L_state)
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_binary_looping_state_stays_zero_where_prior_prob_is_zero(sampler_inputs) -> None:
    _, L_state = _call_binary_looping(sampler_inputs, seed=1)
    mask = sampler_inputs.L_prior_prob == 0
    assert np.all(L_state[mask] == 0.0)
