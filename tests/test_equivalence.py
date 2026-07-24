"""Cross-seed / cross-backend equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest

from magical import estimation


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays.

    Falls back to 1.0 when both flat vectors are constant (degenerate case
    where every entry is identical — correlation is undefined but the arrays
    agree perfectly).
    """
    x = np.asarray(a).ravel()
    y = np.asarray(b).ravel()
    if np.std(x) == 0 and np.std(y) == 0:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def test_numpy_vs_numpy_cross_seed_high_correlation(fixture) -> None:
    """Two independent numpy runs at different seeds should agree well.

    We use a small iteration count so tests stay fast; the correlation is
    a smoke check that the sampler is exploring roughly the same posterior.
    """
    n_iter = 20
    a = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=n_iter, backend="numpy", seed=1,
    )
    b = estimation(
        fixture.loaded_data, fixture.candidate_circuits, fixture.initial_model,
        iteration_num=n_iter, backend="numpy", seed=2,
    )
    corr_b = _correlation(a.TF_Peak_Binding_prob, b.TF_Peak_Binding_prob)
    corr_l = _correlation(a.Peak_Gene_Looping_prob, b.Peak_Gene_Looping_prob)
    assert corr_b >= 0.90, f"TF-Peak correlation too low: {corr_b:.3f}"
    assert corr_l >= 0.90, f"Peak-Gene correlation too low: {corr_l:.3f}"


def test_numpy_vs_cpp_placeholder(fixture) -> None:
    """Will be filled in during stage P2 (pybind11 backend)."""
    pytest.importorskip("magical._cpp", reason="cpp backend not implemented yet (stage P2)")
    pytest.skip("filled in during stage P2")
