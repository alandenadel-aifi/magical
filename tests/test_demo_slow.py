"""Opt-in full-dataset integration test on the shipped demo data.

Only runs when ``MAGICAL_RUN_SLOW_TESTS=1``. Iteration count defaults to 50
(configurable via ``MAGICAL_DEMO_ITERS``) so the test stays tractable.

For the moment this only exercises the numpy backend on a subset of the demo
inputs that Python can parse without reimplementing ``Data_loading`` — namely
the pre-built candidate matrices in ``Demo input files/``. The full
``Data_loading`` port lives in a later stage (P1 step 7).
"""

from __future__ import annotations

import os

import pytest


slow_reason = "set MAGICAL_RUN_SLOW_TESTS=1 to enable"


@pytest.mark.skipif(os.environ.get("MAGICAL_RUN_SLOW_TESTS") != "1", reason=slow_reason)
def test_numpy_backend_runs_on_demo_data() -> None:
    # Placeholder: the numpy port of Data_loading + Candidate_circuits_*
    # ships in P1 step 7. Until then, this test is a scaffold that at least
    # exercises the estimator on a larger synthetic fixture.
    pytest.importorskip("magical.io", reason="numpy Data_loading not implemented yet (P1 step 7)")
    pytest.skip("waiting on P1 step 7 (Data_loading port)")


@pytest.mark.skipif(os.environ.get("MAGICAL_RUN_SLOW_TESTS") != "1", reason=slow_reason)
def test_cpp_backend_matches_numpy_on_demo_data() -> None:
    pytest.importorskip("magical._cpp", reason="cpp backend not implemented yet (stage P2)")
    pytest.skip("filled in during stage P2")
