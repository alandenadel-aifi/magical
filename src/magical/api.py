"""Public facade for MAGICAL.

Mirrors [re-implementation/R/MAGICAL_functions.R](../../re-implementation/R/MAGICAL_functions.R)'s
``backend = c("r", "cpp")`` pattern — Python-side we expose
``backend = {"numpy", "cpp"}`` where ``"cpp"`` is stage P2.
"""

from __future__ import annotations

from typing import Literal

from .estimation import magical_estimation as _magical_estimation_numpy
from .types import CandidateCircuits, EstimationResult, InitialModel, LoadedData


BackendName = Literal["numpy", "cpp"]


def estimation(
    loaded_data: LoadedData,
    candidate_circuits: CandidateCircuits,
    initial_model: InitialModel,
    iteration_num: int,
    *,
    backend: BackendName = "numpy",
    seed: int | None = None,
    verbose: bool = False,
) -> EstimationResult:
    """Public estimation facade."""
    if backend == "numpy":
        return _magical_estimation_numpy(
            loaded_data,
            candidate_circuits,
            initial_model,
            iteration_num=iteration_num,
            seed=seed,
            verbose=verbose,
        )
    if backend == "cpp":
        raise NotImplementedError(
            "cpp backend not implemented yet (stage P2). Use backend='numpy'."
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'cpp'")
