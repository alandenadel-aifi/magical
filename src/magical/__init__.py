"""Top-level ``magical`` package."""

from __future__ import annotations

from .api import estimation
from .initialization import magical_initialization
from .output import magical_circuits_output
from .types import (
    CandidateCircuits,
    EstimationResult,
    InitialModel,
    LoadedData,
    TFA,
)

__all__ = [
    "estimation",
    "magical_initialization",
    "magical_circuits_output",
    "CandidateCircuits",
    "EstimationResult",
    "InitialModel",
    "LoadedData",
    "TFA",
]
