"""RNG helpers.

The R code truncates every ``rnorm`` draw at +/-3 before scaling by the
posterior standard deviation. We keep the same behaviour so posterior
distributions match up to statistical equivalence.
"""

from __future__ import annotations

import numpy as np


def truncated_standard_normal(rng: np.random.Generator, size: int | tuple[int, ...] = 1) -> np.ndarray:
    """Draw N(0, 1) samples clipped to ``[-3, 3]``.

    Matches the R idiom::

        aa = rnorm(k)
        aa[which(aa - 3 > 0)] = 3
        aa[which(aa + 3 < 0)] = -3
    """
    x = rng.standard_normal(size=size)
    return np.clip(x, -3.0, 3.0)


def inv_gamma(rng: np.random.Generator, shape: float, rate: float) -> float:
    """Draw ``1 / Gamma(shape, rate=rate)``.

    R spelling: ``1 / rgamma(1, shape = shape, rate = rate)``.
    numpy's ``rng.gamma`` takes ``scale = 1/rate``.
    """
    g = rng.gamma(shape=shape, scale=1.0 / rate)
    return float(1.0 / g)
