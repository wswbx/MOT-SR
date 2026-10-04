"""The author-supplied final EMRI correction and its fitted parameters."""

from typing import Sequence

import numpy as np


PARAMS = (
    3.318812654787899,
    -11.17336476987488,
    -3138.3134817337836,
    0.46344189558754767,
    0.652276432405609,
    -0.09656828719277089,
    -0.9677413934633297,
    0.7694343703596733,
    -0.7386844558157266,
    0.27297654515575154,
)


def delta_p_dot(
    p: np.ndarray,
    e: np.ndarray,
    eta: np.ndarray,
    params: Sequence[float] = PARAMS,
) -> np.ndarray:
    """Evaluate the supplied expression on broadcast-compatible inputs.

    The expression uses params[0:3]. The complete ten-entry parameter
    vector is retained in PARAMS and params.json for archival purposes.
    Inputs must follow the variable and unit conventions of the EMRI data.
    """
    p = np.asarray(p, dtype=float)
    e = np.asarray(e, dtype=float)
    eta = np.asarray(eta, dtype=float)
    e_squared = e**2
    return (
        (1.0 + params[2] * eta * p / (1.0 + e_squared))
        * (1.0 + e_squared / (1.0 + e_squared))
        / (1.0 + eta * np.exp(params[0] * p + params[1] * e))
    )
