"""Final Oscillation 2 equation structure from MOT-SR (LLaMA-3.1).

The fitted coefficient vector must be supplied in params.
"""

import numpy as np


def equation(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Return the acceleration using a ten-entry fitted coefficient vector."""
    dampening_term = params[0] * v
    natural_frequency_term = params[1] * np.exp(-params[2] * x) * (x + params[3] * v)
    driving_force_term = params[4] * np.sin(params[5] * (t + params[6] * x))
    true_balance_term = (
        params[7] * v * (v ** 2 + x ** 2)
        + params[8] * v * (v ** 2 - x ** 2)
    )
    return -(
        dampening_term
        + natural_frequency_term * (1 + params[9] * np.cos(v))
        + true_balance_term
        + driving_force_term
    )
