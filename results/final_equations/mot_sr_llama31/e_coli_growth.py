"""Final E. coli Growth equation structure from MOT-SR (LLaMA-3.1).

The fitted coefficient vector must be supplied in params.
"""

import numpy as np


def equation(
    b: np.ndarray,
    s: np.ndarray,
    temp: np.ndarray,
    pH: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Return the growth rate using a ten-entry fitted coefficient vector."""
    growth_rate = (
        params[0]
        * b ** params[1]
        * s ** params[2]
        * temp ** params[3]
        * (
            1
            + (
                params[4] * (s ** params[5] * (np.sin(params[6] * pH) + 1)) / 2
                + temp ** params[7] * (np.cos(params[8] * temp) + 1)
            )
        ) ** params[9]
    )
    return growth_rate
