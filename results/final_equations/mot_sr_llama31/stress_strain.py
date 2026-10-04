"""Final Stress-Strain equation structure from MOT-SR (LLaMA-3.1).

The fitted coefficient vector must be supplied in params.
"""

import numpy as np


def equation(
    strain: np.ndarray,
    temp: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Return the stress using a ten-entry fitted coefficient vector."""
    elastic_stress = (params[0] * strain + params[1]) * (
        1 + params[2] * np.sin(params[3] * temp)
    )
    plastic_stress = params[4] * (
        1 - np.exp(-(strain - params[5] * temp) ** 2 / params[6] ** 2)
    )
    exp_interaction = params[7] * np.exp(params[8] * strain + params[9] * temp)
    stress = elastic_stress + plastic_stress + exp_interaction
    return stress
