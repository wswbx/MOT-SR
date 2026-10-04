"""Final Oscillation 1 equation structure from MOT-SR (LLaMA-3.1).

The fitted coefficient vector must be supplied in params.
"""

import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Return the acceleration using a ten-entry fitted coefficient vector."""
    linear_term = -params[0] * x + params[1] * v
    quadratic_term = -params[2] * x ** 2 + params[3] * v ** 2
    cubic_term = params[4] * x ** 3 + params[5] * v ** 3
    driving_term = -params[6] * x * v + params[7] * np.sin(params[8] * x)
    damping_term = params[9] * v
    weighted_terms = [
        ('linear_term', linear_term, 2.0),
        ('quadratic_term', quadratic_term, 1.5),
        ('cubic_term', cubic_term, 1.0),
        ('driving_term', driving_term, 2.0),
        ('damping_term', damping_term, 1.5),
    ]
    result = np.zeros_like(x)
    for term_name, term_value, weight in weighted_terms:
        result += weight * term_value
    return result
