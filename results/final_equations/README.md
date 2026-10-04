# Final equations

This directory archives four benchmark equation structures from MOT-SR
(LLaMA-3.1) and the final EMRI correction supplied by the authors.

## MOT-SR (LLaMA-3.1)

The directory name `mot_sr_llama31` uses lowercase snake_case, consistent
with Python module naming. Each file exposes the original `equation`
function interface and retains the executable mathematical structure.

| Task | File | Inputs | Fitted coefficients |
| --- | --- | --- | --- |
| Oscillation 1 | [oscillation_1.py](mot_sr_llama31/oscillation_1.py) | `x`, `v` | `params[0]` through `params[9]` |
| Oscillation 2 | [oscillation_2.py](mot_sr_llama31/oscillation_2.py) | `t`, `x`, `v` | `params[0]` through `params[9]` |
| E. coli Growth | [e_coli_growth.py](mot_sr_llama31/e_coli_growth.py) | `b`, `s`, `temp`, `pH` | `params[0]` through `params[9]` |
| Stress-Strain | [stress_strain.py](mot_sr_llama31/stress_strain.py) | `strain`, `temp` | `params[0]` through `params[9]` |

The original fitted coefficient vectors for these four benchmark equations
are currently unavailable. Numerical evaluation requires fitting their
coefficients on the corresponding training data. Parameter indexing,
fixed constants, operators and numerical function calls are preserved.

## EMRI correction

The executable expression is in [emri/equation.py](emri/equation.py), with
the complete supplied parameter vector in [emri/params.json](emri/params.json).

$$
\Delta\dot p(p,e,\eta)=
\left(1-3138.3134817337836\,\frac{\eta p}{1+e^2}\right)
\frac{1+\dfrac{e^2}{1+e^2}}
{1+\eta\exp\!\left(3.318812654787899\,p-11.17336476987488\,e\right)}.
$$

The expression uses `params[0]`, `params[1]` and `params[2]`. All ten
supplied parameter values are archived, including the seven entries that
do not appear in this expression.

Use `p`, `e` and `eta` according to the definitions and units of the EMRI
data-generation and evaluation pipeline. The implementation reproduces
the expression above directly.

## Usage

Run from the repository root in the existing MOT-SR Python environment.
These equation files require NumPy.

```python
import numpy as np
from results.final_equations.emri.equation import delta_p_dot

p = np.array([8.0, 10.0])
e = np.array([0.2, 0.4])
eta = np.array([1e-5, 1e-5])
correction = delta_p_dot(p, e, eta)
```

For benchmark evaluation, import the selected function and supply its
fitted coefficient vector:

```python
from results.final_equations.mot_sr_llama31.oscillation_1 import equation

# fitted_params is the ten-entry coefficient vector fitted on training data.
# acceleration = equation(x, v, fitted_params)
```
