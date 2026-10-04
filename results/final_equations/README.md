# Final equations

This directory archives four benchmark equation structures from MOT-SR
(LLaMA-3.1) and the final EMRI correction supplied by the authors.

## MOT-SR (LLaMA-3.1)

Each benchmark file provides an `equation` function with a ten-entry
fitted coefficient vector supplied through `params`.

| Task | File | Inputs | Fitted coefficients |
| --- | --- | --- | --- |
| Oscillation 1 | [oscillation_1.py](mot_sr_llama31/oscillation_1.py) | `x`, `v` | `params[0]` through `params[9]` |
| Oscillation 2 | [oscillation_2.py](mot_sr_llama31/oscillation_2.py) | `t`, `x`, `v` | `params[0]` through `params[9]` |
| E. coli Growth | [e_coli_growth.py](mot_sr_llama31/e_coli_growth.py) | `b`, `s`, `temp`, `pH` | `params[0]` through `params[9]` |
| Stress-Strain | [stress_strain.py](mot_sr_llama31/stress_strain.py) | `strain`, `temp` | `params[0]` through `params[9]` |

For numerical evaluation, fit the coefficients on the corresponding
training data and pass the resulting vector as `params`.

## EMRI correction

The executable expression is in [emri/equation.py](emri/equation.py), with
the complete supplied parameter vector in [emri/params.json](emri/params.json).

$$
\Delta\dot p(p,e,\eta)=
\left(1-3138.3134817337836\,\frac{\eta p}{1+e^2}\right)
\frac{1+\dfrac{e^2}{1+e^2}}
{1+\eta\exp\!\left(3.318812654787899\,p-11.17336476987488\,e\right)}.
$$

The full ten-entry parameter vector is stored in `params.json`. The
expression uses `params[0]`, `params[1]` and `params[2]`.
`delta_p_dot` accepts broadcast-compatible inputs `p`, `e` and `eta` and
evaluates the expression above.

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
