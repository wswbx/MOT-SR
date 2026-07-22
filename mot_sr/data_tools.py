# data_tools.py
"""Light‑weight data‑analysis helper functions for LLM‑SR supervisor experiments.

 **Interface contract**
Each callable must accept **`x, y`** (1‑D array‑like or list‑like) and return a Python
`dict` that can be JSON‑serialised.  Keys are up to the function – but every dict should
at least contain a top‑level key `"metric"` for quick numerical comparisons.

Feel free to extend or replace any of these functions with more sophisticated analysis
(FFT, PCA, partial‑dependence, SHAP, …) later – the supervisor LLM only needs the name.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
from scipy import stats

import pywt
import nolds
from sklearn.decomposition import PCA

# __all__ = [
#     "corr",          # Pearson correlation
#     "spearman_corr", # Spearman rank correlation
#     "lin_reg",       # Simple linear regression summary
#     "residual_var",  # Variance of residuals from linear fit
# ]

__all__ = [
    "corr", 
    "pearson_corr",                 
    "spearman_corr",                
    "lin_reg",                      
    "residual_var",                 
    "mutual_info",                  
    "fft_cross_freq",               
    "wavelet_corr",                 
    "pca_mapping",                  
    "lyapunov_relation",            
    "corr_dim_relation",            
    "ks_test_diff",                 
    "dtw_distance",                 
    "granger_causality",            
    "mutual_info_regression_score", 
    "ccm_causality",                
]

def _as_array(a):
    """Helper to coerce list‑likes to 1‑D `np.ndarray`."""
    return np.asarray(a, dtype=float).ravel()

# ---------------------------------------------------------------------------
#  Tool functions – keep them **side‑effect free** so they can be safely called
#  by the main pipeline.
# ---------------------------------------------------------------------------

def corr(x, y) -> Dict[str, Any]:
    """Return Pearson correlation r and two‑tailed p‑value."""
    x, y = _as_array(x), _as_array(y)
    r, p = stats.pearsonr(x, y)
    return {"tool": "corr", "metric": r, "p_value": p}


def spearman_corr(x, y) -> Dict[str, Any]:
    """Return Spearman rank correlation ρ and p‑value."""
    x, y = _as_array(x), _as_array(y)
    rho, p = stats.spearmanr(x, y)
    return {"tool": "spearman_corr", "metric": rho, "p_value": p}


def lin_reg(x, y) -> Dict[str, Any]:
    """Fit y = a + b·x (ordinary least squares)."""
    x, y = _as_array(x), _as_array(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "tool": "lin_reg",
        "metric": r_value ** 2,  # R² as primary quality metric
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
    }


def residual_var(x, y) -> Dict[str, Any]:
    """Variance of residuals after simple linear regression."""
    x, y = _as_array(x), _as_array(y)
    fit = lin_reg(x, y)
    y_hat = fit["slope"] * x + fit["intercept"]
    resid = y - y_hat
    var = float(np.var(resid, ddof=1))
    out = {"tool": "residual_var", "metric": var}
    out.update(fit)  # include regression stats for convenience
    return out

###########################################################
###########################################################




# pip install nolds pywt


def pearson_corr(x, y):
    x, y = _as_array(x), _as_array(y)
    r, p = stats.pearsonr(x, y)
    return {"tool": "pearson_corr", "metric": r, "p_value": p}


def spearman_corr(x, y):
    x, y = _as_array(x), _as_array(y)
    rho, p = stats.spearmanr(x, y)
    return {"tool": "spearman_corr", "metric": rho, "p_value": p}


def lin_reg(x, y):
    x, y = _as_array(x), _as_array(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "tool": "lin_reg",
        "metric": r_value ** 2,  
        "details": {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        }
    }


def residual_var(x, y):
    x, y = _as_array(x), _as_array(y)
    fit = lin_reg(x, y)
    y_hat = fit["details"]["slope"] * x + fit["details"]["intercept"]
    resid = y - y_hat
    var = float(np.var(resid, ddof=1))
    return {
        "tool": "residual_var",
        "metric": var,
        "details": fit["details"]
    }


def mutual_info(x, y, bins=10):
    x, y = _as_array(x), _as_array(y)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = stats.entropy(c_xy.flatten())  
    return {
        "tool": "mutual_info",
        "metric": float(mi),
        "details": {"bins": bins}
    }


def fft_cross_freq(x, y, fs=1.0):
    x, y = _as_array(x), _as_array(y)
    fx = np.fft.fftfreq(len(x), d=1/fs)
    fy = np.fft.fftfreq(len(y), d=1/fs)
    px = np.abs(np.fft.fft(x)) ** 2
    py = np.abs(np.fft.fft(y)) ** 2
    fx_dom = fx[np.argmax(px[1:]) + 1]
    fy_dom = fy[np.argmax(py[1:]) + 1]
    return {
        "tool": "fft_cross_freq",
        "metric": abs(fx_dom - fy_dom),
        "details": {
            "x_dominant_freq": fx_dom,
            "y_dominant_freq": fy_dom
        }
    }


def wavelet_corr(x, y, wavelet='db4', level=3):
    x, y = _as_array(x), _as_array(y)
    cx = pywt.wavedec(x, wavelet, level=level)
    cy = pywt.wavedec(y, wavelet, level=level)
    energy_x = np.array([np.sum(c ** 2) for c in cx])
    energy_y = np.array([np.sum(c ** 2) for c in cy])
    corr = np.corrcoef(energy_x, energy_y)[0, 1]
    return {
        "tool": "wavelet_corr",
        "metric": float(corr),
        "details": {
            "energy_x": energy_x.tolist(),
            "energy_y": energy_y.tolist()
        }
    }


def pca_mapping(x, y, n_components=1):
    x, y = _as_array(x), _as_array(y)
    data = np.stack([x, y], axis=1)
    pca = PCA(n_components=n_components).fit(data)
    return {
        "tool": "pca_mapping",
        "metric": pca.explained_variance_ratio_[0],
        "details": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "components": pca.components_.tolist()
        }
    }


def lyapunov_relation(x, y):
    x, y = _as_array(x), _as_array(y)
    
    data_len = min(len(x), len(y))
    emb_dim = max(2, min(10, data_len // 10))  
    lag = max(1, min(10, data_len // 20))  

    lce_x = nolds.lyap_r(x, emb_dim=emb_dim, lag=lag)
    lce_y = nolds.lyap_r(y, emb_dim=emb_dim, lag=lag)
    diff = abs(lce_x - lce_y)
    return {
        "tool": "lyapunov_relation",
        "metric": diff,
        "details": {
            "lyap_x": float(lce_x),
            "lyap_y": float(lce_y),
            "emb_dim": emb_dim,
            "lag": lag
        }
    }


def corr_dim_relation(x, y, emb_dim=2):
    x, y = _as_array(x), _as_array(y)
    d_x = nolds.corr_dim(x, emb_dim=emb_dim)
    d_y = nolds.corr_dim(y, emb_dim=emb_dim)
    return {
        "tool": "corr_dim_relation",
        "metric": abs(d_x - d_y),
        "details": {
            "corr_dim_x": d_x,
            "corr_dim_y": d_y,
            "embedding_dim": emb_dim
        }
    }

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tslearn.metrics import dtw
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression


def ks_test_diff(x, y):
    x, y = _as_array(x), _as_array(y)
    stat, p = ks_2samp(x, y)
    return {
        "tool": "ks_test_diff",
        "metric": stat,
        "p_value": p
    }


def dtw_distance(x, y):
    x, y = _as_array(x), _as_array(y)
    dist = dtw(x, y)
    return {
        "tool": "dtw_distance",
        "metric": dist
    }


def granger_causality(x, y, maxlag=3):
    x, y = _as_array(x), _as_array(y)
    df = pd.DataFrame({'y': y, 'x': x})
    result = grangercausalitytests(df[['y', 'x']], maxlag=maxlag)
    pvals = {f"lag_{lag}": result[lag][0]['ssr_ftest'][1] for lag in result}
    return {
        "tool": "granger_causality",
        "metric": min(pvals.values()),
        "details": pvals
    }


def mutual_info_regression_score(x, y):
    x, y = _as_array(x), _as_array(y)
    from sklearn.metrics import mutual_info_score
    mi = mutual_info_score(x, y)

    # mi = mutual_info_regression(np.array(x).reshape(-1, 1), y)
    return {
        "tool": "mutual_info_regression_score",
        "metric": float(mi)
        # "metric": float(mi[0])

    }


def ccm_causality(x, y, E=2, start=5, step_ratio=0.1, max_ratio=0.8):
    x, y = _as_array(x), _as_array(y)
    from pyEDM import CCM  
    df = pd.DataFrame({'x': x, 'y': y})

    
    data_len = min(len(x), len(y))
    
    
    step = max(1, int(data_len * step_ratio))  
    max_size = max(start, int(data_len * max_ratio))  
    
    
    lib_sizes = range(start, max_size + 1, step)

    
    libSizes = " ".join(map(str, lib_sizes))

    
    # libSizes = generate_dynamic_libSizes(x, y, start=5, step_ratio=0.1, max_ratio=0.8)
    
    
    result = CCM(dataFrame=df, E=E, columns='x', target='y', libSizes=libSizes, sample=1, showPlot=False)

    # result = CCM(dataFrame=df, E=E, columns='x', target='y', libSizes="10 50 10", sample=1, showPlot=False)
    
    # score = result

    # score = result['rho'].values[-1]
    return {
        "tool": "ccm_causality",
        # "metric": score,
        # "metric": float(score),
        "details": result.to_dict()
    }






###########################################################
###########################################################







# ---------------------------------------------------------------------------
#  Optional registry for dynamic look‑up in the pipeline
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {name: globals()[name] for name in __all__}

if __name__ == "__main__":
    # Quick sanity test – will be removed or replaced by unit tests later
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + rng.normal(scale=0.5, size=100)
    for fn_name in __all__:
        fn = TOOL_REGISTRY[fn_name]
        fn(x, y)["metric"]
