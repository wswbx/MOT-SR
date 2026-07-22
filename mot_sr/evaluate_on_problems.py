import numpy as np
import time
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS
from scipy.optimize import minimize
from typing import Sequence, Dict, Any,List

DECIMAL_PLACES = 3
TAU = 0.1
def _worst_region(
    result_data: np.ndarray,
    *,
    top_frac: float = 0.1,
) -> Dict[str, Any]:
    if result_data is None or len(result_data) == 0:
        return {"ranges": [], "mean_residual": 0.0, "worst_region_info": {}}
    
    
    n_samples, n_cols = result_data.shape
    
    
    residuals = result_data[:, -1]
    
    target_values = result_data[:, -2] if n_cols > 1 else None
    
    feature_values = result_data[:, :-2] if n_cols > 2 else None
    
    abs_residuals = np.abs(residuals)
    
    
    n_worst = max(1, int(n_samples * top_frac))
    
    
    worst_indices = np.argsort(abs_residuals)[-n_worst:]
    worst_threshold = abs_residuals[worst_indices[0]]  
    
    
    worst_residuals = residuals[worst_indices]
    worst_abs_residuals = abs_residuals[worst_indices]
    
    
    worst_region_info = {
        "sample_count": int(n_worst),
        "indices": worst_indices.tolist(),
        "threshold": float(worst_threshold),
        "mean_residual": float(np.mean(worst_residuals)),
        "mean_abs_residual": float(np.mean(worst_abs_residuals)),
        "max_abs_residual": float(np.max(worst_abs_residuals)),
        "min_abs_residual": float(np.min(worst_abs_residuals)),
        "std_residual": float(np.std(worst_residuals))
    }
    
    ranges = []
    
    
    if feature_values is not None:
        n_features = feature_values.shape[1]
        worst_features = feature_values[worst_indices]
        
        for feature_idx in range(n_features):
            worst_feature_vals = worst_features[:, feature_idx]
            
            feature_range = {
                "variable_name": f"x{feature_idx}",
                "feature_index": feature_idx,
                "worst_region": {
                    "min": float(np.min(worst_feature_vals)),
                    "max": float(np.max(worst_feature_vals)),
                    "mean": float(np.mean(worst_feature_vals)),
                    "std": float(np.std(worst_feature_vals))
                }
            }
            ranges.append(feature_range)
            
    target_info = None
    if target_values is not None:
        worst_targets = target_values[worst_indices]
        target_info = {
            "variable_name": "y",
            "worst_region": {
                "min": float(np.min(worst_targets)),
                "max": float(np.max(worst_targets)),
                "mean": float(np.mean(worst_targets)),
                "std": float(np.std(worst_targets))
            }
        }
        
    result = {
        "ranges": ranges,
        "target_info": target_info,
        "mean_residual": worst_region_info['mean_residual'],  
        "mean_abs_residual": worst_region_info['mean_abs_residual'],
        "worst_region_info": worst_region_info,
        "data_shape": list(result_data.shape),
        "analysis_fraction": top_frac
    }
    
    return result



def single_start_bfgs(loss_func, n_params=10, x0=None):
    if x0 is None:
        x0 = np.random.uniform(low=-1, high=1, size=n_params)
    return minimize(
        loss_func,
        x0,
        method='BFGS',
        options={
            'maxiter': 500,
            'gtol': 1e-10,
            'eps': 1e-12,
            'disp': False
        }
    )
def acc_tau(y_true, y_pred, tau=0.1):
    rel_err = np.abs(y_pred - y_true) / np.abs(y_true)
    return int(rel_err.max() <= tau), rel_err.max()

def evaluate(fit_data: dict, eval_data: dict, equation) -> tuple:
    fit_inputs, fit_outputs = fit_data['inputs'], fit_data['outputs']
    
    eval_inputs, eval_outputs = eval_data['inputs'], eval_data['outputs']

    X_fit = fit_inputs
    X_eval = eval_inputs
    
    
    def optimization_loss(params):
        y_pred = equation(*X_fit.T, params)
        return np.mean((y_pred - fit_outputs) ** 2)

    # Single-start BFGS optimization on fit_data
    loss_partial = lambda params: optimization_loss(params)
    result = single_start_bfgs(loss_partial, n_params=MAX_NPARAMS)
    
    
    optimized_params = result.x
    optimization_loss_value = result.fun
    y_pred_fit = equation(*X_fit.T, optimized_params)
    acc, max_err = acc_tau(fit_outputs, y_pred_fit, TAU)
    fit_output_variance = np.var(fit_outputs)
    fit_nmse = optimization_loss_value / fit_output_variance if fit_output_variance != 0 else optimization_loss_value
    print(
        "[Eval] Fit optimization complete: "
        f"mse={optimization_loss_value:.6g}, "
        f"nmse={fit_nmse:.6g}, "
        f"acc_tau={acc}, "
        f"max_rel_err={max_err:.6g}"
    )

    if np.isnan(optimization_loss_value) or np.isinf(optimization_loss_value):
        print("Optimization loss is NaN or Inf")
        return None, None
    else:
        equation_start_time = time.time()
        optimized_predictions = equation(*X_eval.T, optimized_params)
        equation_time = time.time() - equation_start_time
        
        
        mse = np.mean((optimized_predictions - eval_outputs) ** 2)
        
        
        eval_output_variance = np.var(eval_outputs)
        if eval_output_variance == 0:
            print("[Eval] Warning: eval_outputs has zero variance; using MSE as NMSE.")
            nmse = mse
        else:
            nmse = mse / eval_output_variance

        residuals = eval_outputs - optimized_predictions
        
        
        result_data = np.column_stack((X_eval, eval_outputs, residuals))

        worst_region_analysis = _worst_region(result_data, top_frac=0.1)
        
        
        complete_result_data = {
            "optimized_params": optimized_params.tolist(),
            "optimization_loss": float(optimization_loss_value),
            "optimization_nmse": float(fit_nmse),
            "evaluation_mse": float(mse),
            "evaluation_nmse": float(nmse),
            "evaluation_output_variance": float(eval_output_variance),
            "worst_region_analysis": worst_region_analysis,
            "data_info": {
                "optimization_samples": int(fit_inputs.shape[0]),
                "evaluation_samples": int(eval_inputs.shape[0]), 
                "n_features": int(result_data.shape[1] - 2),  
                "residual_stats": {
                    "mean": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                    "min": float(np.min(residuals)),
                    "max": float(np.max(residuals))
                }
            }
        }
        print(
            "[Eval] Evaluation complete: "
            f"mse={mse:.6g}, "
            f"nmse={nmse:.6g}, "
            f"params={optimized_params.tolist()}"
        )
        
        
        return -nmse, complete_result_data, equation_time
