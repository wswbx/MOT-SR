"""Lightweight helper that executes the analysis tool specified by *ScientistLLM*.

Requirements
------------
Caller prepares **two** Python objects in memory and passes them to
`run_supervisor_tool`:

* ``supervisor_output`` – dict from ``ScientistLLM.analyze``
  ``{"analysis_tools": ["corr", "scatter"]}``
* ``dataset`` – pre‑split dict ::

      dataset = {
         "id":  {..."inputs": X_id,  "outputs": y_id},
         "ood": {..."inputs": X_ood, "outputs": y_ood},
      }

The function concatenates all ID/OOD data, analyzes each independent variable 
against the dependent variable using the specified tools, and returns the results.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from mot_sr import data_tools  # local module with analysis functions

__all__ = ["run_supervisor_tool"]


def _concat_inputs(dataset: Dict, col: int) -> np.ndarray:
    """Vertically stack the ID + OOD inputs for column *col*."""
    return np.concatenate([
        dataset["id"]["inputs"][:, col],
        dataset["ood"]["inputs"][:, col],
    ])


def _concat_outputs(dataset: Dict) -> np.ndarray:
    """Vertically stack the ID + OOD outputs (dependent variable)."""
    return np.concatenate([
        dataset["id"]["outputs"],
        dataset["ood"]["outputs"],
    ])


def run_supervisor_tool(
    supervisor_output: Dict[str, object],
    dataset: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, object]:
    """Execute the Scientist‑chosen tools on all independent variables vs dependent variable.

    Args:
        supervisor_output: Dict containing "analysis_tools" list
        dataset: Dict with "id" and "ood" subdicts containing "inputs" and "outputs"

    Returns:
        Dict with analysis results for each variable-tool combination
    """
    tool_names: List[str] = supervisor_output["analysis_tools"]
    
    
    sample_inputs = dataset["id"]["inputs"]
    n_features = sample_inputs.shape[1]  
    
    
    y_combined = _concat_outputs(dataset)
    
    
    results = {}
    
    
    for var_idx in range(n_features):
        
        x_combined = _concat_inputs(dataset, var_idx)
        
        var_name = f"x{var_idx}"  
        
        for tool_name in tool_names:
            
            tool_fn = getattr(data_tools, tool_name, None)
            if tool_fn is None:
                print(f"[DataAnalysis] Warning: tool '{tool_name}' was not found.")
                continue
            
            try:
                
                analysis_result = tool_fn(x_combined, y_combined)
                
                
                result_key = f"{var_name}_{tool_name}"
                results[result_key] = analysis_result
                
            except Exception as e:
                print(f"[DataAnalysis] Error running {tool_name} on {var_name}: {e}")
                
                results[f"{var_name}_{tool_name}"] = {
                    "error": str(e),
                    "tool": tool_name,
                    "variable": var_name
                }
    
    
    results["_metadata"] = {
        "n_variables": n_features,
        "tools_used": tool_names,
        "total_samples": len(y_combined),
        "id_samples": len(dataset["id"]["outputs"]),
        "ood_samples": len(dataset["ood"]["outputs"])
    }
    
    return results
