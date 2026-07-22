# scientist_llm.py
"""Supervisor‑LLM wrapper – meta‑prompt + dynamic prompt + tool & variable selection.

The *scientist* LLM receives two prompts per iteration:
1. Static meta‑prompt – explains the standing task.
2. Dynamic prompt     – current best equations, residual stats, whitelist of tools/variables.

It must reply with STRICT JSON:
```
{
  "next_prompt":  "...",          # guidance for generator LLM
  "analysis_tool": "tool_name",   # from allowed_tools
  "variables":     ["v1", "v2"]   # exactly two, from allowed_variables
}
```
"""
from __future__ import annotations
import re, json
import json
import os
import http.client
from typing import Sequence, Dict, Any,List

import requests
_ALLOWED_TOOLS: set[str] = {
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
}

class ScientistLLM:
    """Wrapper around an external LLM that acts as the scientific supervisor."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_meta_prompt() -> str:
        allowed_list = ", ".join(sorted(_ALLOWED_TOOLS))

        return (
            # -------------------------------------------------------------
            
            # -------------------------------------------------------------
            "SYSTEM ROLE: You are a **scientist‑assistant LLM** supervising a "
            "symbolic‑regression pipeline.\n\n"
            "You must analyze existing equations, select appropriate analysis "
            "tools, and produce guidance for a generator LLM.\n\n"

            # -------------------------------------------------------------
            
            # -------------------------------------------------------------
            "## Allowed analysis tools\n"
            f"{allowed_list}\n\n"
            "Choose **up to 3** tools—exclusively from the list above—based **solely on the diagnostic needs revealed by the residual data you receive**."
            "If a desired tool is not listed, omit it.\n\n"

            # -------------------------------------------------------------
            
            # -------------------------------------------------------------
            "## RESPONSE FORMAT (STRICT)\n"
            "Return exactly one JSON object (no markdown fences). It must contain:\n"
            "1. \"structure_insight_prompt_for_generator_LLM\" – either a string, or\n"
            "   an object with `guidance.prompt_for_generator_LLM` (string).\n"
            "2. \"analysis_tools\" – a JSON array of 1–5 tool names from the allowed list.\n\n"
            "Escape newlines inside strings as `\\n`. After the closing brace `}`, "
            "output nothing else.\n"
        )

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        allowed_tools: Sequence[str],
        allowed_variables: Sequence[str],
        meta_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        if not allowed_tools:
            raise ValueError("allowed_tools must be a non‑empty sequence")
        if not allowed_variables:
            raise ValueError("allowed_variables must be a non‑empty sequence")

        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key or os.getenv("API_KEY")
        self.allowed_tools = list(allowed_tools)
        self.allowed_variables = list(allowed_variables)
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.meta_prompt = meta_prompt or self._default_meta_prompt()



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_equations(
        self,
        best_eqs: Sequence[str],
        scores: Sequence[float],
        residuals: Sequence[Sequence[float]],
        context: str = "",
    ) -> Dict[str, Any]:
        prompt = f"{self.meta_prompt}\n\n" + self._build_dynamic_prompt(best_eqs, scores, residuals, context)
        raw = self._query_llm(prompt)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _build_dynamic_prompt(
        self,
        eqs: Sequence[str],
        scores: Sequence[Any],  
        residuals: Sequence[Any],  
        context: str,
    ) -> str:
        blocks: list[str] = []
        # print("=============================residuals看这里===============================")
        # print(residuals)
        for i, (eq, sc, res) in enumerate(zip(eqs, scores, residuals)):

            
            if isinstance(sc, (list, tuple)) and len(sc) >= 3:
                
                id_score = sc[0]
                ood_score = sc[1]
                eval_time = sc[2]
                
                
                accuracy_avg = (id_score + ood_score) / 2
                generalization_gap = abs(id_score - ood_score)
                efficiency_score = 1.0 / (1.0 + eval_time)
                
                score_block = (
                    f"ID Score     : {id_score:.6g} (in-distribution performance)\n"
                    f"OOD Score    : {ood_score:.6g} (out-of-distribution performance)\n"
                    f"Eval Time    : {eval_time:.6g}s (computational efficiency)\n"
                    f"Avg Accuracy : {accuracy_avg:.6g} (overall accuracy)\n"
                    f"Gen Gap      : {generalization_gap:.6g} (generalization stability)\n"
                    f"Efficiency   : {efficiency_score:.6g} (time-normalized performance)"
                )
            elif isinstance(sc, (list, tuple)):
                
                score_names = ["Score_1", "Score_2", "Score_3", "Score_4", "Score_5"]
                score_lines = []
                for j, s in enumerate(sc):
                    score_name = score_names[j] if j < len(score_names) else f"Score_{j+1}"
                    score_lines.append(f"{score_name:<12}: {s:.6g}")
                score_block = "\n".join(score_lines)
            else:
                
                score_block = f"Score        : {sc:.6g}"
            

            residual_analysis_blocks = []
            dataset_names = ["In-Distribution (ID)", "Out-of-Distribution (OOD)"]
            
            if isinstance(res, list) and len(res) >= 2:
                for dataset_idx, (dataset_name, residual_data) in enumerate(zip(dataset_names, res[:2])):
                    # print(residual_data)
                    
                    if isinstance(residual_data, dict):
                        worst_region_analysis = residual_data
                        
                        if worst_region_analysis:
                            
                            worst_region_info = worst_region_analysis.get("worst_region_info", {})
                            ranges = worst_region_analysis.get("ranges", [])
                            target_info = worst_region_analysis.get("target_info", {})
                            
                            
                            dataset_analysis = f"\n**{dataset_name} Dataset Analysis:**\n"
                            
                            
                            if worst_region_info:
                                dataset_analysis += (
                                    f"  Worst Region Statistics:\n"
                                    f"    - Sample Count: {worst_region_info.get('sample_count', 0)} samples\n"
                                    f"    - Threshold: {worst_region_info.get('threshold', 0.0):.6g}\n"
                                    f"    - Mean Residual: {worst_region_info.get('mean_residual', 0.0):.6g}\n"
                                    f"    - Mean Abs Residual: {worst_region_info.get('mean_abs_residual', 0.0):.6g}\n"
                                    f"    - Max Abs Residual: {worst_region_info.get('max_abs_residual', 0.0):.6g}\n"
                                    f"    - Std Residual: {worst_region_info.get('std_residual', 0.0):.6g}\n"
                                )
                            
                            
                            if ranges:
                                dataset_analysis += "  Input Variables in Worst Region:\n"
                                for var_range in ranges:
                                    var_name = var_range.get("variable_name", "unknown")
                                    worst_region = var_range.get("worst_region", {})
                                    dataset_analysis += (
                                        f"    {var_name}: [{worst_region.get('min', 0.0):.6g}, "
                                        f"{worst_region.get('max', 0.0):.6g}] "
                                    )
                            
                            
                            if target_info and "worst_region" in target_info:
                                target_worst = target_info["worst_region"]
                                dataset_analysis += (
                                    f"  Target Variable (y) in Worst Region:\n"
                                    f"    y: [{target_worst.get('min', 0.0):.6g}, "
                                    f"{target_worst.get('max', 0.0):.6g}] "
                                )
                            
                            
                            data_info = residual_data.get("data_info", {})
                            if data_info:
                                dataset_analysis += (
                                    f"  Dataset Overview:\n"
                                    f"    - Total Samples: {data_info.get('n_samples', 0)}\n"
                                    f"    - Features: {data_info.get('n_features', 0)}\n"
                                )
                                
                                residual_stats = data_info.get("residual_stats", {})
                                if residual_stats:
                                    dataset_analysis += (
                                        f"    - Overall Residual Stats:\n"
                                        f"      Mean: {residual_stats.get('mean', 0.0):.6g}, "
                                        f"Range: [{residual_stats.get('min', 0.0):.6g}, "
                                        f"{residual_stats.get('max', 0.0):.6g}]\n"
                                    )
                            
                            residual_analysis_blocks.append(dataset_analysis)
                        else:
                            residual_analysis_blocks.append(f"\n**{dataset_name} Dataset:** No worst region analysis available\n")
                    else:
                        residual_analysis_blocks.append(f"\n**{dataset_name} Dataset:** Invalid residual data format\n")
            else:
                residual_analysis_blocks.append("\n**Residual Analysis:** No valid residual data available\n")
            
            
            combined_residual_analysis = "".join(residual_analysis_blocks)
            
            
            blocks.append(
                f"### Equation {i+1}\n"
                f"Expression   : {eq}\n"
                f"{score_block}\n"
                f"{combined_residual_analysis}"
            )
        
        tool_list = ", ".join(self.allowed_tools)
        var_list = ", ".join(self.allowed_variables)
        
        return (
            f"## Context\n{context}\n\n"
            f"## Mathematical Pattern Analysis Task\n"
            f"Analyze the following equations to identify **common mathematical patterns and shared structural elements**:\n\n"
            + "\n".join(blocks) + "\n"
            f"## Available Analysis Tools\n{tool_list}\n"
            f"## Pattern Analysis Instructions\n"
            f"1. **Examine Mathematical Structures**: Look across all equations for:\n"
            f"   - Recurring mathematical functions (trigonometric, exponential, polynomial)\n"
            f"   - Similar variable interaction patterns (x*y, x+y, x^n, etc.)\n"
            f"   - Consistent coefficient magnitudes or sign patterns\n"
            f"   - Common denominators or fraction structures\n"
            f"   - Similar nesting or grouping of terms\n\n"
            f"2. **Identify Shared Elements**: Find mathematical components that appear in multiple high-performing equations\n\n"
            f"3. **Interpret Mathematical Significance**: Explain what these patterns might indicate about:\n"
            f"   - Underlying physical relationships\n"
            f"   - Mathematical constraints or principles\n"
            f"   - Optimal solution characteristics\n\n"
            f"4. **Generate Guidance**: Create a detailed prompt that will guide future equation generation based on successful patterns\n\n"
            f"5. **Select Analysis Tools**: Choose up to 5 data analysis tools that could help validate the patterns you identified\n\n"
            f"## RESPONSE FORMAT\n"
            f"Return ONLY JSON with keys: structure_insight_prompt_for_generator_LLM, analysis_tools.\n"
            f"- structure_insight_prompt_for_generator_LLM: Detailed guidance based on your pattern analysis for generating better equations\n"
            f"- analysis_tools: Array of up to 5 tool names from the available list\n\n"
            f"Focus on mathematical commonalities and their significance in equation generation."
        )
    

    @staticmethod
    def _worst_region_legacy(
        residuals: Sequence[float],
        *,
        top_frac: float = 0.1,
    ) -> Dict[str, Any]:
        if not residuals:
            return {"ranges": [], "mean": 0.0}
        n_top = max(1, int(len(residuals) * top_frac))
        abs_res = [abs(r) for r in residuals]
        threshold = sorted(abs_res, reverse=True)[n_top - 1]
        worst = [r for r in abs_res if r >= threshold]
        mean_abs = sum(worst) / len(worst)
        return {"ranges": [], "mean": mean_abs}

    @staticmethod
    def _worst_region(
        residual_data: Dict[str, Any],
        *,
        top_frac: float = 0.1,
    ) -> Dict[str, Any]:
        if not residual_data or not isinstance(residual_data, dict):
            return {"ranges": [], "mean": 0.0}
        
        
        worst_region_analysis = residual_data.get("worst_region_analysis", {})
        
        if not worst_region_analysis:
            return {"ranges": [], "mean": 0.0}
        
        
        mean_abs_residual = worst_region_analysis.get("mean_abs_residual", 0.0)
        
        
        ranges = []
        variable_ranges = worst_region_analysis.get("ranges", [])
        
        for var_info in variable_ranges:
            if "worst_region" in var_info:
                worst_region = var_info["worst_region"]
                
                ranges.append((worst_region["min"], worst_region["max"]))
        
        
        if not ranges:
            ranges = [(0.0, 1.0)]  
        
        return {
            "ranges": ranges,
            "mean": mean_abs_residual
        }

    # ------------------------------------------------------------------
    # LLM transport helpers
    # ------------------------------------------------------------------
    def _query_llm(self, prompt: str) -> str:
        if self.endpoint:
            payload = {
                "prompt": prompt, 
                "repeat_prompt": 1,
                "params": {
                    "do_sample": True,
                    "temperature": self.temperature, 
                    "max_tokens": self.max_tokens,
                    "add_special_tokens": False,
                    "skip_special_tokens": True,
                }
            }
            resp = requests.post(self.endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            
            response_content = resp.json()["content"]
            
            if isinstance(response_content, list):
                return response_content[0]
            return response_content

        
        conn = http.client.HTTPSConnection("api.openai.com")
        body = json.dumps(
            {
                "model": self.model_name or "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        conn.request("POST", "/v1/chat/completions", body, headers)
        data = json.loads(conn.getresponse().read().decode())
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------


    def _extract_json_from_response(self, text: str) -> str:
        
        json_str = self._extract_json_from_code_block(text)
        if json_str:
            return json_str
        
        
        json_str = self._extract_raw_json(text)
        if json_str:
            return json_str
        
        
        return self._extract_flexible_json(text)

    def _extract_json_from_code_block(self, text: str) -> str:
        import re
        
        patterns = [
            r'```json\s*\n?(.*?)\n?```',
            r'```\s*\n?(.*?)\n?```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                json_content = match.group(1).strip()
                if self._is_likely_json(json_content):
                    return json_content
        
        return ""

    def _extract_raw_json(self, text: str) -> str:
        depth = 0
        start = None
        last_json = ""          

        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i    
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    
                    last_json = text[start : i + 1]
                    start = None

        return last_json


    def _is_likely_json(self, text: str) -> bool:
        text = text.strip()
        if not text.startswith('{') or not text.endswith('}'):
            return False
        
        
        required_fields = ['structure_insight_prompt_for_generator_LLM', 'analysis_tools']
        return all(field in text for field in required_fields)

    def _extract_flexible_json(self, text: str) -> str:
        import re
        
        lines = text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            
            if any(key in line.lower() for key in ['structure_insight_prompt_for_generator_llm', 'analysis_tools']):
                in_json = True
            
            if in_json:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and '}' in line:
                    break
        
        if json_lines:
            potential_json = ' '.join(json_lines)
            potential_json = self._fix_common_json_issues(potential_json)
            if self._is_likely_json(potential_json):
                return potential_json
        
        return ""


    def _validate_json_structure(self, data: Dict[str, Any]) -> None:
        
        if not isinstance(data, dict):
            raise ValueError("Root JSON must be an object")

        required_keys = {"structure_insight_prompt_for_generator_LLM", "analysis_tools"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"Missing required key(s): {', '.join(sorted(missing))}")

        # ---------- structure_insight ----------
        insight = data["structure_insight_prompt_for_generator_LLM"]

        if isinstance(insight, str):
            if not insight.strip():
                raise ValueError("structure_insight_prompt_for_generator_LLM string is empty")

        elif isinstance(insight, dict):
            guidance_obj = insight.get("guidance") or insight.get("generate_guidance")
            if not (isinstance(guidance_obj, dict)
                    and isinstance(guidance_obj.get("prompt_for_generator_LLM"), str)
                    and guidance_obj["prompt_for_generator_LLM"].strip()):
                raise ValueError(
                    "structure_insight_prompt_for_generator_LLM must contain "
                    "'guidance.prompt_for_generator_LLM' "
                    "(or legacy 'generate_guidance.prompt_for_generator_LLM')"
                )
        else:
            raise ValueError(
                "structure_insight_prompt_for_generator_LLM must be a string "
                "or an object with guidance.prompt_for_generator_LLM"
            )

        # ---------- analysis_tools ----------
        tools = data["analysis_tools"]
        if not isinstance(tools, list):
            raise ValueError("analysis_tools must be a list")
        # if not 1 <= len(tools) <= 3:
        #     raise ValueError("analysis_tools must contain between 1 and 3 items")

        
        filtered: list[str] = []
        for t in tools:
            if t in _ALLOWED_TOOLS:
                filtered.append(t)
            else:
                print(f"[ScientistLLM] Warning: skipping unknown tool: {t}")

        if not filtered:
            raise ValueError(
                "All requested analysis_tools were unknown. "
                f"Allowed tools = {sorted(_ALLOWED_TOOLS)}"
            )

        
        if len(filtered) > 3:
            import random
            filtered = random.sample(filtered, 3)
        
        data["analysis_tools"] = filtered


    def _extract_structure_insight_text(self, data: Dict[str, Any]) -> str:
        structure_insight = data["structure_insight_prompt_for_generator_LLM"]
        
        if isinstance(structure_insight, str):
            
            return structure_insight
        elif isinstance(structure_insight, dict):
            
            guidance = structure_insight.get("guidance") or structure_insight.get("generate_guidance", {})

            prompt_text = guidance.get("prompt_for_generator_LLM", "")
            
            
            enhanced_text = self._create_enhanced_structure_text(structure_insight, prompt_text)
            return enhanced_text
        else:
            return ""

    def _create_enhanced_structure_text(self, structure_data: Dict[str, Any], base_prompt: str) -> str:
        parts = []
        
        
        if base_prompt:
            parts.append("## Core Generation Guidance")
            parts.append(base_prompt)
            parts.append("")
        
        
        math_structures = structure_data.get("mathematical_structures", {})
        if math_structures:
            parts.append("## Mathematical Structure Analysis")
            
            recurring_functions = math_structures.get("recurring_functions", [])
            if recurring_functions:
                parts.append(f"**Recurring Functions**: {', '.join(recurring_functions)}")
            
            patterns = math_structures.get("variable_interaction_patterns", [])
            if patterns:
                parts.append(f"**Variable Patterns**: {', '.join(patterns)}")
            
            coefficients = math_structures.get("coefficient_relationships", [])
            if coefficients:
                parts.append(f"**Coefficient Patterns**: {', '.join(coefficients)}")
            
            parts.append("")
        
        
        shared_elements = structure_data.get("shared_elements", {})
        if shared_elements:
            parts.append("## Shared Mathematical Elements")
            
            common_terms = shared_elements.get("common_terms", [])
            if common_terms:
                parts.append(f"**Common Terms**: {', '.join(common_terms)}")
            
            nested_structures = shared_elements.get("nested_structures", [])
            if nested_structures:
                parts.append(f"**Nested Structures**: {', '.join(nested_structures)}")
            
            fraction_structures = shared_elements.get("fraction_structures", [])
            if fraction_structures:
                parts.append(f"**Fraction Patterns**: {', '.join(fraction_structures)}")
            
            parts.append("")
        
        
        interpretation = structure_data.get("interpretation", {})
        if interpretation:
            parts.append("## Physical and Mathematical Interpretation")
            
            physical_rel = interpretation.get("physical_relationships", [])
            if physical_rel:
                parts.append(f"**Physical Context**: {', '.join(physical_rel)}")
            
            constraints = interpretation.get("mathematical_constraints", [])
            if constraints:
                parts.append(f"**Mathematical Constraints**: {', '.join(constraints)}")
            
            characteristics = interpretation.get("optimal_solution_characteristics", [])
            if characteristics:
                parts.append(f"**Solution Characteristics**: {', '.join(characteristics)}")
            
            parts.append("")
        
        
        parts.append("## Generation Instructions")
        parts.append("Based on the above analysis, focus on:")
        parts.append("1. Incorporating the identified recurring mathematical functions")
        parts.append("2. Using the successful variable interaction patterns")
        parts.append("3. Maintaining coefficient relationships that work well")
        parts.append("4. Respecting the physical constraints and relationships")
        parts.append("5. Building upon the shared mathematical elements")
        
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        import json, re

        raw = raw.strip()

        
        json_str = self._extract_json_from_response(raw)
        if not json_str:
            raise ValueError(f"No valid JSON found in response: {raw}")

        
        def _sanitize_unescaped_newlines(s: str) -> str:
            """Replace bare newlines *inside quoted strings* with \\n."""
            def _fix(match: re.Match[str]) -> str:
                return '"' + match.group(1).replace('\n', r'\n') + '"'
            return re.sub(r'"((?:[^"\\]|\\.)*)"', _fix, s, flags=re.DOTALL)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:         
            json_str = _sanitize_unescaped_newlines(json_str)
            data = json.loads(json_str)

        
        self._validate_json_structure(data)

        
        structure_text = self._extract_structure_insight_text(data)

        
        result = {
            "structure_insight_prompt_for_generator_LLM": structure_text,
            "analysis_tools": data["analysis_tools"],
            "raw_structure_data": data["structure_insight_prompt_for_generator_LLM"],
        }

        print(
            f"[Scientist] Parsed pattern analysis: "
            f"{len(structure_text)} chars, "
            f"{len(result['analysis_tools'])} tools"
        )
        return result
