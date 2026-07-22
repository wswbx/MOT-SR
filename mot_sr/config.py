"""Configuration of a LLMSR experiments
."""
from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import Type
import os

from mot_sr import sampler
from mot_sr import evaluator
from mot_sr.scientist_defaults import DEFAULT_SCIENTIST_TOOLS
from typing import Dict, List, Optional, Any

# @dataclasses.dataclass(frozen=True)
# class ExperienceBufferConfig:
#     """Configures Experience Buffer parameters.
    
#     Args:
#         functions_per_prompt (int): Number of previous hypotheses to include in prompts
#         num_islands (int): Number of islands in experience buffer for diversity
#         reset_period (int): Seconds between weakest island resets
#         cluster_sampling_temperature_init (float): Initial cluster softmax sampling temperature
#         cluster_sampling_temperature_period (int): Period for temperature decay
#     """
#     functions_per_prompt: int = 2 
#     num_islands: int = 10 
#     reset_period: int = 4 * 60 * 60
#     cluster_sampling_temperature_init: float = 0.1
#     cluster_sampling_temperature_period: int = 30_000



@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters."""
    functions_per_prompt: int = 1 
    num_islands: int = 100  
    reset_period: int = 0  


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for LLMSR experiments.
   
   Args:
       experience_buffer: Evolution multi-population settings
       num_samplers (int): Number of parallel samplers
       num_evaluators (int): Number of parallel evaluators
       samples_per_prompt (int): Number of hypotheses per prompt
       evaluate_timeout_seconds (int): Hypothesis evaluation timeout
       use_api (bool): API usage flag
   """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1 
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30  
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"
    max_sample_nums: Optional[int] = None
    enable_scientist: bool = False
    scientist_endpoint: Optional[str] = None
    scientist_model_name: Optional[str] = None
    scientist_api_key: Optional[str] = None
    scientist_meta_prompt: Optional[str] = None
    scientist_temperature: float = 0.7
    scientist_max_tokens: int = 512
    scientist_allowed_tools: List[str] = field(default_factory=lambda: list(DEFAULT_SCIENTIST_TOOLS))
    scientist_allowed_variables: List[str] = field(default_factory=lambda: [f"x{i}" for i in range(10)])
    scientist_dataset: Optional[Any] = None
    scientist_var_index_map: Dict[str, int] = field(default_factory=lambda: {f"x{i}": i for i in range(10)})
    scientist_problem_context: str = ""
    scientist_min_islands_for_analysis: int = 3

@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]
