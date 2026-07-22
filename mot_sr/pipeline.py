from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence

from mot_sr import code_manipulation
from mot_sr import config as config_lib
from mot_sr import evaluator
from mot_sr import buffer
from mot_sr import sampler
from mot_sr import profile
from typing import Dict, List, Optional

def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]


def create_sampler(
    database: buffer.ExperienceBuffer,
    evaluators: Sequence[evaluator.Evaluator],
    config: config_lib.Config,
    max_sample_nums: int | None, ###############################################
    scientist_config: Optional[Dict] = None,  
) -> sampler.Sampler:
    return sampler.Sampler(
        database=database,
        evaluators=evaluators,
        samples_per_prompt=config.samples_per_prompt,
        config=config,
        max_sample_nums=max_sample_nums,
        # max_sample_nums=config.max_sample_nums,
        llm_class=sampler.LocalLLM,
        scientist_config=scientist_config,  
    )

# ...existing code...

def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        scientist_config: Optional[Dict] = None,  
        **kwargs
):
    """ Launch a LLMSR experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
        scientist_config: configuration for scientist LLM integration.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, profiler=profiler)

    
    samplers = []
    for _ in range(config.num_samplers):
        samplers.append(create_sampler(
            database=database,
            evaluators=evaluators,
            max_sample_nums=max_sample_nums,
            config=config,
            scientist_config=scientist_config,  
        ))

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler)
