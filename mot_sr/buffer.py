from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy
import ast
import math
from mot_sr import code_manipulation
from mot_sr import config as config_lib
from mot_sr import symbolic_complexity

# mmmmmm

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """ A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # # Initialize empty islands.
        # self._islands: list[Island] = []
        # for _ in range(config.num_islands):
        #     self._islands.append(
        #         Island(template, function_to_evolve, config.functions_per_prompt,
        #                config.cluster_sampling_temperature_init,
        #                config.cluster_sampling_temperature_period))
        # self._best_score_per_island: list[float] = (
        #         [-float('inf')] * config.num_islands)
        # self._best_program_per_island: list[code_manipulation.Function | None] = (
        #         [None] * config.num_islands)
        # self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
        #         [None] * config.num_islands)

        # self._last_reset_time: float = time.time()


        
        # Initialize population
        self._population: list[code_manipulation.Function] = []
        self._pop_size = config.num_islands  # Use `num_islands` as the population size
        self._generation = 0

    def get_pareto_front_programs(self) -> list[code_manipulation.Function]:
        """Returns a shallow copy of the current Pareto-front population."""
        return list(self._population)



    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)
    



    def get_prompt_from_function(self, function: code_manipulation.Function) -> Prompt:
        """Constructs a prompt from a given function, including all previous versions."""


        all_functions = copy.deepcopy(function)  

        all_functions = [function]  
        all_functions_str = [str(f) for f in all_functions]  

        
        next_version = len(all_functions)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            all_functions[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                    f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        all_functions.append(header)

        
        prompt = dataclasses.replace(self._template, functions=all_functions)

        
        prompt_code = str(prompt)
        version_generated = next_version

        return Prompt(prompt_code, version_generated, None)
    


    # def _merge_and_update_population(self, offspring: list[code_manipulation.Function]) -> None:
        """Merge parent and offspring populations, and update the population."""

        # combined_population = self._population + offspring
        
        combined_population = self._population

        
        scores = [program.score for program in combined_population]
        

        scores_array = np.array([[-s[0], -s[1], s[2], s[3], s[4]] for s in scores])

        
        scores_array[:, 2] = 0  
        scores_array[:, 3] = 0  
        


        #############################################################################
        
        
        min_values = np.min(scores_array[:, :2], axis=1)

        
        log_min_values = np.array([math.floor(math.log10(abs(val))) if val > 0 else -9999 for val in min_values])

        log_min_threshold = np.min(log_min_values)

        
        for i, row in enumerate(scores_array):
            min_value = min(row[0], row[1]) 
            
            if min_value > 10 ** ( math.floor(log_min_threshold*0.5)):
                
                scores_array[i, 0] = 999999
                scores_array[i, 1] = 999999
                
                scores_array[i, 4] = 999999

        
        ###########################################################################



        
        nondom_idx = NonDominatedSorting().do(scores_array, only_non_dominated_front=True)

        self._population = [combined_population[i] for i in nondom_idx[:self._pop_size]]
        self._generation += 1

        
        updated_scores = [program.score for program in self._population]
        updated_scores_array = np.array([[-s[0], -s[1], s[2], s[3], s[4]] for s in updated_scores])



        

        
        logging.info('Generation %d: Pareto front scores: %s', self._generation, [scores[i] for i in nondom_idx])

    def _merge_and_update_population(self, offspring: list[code_manipulation.Function]) -> None:
        """Merge parent and offspring populations, and update the population."""

        # combined_population = self._population + offspring
        
        combined_population = self._population

        
        scores = [program.score for program in combined_population]
        

        scores_array = np.array([[-s[0], -s[1], s[2], s[3], s[4]] for s in scores])

        
        scores_array[:, 2] = 0  
        scores_array[:, 3] = 0  
        


        
        
        # scores_array[:, 3] = [math.floor(math.log10(abs(s[3]))) if s[3] != 0 else 0 for s in scores]

        
        #############################################################################
        
        
        min_values = np.min(scores_array[:, :2], axis=1)

        
        log_min_values = np.array([math.floor(math.log10(abs(val))) if val > 0 else -9999 for val in min_values])

        log_min_threshold = np.min(log_min_values)

        
        for i, row in enumerate(scores_array):
            min_value = min(row[0], row[1])  
            
            if min_value > 10 ** ( math.floor(log_min_threshold*0.5)):
                
                scores_array[i, 0] = 999999
                scores_array[i, 1] = 999999
                
                scores_array[i, 4] = 999999

        
        ###########################################################################

        
        
        pareto_columns = scores_array[:, [0, 1, 4]]
        # pareto_columns01 = pareto_columns

        #￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
        
        unique_indices = []
        seen = {}

        for i, row in enumerate(pareto_columns):
            key = (float(f"{row[0]:.3g}"), float(f"{row[1]:.3g}"))
            if key not in seen:
                seen[key] = i
                unique_indices.append(i)
            else:
                
                existing_index = seen[key]
                if pareto_columns[i, 2] < pareto_columns[existing_index, 2]:
                    
                    seen[key] = i
                    unique_indices.remove(existing_index)
                    unique_indices.append(i)
                    
                    scores_array[existing_index, [0, 1, 4]] = 999999
                else:
                    
                    scores_array[i, [0, 1, 4]] = 999999

        
        pareto_columns = scores_array[: , [0, 1, 4]]

        #￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥


        
        nondom_idx = NonDominatedSorting().do(pareto_columns, only_non_dominated_front=True)
        
        # pareto_columns = scores_array[:, [0, 1, 3]]
        
        # nondom_idx = NonDominatedSorting().do(pareto_columns, only_non_dominated_front=True)
        
        # print("Non-dominatedNon-dominatedNon-dominatedNon-dominatedNon-dominated")
        # print("Non-dominated indices based on columns 1, 2, and 5:", nondom_idx)



        
        # nondom_idx = NonDominatedSorting().do(scores_array, only_non_dominated_front=True)

        # print("Non-dominatedNon-dominatedNon-dominatedNon-dominatedNon-dominated")
        # print("Non-dominated indices:", nondom_idx)

        
        self._population = [combined_population[i] for i in nondom_idx[:self._pop_size]]
        self._generation += 1

        
        updated_scores = [program.score for program in self._population]
        updated_scores_array = np.array([[-s[0], -s[1], s[2], s[3], s[4]] for s in updated_scores])

        
        logging.info('Generation %d: Pareto front scores: %s', self._generation, [scores[i] for i in nondom_idx])



    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            **kwargs 
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)


    
    
    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            **kwargs
    ) -> None:
        """Registers new `program` skeleton hypotheses in the experience buffer."""
        
        
        evaluate_time = kwargs.get('evaluate_time', None)
        residual_data = kwargs.get('residual_data', {}) or {}
        equation_time = kwargs.get('equation_time', None)  
        ast_length = symbolic_complexity.simplified_function_ast_length(program)
        
        if "dataset_id" not in scores_per_test:
            scores_per_test["dataset_id"] = -9999999 
            
            evaluate_time = 99999999 
        if "dataset_ood" not in scores_per_test:
            
            scores_per_test["dataset_ood"] = -9999999 
            evaluate_time = 99999999 
        
        program.score = [
            scores_per_test["dataset_id"],  
            scores_per_test["dataset_ood"],  
            evaluate_time,  
            equation_time,  
            ast_length  
        ] 
        
        if "dataset_id" not in residual_data:
            residual_data["dataset_id"] = None 
        if "dataset_ood" not in residual_data:
            residual_data["dataset_ood"] = None 

        program.residual_data = {
            "dataset_id": residual_data["dataset_id"],
            "dataset_ood": residual_data["dataset_ood"],
        }

        self._population.append(program)

        
        if len(self._population) > self._pop_size:
            self._merge_and_update_population([])

        profiler: profile.Profiler = kwargs.get('profiler', None)

        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            profiler.register_function(program)


    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0


    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1


    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated


    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """ Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create header of new function to be completed
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        
        return str(prompt)


class Cluster:
    """ A cluster of programs on the same island and with the same Signature. """

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
