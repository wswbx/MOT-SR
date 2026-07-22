""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from mot_sr import evaluator
from mot_sr import buffer
from mot_sr import config as config_lib
import requests
import json
import http.client
import os

from mot_sr import scientist_llm
from mot_sr import data_analysis_runner
from typing import Collection, Sequence, Type, Dict, List, Optional, Tuple
from mot_sr.scientist_defaults import DEFAULT_SCIENTIST_TOOLS
Port = '5000'

# ok！ pip install codebleu
from codebleu.syntax_match import calc_syntax_match

class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
            scientist_config: Optional[Dict] = None,  
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

        
        self._scientist = None
        self._dataset = None
        self._var_index_map = None
        self._iteration_count = 0
        self._scientist_enabled = False
        
        if scientist_config:
            self._setup_scientist(scientist_config)

    def _setup_scientist(self, scientist_config: Dict):
        try:
            endpoint = scientist_config.get("endpoint")
            model_name = scientist_config.get("model_name")
            api_key = scientist_config.get("api_key")
            meta_prompt = scientist_config.get("meta_prompt")
            allowed_tools = scientist_config.get(
                "allowed_tools",
                DEFAULT_SCIENTIST_TOOLS,
            )
            allowed_variables = scientist_config.get("allowed_variables", [f"x{i}" for i in range(10)])
            temperature = scientist_config.get("temperature", 0.7)
            max_tokens = scientist_config.get("max_tokens", 512)
            
            self._scientist = scientist_llm.ScientistLLM(
                endpoint=endpoint,
                model_name=model_name,
                api_key=api_key,
                allowed_tools=allowed_tools,
                allowed_variables=allowed_variables,
                
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            
            self._dataset = scientist_config.get("dataset")
            self._var_index_map = scientist_config.get("var_index_map", {f"x{i}": i for i in range(10)})
            self._problem_context = scientist_config.get("problem_context", "")
            self._min_islands_for_analysis = scientist_config.get("min_islands_for_analysis", 3)
            self._scientist_enabled = True
            
            print("[Scientist] Successfully initialized scientist LLM")
            
        except Exception as e:
            print(f"[Scientist] Failed to initialize scientist LLM: {e}")
            self._scientist_enabled = False

    def _program_objectives(self, program) -> Tuple[float, float, float]:
        """Return comparable objective values as (ID NMSE, OOD NMSE, AST length)."""
        score = getattr(program, "score", None)
        if not score or len(score) < 5:
            raise ValueError("Program is missing multi-objective score information.")
        return (-score[0], -score[1], score[4])

    def _select_representative_pareto_programs(self, programs, max_programs: int = 3):
        """Select up to `max_programs` representative programs from the current Pareto front."""
        if not programs:
            return []

        unique_programs = []
        seen_signatures = set()
        for program in programs:
            signature = str(program)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_programs.append(program)

        if len(unique_programs) <= max_programs:
            return unique_programs

        representatives = []
        seen_representatives = set()

        def add_program(program):
            signature = str(program)
            if signature in seen_representatives:
                return
            seen_representatives.add(signature)
            representatives.append(program)

        add_program(min(unique_programs, key=lambda program: self._program_objectives(program)[0]))
        add_program(min(unique_programs, key=lambda program: self._program_objectives(program)[1]))
        add_program(min(unique_programs, key=lambda program: self._program_objectives(program)[2]))

        if len(representatives) < max_programs:
            remaining = sorted(unique_programs, key=self._program_objectives)
            for program in remaining:
                add_program(program)
                if len(representatives) >= max_programs:
                    break

        return representatives[:max_programs]

    def _build_scientist_samples_from_pareto_front(self) -> List[Dict]:
        """Build representative Pareto-front samples for the scientist LLM."""
        pareto_programs = self._database.get_pareto_front_programs()
        representative_programs = self._select_representative_pareto_programs(pareto_programs, max_programs=3)

        samples = []
        for program in representative_programs:
            residual_data = getattr(program, "residual_data", None)
            if not isinstance(residual_data, dict):
                continue
            if residual_data.get("dataset_id") is None and residual_data.get("dataset_ood") is None:
                continue
            samples.append({
                "function": program,
                "sample": str(program),
                "residual_data": {
                    "dataset_id": residual_data.get("dataset_id"),
                    "dataset_ood": residual_data.get("dataset_ood"),
                },
            })

        return samples


    def select_parent(self) -> buffer.code_manipulation.Function:
        """Select a parent from the population based on AST diversity using calc_syntax_match."""
        
        valid_population = [f for f in self._database._population if not np.isinf(np.array(f.score)).any()]

        if not valid_population:
            raise ValueError("No valid individuals in the population to select from.")

        
        crt_pop_size = len(valid_population)
        diversity_matrix = np.zeros((crt_pop_size, crt_pop_size))
        for i in range(crt_pop_size):
            for j in range(i + 1, crt_pop_size):
                
                func_i = f"def {valid_population[i].name}({valid_population[i].args}) -> {valid_population[i].return_type}:\n{valid_population[i].body}"
                func_j = f"def {valid_population[j].name}({valid_population[j].args}) -> {valid_population[j].return_type}:\n{valid_population[j].body}"

                
                # print(f"Comparing:\n{func_i}\nAND\n{func_j}")
                diversity_matrix[i, j] = -calc_syntax_match([func_i], func_j, 'python')
                diversity_matrix[j, i] = -calc_syntax_match([func_j], func_i, 'python')
                

        
        diversity_scores = diversity_matrix.sum(axis=0)

        
        probabilities = np.exp(diversity_scores) / np.exp(diversity_scores).sum()

        
        selected_parent = np.random.choice(valid_population, p=probabilities, replace=False)

        return selected_parent

    def select_parent_probability(self) -> List[buffer.code_manipulation.Function]:
        """Select parents from the population based on AST diversity using calc_syntax_match.
        Returns samples sorted by probability in descending order until cumulative probability > 50%."""
        
        valid_population = [f for f in self._database._population if not np.isinf(np.array(f.score)).any()]

        if not valid_population:
            raise ValueError("No valid individuals in the population to select from.")

        
        crt_pop_size = len(valid_population)
        diversity_matrix = np.zeros((crt_pop_size, crt_pop_size))
        for i in range(crt_pop_size):
            for j in range(i + 1, crt_pop_size):
                
                func_i = f"def {valid_population[i].name}({valid_population[i].args}) -> {valid_population[i].return_type}:\n{valid_population[i].body}"
                func_j = f"def {valid_population[j].name}({valid_population[j].args}) -> {valid_population[j].return_type}:\n{valid_population[j].body}"

                
                diversity_matrix[i, j] = -calc_syntax_match([func_i], func_j, 'python')
                diversity_matrix[j, i] = -calc_syntax_match([func_j], func_i, 'python')

        
        diversity_scores = diversity_matrix.sum(axis=0)

        
        probabilities = np.exp(diversity_scores) / np.exp(diversity_scores).sum()

        
        individual_prob_pairs = list(zip(valid_population, probabilities))
        
        
        individual_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        
        selected_individuals = []
        cumulative_probability = 0.0
        
        for i, (individual, prob) in enumerate(individual_prob_pairs):
            cumulative_probability += prob
            selected_individuals.append(individual)
            if cumulative_probability > 0.5:
                break
        
        return selected_individuals
    
    def select_parent_and_probability(self) -> Tuple[buffer.code_manipulation.Function, List[Tuple[buffer.code_manipulation.Function, float]]]:
        """Select a parent from the population based on AST diversity using calc_syntax_match.
        Returns the selected parent and a list of all individuals with their probabilities (sorted by probability descending).
        """
        
        valid_population = [f for f in self._database._population if not np.isinf(np.array(f.score)).any()]

        if not valid_population:
            raise ValueError("No valid individuals in the population to select from.")

        
        crt_pop_size = len(valid_population)
        diversity_matrix = np.zeros((crt_pop_size, crt_pop_size))
        for i in range(crt_pop_size):
            for j in range(i + 1, crt_pop_size):
                
                func_i = f"def {valid_population[i].name}({valid_population[i].args}) -> {valid_population[i].return_type}:\n{valid_population[i].body}"
                func_j = f"def {valid_population[j].name}({valid_population[j].args}) -> {valid_population[j].return_type}:\n{valid_population[j].body}"

                
                diversity_matrix[i, j] = -calc_syntax_match([func_i], func_j, 'python')
                diversity_matrix[j, i] = -calc_syntax_match([func_j], func_i, 'python')

        
        diversity_scores = diversity_matrix.sum(axis=0)

        
        probabilities = np.exp(diversity_scores) / np.exp(diversity_scores).sum()

        
        individual_prob_pairs = list(zip(valid_population, probabilities))
        
        
        individual_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        
        selected_parent = np.random.choice(valid_population, p=probabilities, replace=False)
        
        
        cumulative_probability = 0.0
        top_50_percent = []
        
        for i, (individual, prob) in enumerate(individual_prob_pairs):
            cumulative_probability += prob
            top_50_percent.append((individual, prob))
            if cumulative_probability > 0.5:
                break
        
        return selected_parent, individual_prob_pairs

    def select_parent_top_50_percent(self) -> List[buffer.code_manipulation.Function]:
        """Get the top 50% probability individuals from the last parent selection.
        This is a convenience method that extracts the top 50% from select_parent results.
        """
        try:
            _, individual_prob_pairs = self.select_parent()
            
            
            selected_individuals = []
            cumulative_probability = 0.0
            
            for individual, prob in individual_prob_pairs:
                cumulative_probability += prob
                selected_individuals.append(individual)
                
                
                if cumulative_probability > 0.5:
                    break
            
            return selected_individuals
            
        except Exception as e:
            print(f"[Evolution] Error getting top 50% individuals: {e}")
            return []
    # def sample(self, **kwargs):
    #     """ Continuously gets prompts, samples programs, sends them for analysis. """
    
        
    #     while True:
    
            
    
    #         original_prompt = self._database.get_prompt()
    #         print("=========================获取到的提示词========================", original_prompt)
    
    #         if scientist_guidance:
    #             enhanced_code = f"""{scientist_guidance}

    # {original_prompt.code}"""
    #             print(f"[Generation] Enhanced prompt with scientist guidance")
    #         else:
    #             enhanced_code = original_prompt.code
                
    #         print(f"[Generation] Using prompt from database (island_id: {original_prompt.island_id})")
            
    #         reset_time = time.time()
    #         samples = self._llm.draw_samples(enhanced_code, self.config)
    #         sample_time = (time.time() - reset_time) / self._samples_per_prompt
    #         print(f"[Generation] Generated {len(samples)} samples in {sample_time:.3f}s per sample")

    
    #         for sample in samples:
    #             self._global_sample_nums_plus_one()
    #             cur_global_sample_nums = self._get_global_sample_nums()
    #             chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
    #             chosen_evaluator.analyse(
    #                 sample,
    #                 original_prompt.island_id,
    #                 original_prompt.version_generated,
    #                 **kwargs,
    #                 global_sample_nums=cur_global_sample_nums,
    #                 sample_time=sample_time
    #             )
            
    
    #         if self._scientist_enabled and self._scientist and self._dataset:
    #             next_guidance = self._run_scientist_analysis()
    #             print("========================科学家大模型的输出========================", next_guidance)
    #             if next_guidance:
    
    #                 scientist_guidance = next_guidance
    #                 print(f"[Scientist] Updated guidance for next iteration")
            
    #         print(f"[Iteration {self._iteration_count}] Completed\n")

        

    # def sample(self, **kwargs):
    #     """ Continuously gets prompts, samples programs, sends them for analysis. """
    
        
    #     while True:
    #         # stop the search process if hit global max sample nums
    #         if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
    #             break
            
    
    #         self._iteration_count += 1
    #         print(f"\n[Iteration {self._iteration_count}] Starting new iteration...")
            
    
    #         parent = None
    #         try:
    
    #             if hasattr(self._database, '_population') and self._database._population:
    #                 parent = self.select_parent()
    #                 print(f"[Evolution] Selected parent from population")
    #         except Exception as e:
    #             print(f"[Evolution] Could not select parent: {e}")
    #             parent = None
            
    
    #         if parent is not None:
    
    #             try:
    #                 original_prompt = self._database.get_prompt_from_function(parent)
    #                 print(f"[Generation] Using prompt from parent function")
    #             except Exception as e:
    #                 print(f"[Evolution] Error getting prompt from parent: {e}")
    #                 original_prompt = self._database.get_prompt()
    #                 print(f"[Generation] Fallback to database prompt")
    #         else:
    
    #             original_prompt = self._database.get_prompt()
    #             print(f"[Generation] Using prompt from database (island_id: {original_prompt.island_id})")
            
    #         print("=========================获取到的提示词========================", original_prompt)
            
    
    #         if scientist_guidance:
    #             enhanced_code = f"""{scientist_guidance}

    # {original_prompt.code}"""
    #             print(f"[Generation] Enhanced prompt with scientist guidance")
    #         else:
    #             enhanced_code = original_prompt.code
            
    #         reset_time = time.time()
    #         samples = self._llm.draw_samples(enhanced_code, self.config)
    #         sample_time = (time.time() - reset_time) / self._samples_per_prompt
    #         print(f"[Generation] Generated {len(samples)} samples in {sample_time:.3f}s per sample")

    
    
            
    #         for sample in samples:
    #             self._global_sample_nums_plus_one()
    #             cur_global_sample_nums = self._get_global_sample_nums()
    #             chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                
    
    #             result = chosen_evaluator.analyse(
    #                 sample,
    
    #                 original_prompt.version_generated,
    #                 **kwargs,
    #                 global_sample_nums=cur_global_sample_nums,
    #                 sample_time=sample_time
    #             )
                
    
    #             if result is not None:
    #                 offspring.append(result)
            
    
    #         if offspring and hasattr(self._database, '_merge_and_update_population'):
    #             try:
    #                 self._database._merge_and_update_population(offspring)
    #                 print(f"[Evolution] Updated population with {len(offspring)} offspring")
    #             except Exception as e:
    #                 print(f"[Evolution] Error updating population: {e}")
            
    
    #         if self._scientist_enabled and self._scientist and self._dataset:
    #             next_guidance = self._run_scientist_analysis()
    #             print("========================科学家大模型的输出========================", next_guidance)
    #             if next_guidance:
    
    #                 scientist_guidance = next_guidance
    #                 print(f"[Scientist] Updated guidance for next iteration")
            
    #         print(f"[Iteration {self._iteration_count}] Completed\n")



    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        scientist_guidance = None  
        
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            
            
            self._iteration_count += 1
            print(f"\n[Iteration {self._iteration_count}] Starting new iteration...")
            
            
            # parent = None
            # try:
            
            #     if hasattr(self._database, '_population') and self._database._population:
            #         parent = self.select_parent()
            #         print(f"[Evolution] Selected parent from population")
            # except Exception as e:
            #     print(f"[Evolution] Could not select parent: {e}")
            #     parent = None
            # parent = self.select_parent()
            selected_parent, all_individuals_with_probs = self.select_parent_and_probability()
            parent = selected_parent
            top_50_percent = []
            cumulative_prob = 0.0
            for individual, prob in all_individuals_with_probs:
                cumulative_prob += prob
                top_50_percent.append(individual)
                if cumulative_prob > 0.5:
                    break
            original_prompt = self._database.get_prompt_from_function(parent)
            
            # if parent is not None:
            
            #     try:
            #         original_prompt = self._database.get_prompt_from_function(parent)
            #         print(f"[Generation] Using prompt from parent function")
            #     except Exception as e:
            #         print(f"[Evolution] Error getting prompt from parent: {e}")
            #         original_prompt = self._database.get_prompt()
            #         print(f"[Generation] Fallback to database prompt")
            # else:
            
            #     original_prompt = self._database.get_prompt()
            #     print(f"[Generation] Using prompt from database (island_id: {original_prompt.island_id})")
            
            
            if scientist_guidance:
                enhanced_code = f"""{scientist_guidance}

    {original_prompt.code}"""
            else:
                enhanced_code = original_prompt.code
            
            reset_time = time.time()
            samples = self._llm.draw_samples(enhanced_code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            print(f"[Generation] Generated {len(samples)} samples in {sample_time:.3f}s per sample")
            total_population_size = len(all_individuals_with_probs)
            top_50_percent_size = len(top_50_percent)
            
            offspring = []  
            
            for sample in samples:
                
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                
                
                analysis_result = chosen_evaluator.analyse(
                    sample,
                    original_prompt.island_id if parent is None else None,  
                    original_prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                
                
                if analysis_result is not None:
                    new_function, residual_data = analysis_result
                    
                    
                    if new_function is not None:
                        offspring.append(new_function)
            
            
            if offspring and hasattr(self._database, '_merge_and_update_population'):
                try:
                    self._database._merge_and_update_population(offspring)
                    print(f"[Evolution] Updated population with {len(offspring)} offspring")
                except Exception as e:
                    print(f"[Evolution] Error updating population: {e}")
            
            
            if self._scientist_enabled and self._scientist and self._dataset:
                pareto_samples = self._build_scientist_samples_from_pareto_front()
                next_guidance = self._run_scientist_analysis(pareto_samples=pareto_samples)
                if next_guidance:
                    scientist_guidance = next_guidance
                    print(f"[Scientist] Updated guidance for next iteration")
            
            print(f"[Iteration {self._iteration_count}] Completed\n")





    # # mmmmmm
    # def sample(self, **kwargs):
    #     """Continuously gets prompts, samples programs, sends them for analysis."""
    
    #         # stop the search process if hit global max sample nums
    #         if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
    #             break

    #         parent = self.select_parent()

    
    #         prompt = self._database.get_prompt_from_function(parent)
    #         reset_time = time.time()
    #         samples = self._llm.draw_samples(prompt.code, self.config)
    #         sample_time = (time.time() - reset_time) / self._samples_per_prompt

    
    #         offspring = []
    #         for sample in samples:
    #             self._global_sample_nums_plus_one()
    #             cur_global_sample_nums = self._get_global_sample_nums()
    #             chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
    #             new_function = chosen_evaluator.analyse(
    #                 sample,
    #                 None,
    #                 prompt.version_generated,
    #                 **kwargs,
    #                 global_sample_nums=cur_global_sample_nums,
    #                 sample_time=sample_time
    #             )
    #             offspring.append(new_function)

    
    #         self._database._merge_and_update_population(offspring)

    # def _run_scientist_analysis(self) -> Optional[str]:
    #     """
    
    #     """
    #     try:
    #         print(f"[Scientist] Starting analysis for iteration {self._iteration_count}")
            
    
    
    #         current_prompt = self._database.get_prompt()
    #         island_id = current_prompt.island_id
            
    #         if island_id is None:
    #             print("[Scientist] No valid island_id found")
    #             return None
            
    
    #         try:
    #             best_score = self._database._best_score_per_island[island_id]
    #             best_program = self._database._best_program_per_island[island_id]
    #             best_scores_per_test = self._database._best_scores_per_test_per_island[island_id]
    #         except (IndexError, KeyError):
    #             print(f"[Scientist] Island {island_id} not found or has no data")
    #             return None
            
    #         if best_program is None or best_score <= -float('inf'):
    #             print(f"[Scientist] Island {island_id} has no valid program")
    #             return None
            
    
    #         code = str(best_program)
    #         if not code.strip():
    #             print(f"[Scientist] Island {island_id} has empty code")
    #             return None
            
    
    
    #         if best_scores_per_test and isinstance(best_scores_per_test, dict):
    #             residuals = best_scores_per_test.get("residuals", residuals)
            
    #         print(f"[Scientist] Analyzing island {island_id}: score={best_score:.6f}")
            
    
    #         best_eqs = [code]
    #         scores = [best_score]
    #         residuals_list = [residuals]
            
    
    #         print(f"[Scientist] Calling scientist LLM to analyze equation from island {island_id}")
    #         guidance = self._scientist.analyze_equations(
    #             best_eqs=best_eqs,
    #             scores=scores,
    #             residuals=residuals_list,
    #             context=self._problem_context
    #         )
            
    
    #         analysis_tasks = guidance.get('analysis_tasks', [])
    #         structure_insight_prompt_for_generator_LLM = guidance.get('structure_insight_prompt_for_generator_LLM', '')
            
    #         print(f"[Scientist] Scientist suggested {len(analysis_tasks)} analysis tasks")
            
    
    #         all_analysis_results = []
    #         combined_analysis_text = []
            
    #         for i, task in enumerate(analysis_tasks):
    #             analysis_tool = task.get('analysis_tool')
    #             variables = task.get('variables', [])
                
    #             print(f"[Scientist] Running analysis task {i+1}: {analysis_tool} on {', '.join(variables)}")
                
    #             try:
    
    #                 task_supervisor_output = {
    #                     "analysis_tool": analysis_tool,
    #                     "variables": variables
    #                 }
                    
    #                 analysis_result = data_analysis_runner.run_supervisor_tool(
    #                     supervisor_output=task_supervisor_output,
    #                     dataset=self._dataset,
    #                     var_index_map=self._var_index_map
    #                 )
                    
    
    #                 analysis_text = self._format_analysis_result(analysis_result)
                    
    #                 all_analysis_results.append({
    #                     "task_id": i+1,
    #                     "tool": analysis_tool,
    #                     "variables": variables,
    #                     "result": analysis_result,
    #                     "formatted_text": analysis_text
    #                 })
                    
    #                 combined_analysis_text.append(f"Task {i+1} ({analysis_tool} on {', '.join(variables)}): {analysis_text}")
                    
    #                 print(f"[Scientist] Task {i+1} completed: {analysis_text}")
                    
    #             except Exception as e:
    #                 error_text = f"Error running {analysis_tool}: {str(e)}"
    #                 print(f"[Scientist] Task {i+1} failed: {error_text}")
                    
    #                 all_analysis_results.append({
    #                     "task_id": i+1,
    #                     "tool": analysis_tool,
    #                     "variables": variables,
    #                     "error": str(e),
    #                     "formatted_text": error_text
    #                 })
                    
    #                 combined_analysis_text.append(f"Task {i+1} ({analysis_tool}): {error_text}")
            
    
    #         if structure_insight_prompt_for_generator_LLM:
    
    #             final_prompt = self._create_enhanced_prompt_multiple_analyses(
    #                 base_prompt=structure_insight_prompt_for_generator_LLM,
    #                 analysis_results=all_analysis_results,
    #                 combined_analysis_text=combined_analysis_text
    #             )
                
    #             print(f"[Scientist] Generated enhanced prompt with {len(all_analysis_results)} analysis results")
    #             return final_prompt
    #         else:
    #             print("[Scientist] No guidance prompt generated by scientist")
    #             return None
        
    #     except Exception as e:
    #         print(f"[Scientist] Error in scientist analysis: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    
    def _run_scientist_analysis(self, pareto_samples: List[Dict] = None) -> Optional[str]:
        try:
            print(f"[Scientist] Starting analysis for iteration {self._iteration_count}")
            
            if pareto_samples:
                all_samples = []
                all_scores = []
                all_residuals = []
                
                for sample_data in pareto_samples:
                    sample = sample_data.get('sample', '')
                    function = sample_data.get('function')
                    residual_info = sample_data.get('residual_data', {})
                    
                    if function and hasattr(function, 'score') and function.score:
                        
                        all_samples.append(sample)
                        all_scores.append(function.score)
                        
                        residuals_for_this_sample = []
                        for dataset_name in ("dataset_id", "dataset_ood"):
                            res_data = residual_info.get(dataset_name)
                            if isinstance(res_data, dict) and 'worst_region_analysis' in res_data:
                                worst_region = res_data['worst_region_analysis']
                                residuals_for_this_sample.append(worst_region)
                            else:
                                residuals_for_this_sample.append(0.0)
                        
                        all_residuals.append(residuals_for_this_sample)
                
                if all_samples:
                    guidance = self._scientist.analyze_equations(
                        best_eqs=all_samples,
                        scores=all_scores,
                        residuals=all_residuals,
                        context=f"{self._problem_context}\n\nAnalyzing equations selected from Pareto front with residual data."
                    )
                    
                    
                    analysis_tools = guidance.get('analysis_tools', [])
                    structure_insight_prompt_for_generator_LLM = guidance.get('structure_insight_prompt_for_generator_LLM', '')
                    
                    
                    analysis_results = None
                    if analysis_tools and self._dataset:
                        supervisor_output = {
                            "analysis_tools": analysis_tools
                        }
                        
                        try:
                            
                            analysis_results = data_analysis_runner.run_supervisor_tool(
                                supervisor_output=supervisor_output,
                                dataset=self._dataset
                            )
                            
                            print(f"[Scientist] Analysis tools completed: {len(analysis_results)} results")
                            
                        except Exception as e:
                            print(f"[Scientist] Error running analysis tools: {e}")
                            analysis_results = None
                    
                    if structure_insight_prompt_for_generator_LLM:
                        enhanced_prompt = self._create_enhanced_prompt_with_pareto_analysis(
                            base_prompt=structure_insight_prompt_for_generator_LLM,
                            sample_data_list=pareto_samples,
                            analysis_tools=analysis_tools,
                            analysis_results=analysis_results
                        )
                        return enhanced_prompt
                    elif structure_insight_prompt_for_generator_LLM:
                        return structure_insight_prompt_for_generator_LLM
            
            print("[Scientist] No valid Pareto-front residual data available, using fallback analysis")
            return self._run_scientist_analysis_fallback()
            
        except Exception as e:
            print(f"[Scientist] Error in scientist analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    # def _create_enhanced_prompt_with_pareto_analysis(
    #     self,
    #     base_prompt: str,
    #     sample_data_list: List[Dict],
    #     analysis_tool: List[str] = None,
    #     analysis_result: Dict = None
    # ) -> str:
    #     """
    
    #     """
    #     prompt_parts = []
        
    
    #     prompt_parts.append("# Scientific Analysis Based on Pareto Front Selection")
    #     prompt_parts.append("")
    #     prompt_parts.append(base_prompt)
    #     prompt_parts.append("")
        
    
    #     # prompt_parts.append("## Selected Samples from Pareto Front")
    #     # prompt_parts.append(f"**Analyzing {len(sample_data_list)} samples** selected from Pareto front:")
    #     # prompt_parts.append("")
        
    #     # for i, sample_data in enumerate(sample_data_list):
    #     #     function = sample_data.get('function')
    #     #     residual_info = sample_data.get('residual_data', {})
            
    #     #     if function and hasattr(function, 'score') and function.score:
    #     #         prompt_parts.append(f"### Sample {i+1}")
    #     #         prompt_parts.append(f"**Scores**: dataset_id={function.score[0]:.6f}, dataset_ood={function.score[1]:.6f}, time={function.score[2]:.6f}")
                
    
    #     #         if residual_info:
    #     #             prompt_parts.append("**Residual Analysis Summary:**")
    #     #             total_cases = len(residual_info)
    #     #             avg_residuals = []
    #     #             max_residuals = []
                    
    #     #             for test_case, res_data in residual_info.items():
    #     #                 if isinstance(res_data, dict) and 'worst_region_analysis' in res_data:
    #     #                     worst_region = res_data['worst_region_analysis']
    #     #                     mean_abs_residual = worst_region.get('mean_abs_residual', 0.0)
    #     #                     max_abs_residual = worst_region.get('max_abs_residual', 0.0)
    #     #                     avg_residuals.append(mean_abs_residual)
    #     #                     max_residuals.append(max_abs_residual)
                    
    #     #             if avg_residuals:
    #     #                 overall_avg = sum(avg_residuals) / len(avg_residuals)
    #     #                 overall_max = max(max_residuals) if max_residuals else 0.0
    #     #                 prompt_parts.append(f"- {total_cases} test cases analyzed")
    #     #                 prompt_parts.append(f"- Average residual: {overall_avg:.6f}")
    #     #                 prompt_parts.append(f"- Maximum residual: {overall_max:.6f}")
                        
    
    #     #                 if overall_avg > 0.1:
    #     #                     prompt_parts.append("  📊 **High residual levels** - Focus on reducing prediction errors")
    #     #                 elif overall_avg > 0.05:
    #     #                     prompt_parts.append("  📈 **Moderate residual levels** - Some improvement possible")
    #     #                 else:
    #     #                     prompt_parts.append("  ✅ **Low residual levels** - Good performance maintained")
                
    #     #         prompt_parts.append("")
        
    
    #     prompt_parts.append("## Data Analysis Insights")
    #     prompt_parts.append(f"**Analysis performed**: {analysis_tool} on variables {', '.join(variables)}")
    #     prompt_parts.append(f"**Result**: {analysis_text}")
    #     prompt_parts.append("")

    
    #     if analysis_result and "metric" in analysis_result:
    #         metric_value = analysis_result["metric"]
    #         result = analysis_result 
            
    #         if analysis_tool in ["pearson_corr", "spearman_corr"]:
    #             prompt_parts.append("## Correlation Analysis Insights")
    #             if abs(metric_value) > 0.7:
    #                 prompt_parts.append(f"🔍 **Strong correlation detected** (r={metric_value:.3f})")
    #                 prompt_parts.append(f"Consider incorporating a strong relationship between {variables[0]} and {variables[1]} in your equation.")
    #                 if metric_value > 0:
    #                     prompt_parts.append("The relationship appears to be positive - as one increases, the other increases.")
    #                 else:
    #                     prompt_parts.append("The relationship appears to be negative - as one increases, the other decreases.")
    #             elif abs(metric_value) > 0.3:
    #                 prompt_parts.append(f"📊 **Moderate correlation detected** (r={metric_value:.3f})")
    #                 prompt_parts.append(f"There is a moderate relationship between {variables[0]} and {variables[1]}.")
    #             else:
    #                 prompt_parts.append(f"📈 **Weak correlation detected** (r={metric_value:.3f})")
    #                 prompt_parts.append(f"The relationship between {variables[0]} and {variables[1]} appears weak or non-linear.")
                    
    #         elif analysis_tool == "lin_reg":
    #             r_squared = metric_value
    #             slope = analysis_result.get("slope", 0)
    #             intercept = analysis_result.get("intercept", 0)
                
    #             prompt_parts.append("## Linear Regression Analysis")
    #             if r_squared > 0.8:
    #                 prompt_parts.append(f"📐 **Strong linear relationship** (R²={r_squared:.3f})")
    #                 prompt_parts.append(f"Linear equation: {variables[1]} ≈ {slope:.3f} × {variables[0]} + {intercept:.3f}")
    #                 prompt_parts.append("Consider incorporating linear terms in your equation.")
    #             elif r_squared > 0.5:
    #                 prompt_parts.append(f"📈 **Moderate linear relationship** (R²={r_squared:.3f})")
    #                 prompt_parts.append("Linear terms may be partially effective, but consider non-linear relationships.")
    #             else:
    #                 prompt_parts.append(f"🔄 **Non-linear relationship suggested** (R²={r_squared:.3f})")
    #                 prompt_parts.append("Linear fit is poor - explore non-linear terms (polynomial, exponential, trigonometric).")
                    
    #         elif analysis_tool == "residual_var":
    #             prompt_parts.append("## Residual Variance Analysis")
    #             prompt_parts.append(f"**Residual variance**: {metric_value:.6f}")
    #             if metric_value > 1.0:
    #                 prompt_parts.append("High residual variance suggests significant unexplained variation.")
    #                 prompt_parts.append("Consider more complex functional forms or additional variables.")
    #             else:
    #                 prompt_parts.append("Low residual variance suggests the linear relationship captures most variation.")

    #         elif analysis_tool == "mutual_info":
    #             bins = result.get("details", {}).get("bins", "unknown")
    #             prompt_parts.append(f"## Mutual Information Analysis")
    #             prompt_parts.append(f"**Mutual information**: {metric_value:.6f} (bins={bins})")
    #             prompt_parts.append("Mutual information measures the dependency between variables. Higher values indicate stronger dependency.")
                

    #         elif analysis_tool == "fft_cross_freq":
    #             details = result.get("details", {})
    #             x_freq = details.get("x_dominant_freq", "unknown")
    #             y_freq = details.get("y_dominant_freq", "unknown")
    #             prompt_parts.append(f"## FFT Cross Frequency Analysis")
    #             prompt_parts.append(f"**Frequency difference**: {metric_value:.6f} (x_freq={x_freq}, y_freq={y_freq})")
    #             prompt_parts.append("FFT cross frequency measures the difference in dominant frequencies between two signals.")


    #         elif analysis_tool == "wavelet_corr":
    #             prompt_parts.append(f"## Wavelet Correlation Analysis")
    #             prompt_parts.append(f"**Wavelet correlation**: {metric_value:.6f}")
    #             prompt_parts.append("Wavelet correlation measures the energy similarity across scales between two signals.")


    #         elif analysis_tool == "pca_mapping":
    #             explained_variance = result.get("details", {}).get("explained_variance_ratio", [])[0]
    #             prompt_parts.append(f"## PCA Mapping Analysis")
    #             prompt_parts.append(f"**Explained variance ratio**: {metric_value:.6f} (explained_variance={explained_variance})")
    #             prompt_parts.append("PCA mapping shows the proportion of variance explained by the principal component.")

    #         elif analysis_tool == "lyapunov_relation":
    #             details = result.get("details", {})
    #             lyap_x = details.get("lyap_x", "unknown")
    #             lyap_y = details.get("lyap_y", "unknown")
    #             prompt_parts.append(f"## Lyapunov Relation Analysis")
    #             prompt_parts.append(f"**Lyapunov difference**: {metric_value:.6f} (lyap_x={lyap_x}, lyap_y={lyap_y})")
    #             prompt_parts.append("Lyapunov relation measures the difference in chaos between two signals.")

    #         elif analysis_tool == "corr_dim_relation":
    #             details = result.get("details", {})
    #             dim_x = details.get("corr_dim_x", "unknown")
    #             dim_y = details.get("corr_dim_y", "unknown")
    #             prompt_parts.append(f"## Correlation Dimension Relation Analysis")
    #             prompt_parts.append(f"**Dimension difference**: {metric_value:.6f} (dim_x={dim_x}, dim_y={dim_y})")
    #             prompt_parts.append("Correlation dimension relation quantifies the complexity difference between two signals.")


    #         elif analysis_tool == "ks_test_diff":
    #             p_value = result.get("p_value", "unknown")
    #             prompt_parts.append(f"## KS Test Difference Analysis")
    #             prompt_parts.append(f"**KS statistic**: {metric_value:.6f} (p_value={p_value})")
    #             prompt_parts.append("KS test difference measures the similarity between two distributions.")


    #         elif analysis_tool == "dtw_distance":
    #             prompt_parts.append(f"## DTW Distance Analysis")
    #             prompt_parts.append(f"**DTW distance**: {metric_value:.6f}")
    #             prompt_parts.append("DTW distance quantifies the similarity between two time series by aligning them.")


    #         elif analysis_tool == "granger_causality":
    #             details = result.get("details", {})
    #             lag_1_p = details.get("lag_1", "unknown")
    #             prompt_parts.append(f"## Granger Causality Analysis")
    #             prompt_parts.append(f"**Minimum p-value**: {metric_value:.6f} (lag_1_p={lag_1_p})")
    #             prompt_parts.append("Granger causality tests whether one time series can predict another.")
    

    #         elif analysis_tool == "mutual_info_regression_score":
    #             prompt_parts.append(f"## Mutual Information Regression Analysis")
    #             prompt_parts.append(f"**Mutual information score**: {metric_value:.6f}")
    #             prompt_parts.append("Mutual information regression measures the dependency between variables in a regression context.")

    #         elif analysis_tool == "ccm_causality":
    #             details = result.get("details", {})
    #             x_y = details.get("x:y", {})
    #             y_x = details.get("y:x", {})
    #             prompt_parts.append(f"## CCM Causality Analysis")
    #             prompt_parts.append(f"**CCM causality score**: {metric_value:.6f} (x->y={x_y}, y->x={y_x})")
    #             prompt_parts.append("CCM causality assesses the causal influence between two time series.")
            
    #         prompt_parts.append("")
        
    
    #     prompt_parts.append("## Generation Strategy")
    #     prompt_parts.append("Based on the analysis of Pareto front samples and data relationships:")
    #     prompt_parts.append("")
        
    
    #     if len(sample_data_list) > 1:
    #         prompt_parts.append("1. **Leverage diversity**: Multiple high-quality solutions have been identified")
    #         prompt_parts.append("2. **Combine strengths**: Look for common patterns across successful samples")
    #         prompt_parts.append("3. **Address weaknesses**: Focus on regions where residuals are consistently high")
    #     else:
    #         prompt_parts.append("1. **Build upon success**: Use the selected high-quality solution as foundation")
    #         prompt_parts.append("2. **Targeted improvement**: Focus on reducing residuals in identified problem areas")
        
    #     if analysis_result:
    #         prompt_parts.append("4. **Data-driven insights**: Incorporate findings from variable relationship analysis")
    #         prompt_parts.append("5. **Variable interactions**: Consider the significant correlations and patterns identified")
        
    #     prompt_parts.append("6. **Maintain balance**: Preserve the multi-objective balance that led to Pareto selection")
    #     prompt_parts.append("7. **Consider trade-offs**: Balance accuracy, generalization, and computational efficiency")
    #     prompt_parts.append("")
        
    #     return "\n".join(prompt_parts)

    def _create_enhanced_prompt_with_pareto_analysis(
        self,
        base_prompt: str,
        sample_data_list: List[Dict],
        analysis_tools: List[str] = None,
        analysis_results: Dict = None,
        raw_structure_data: Dict = None
    ) -> str:
        prompt_parts = []
        
        
        prompt_parts.append("# Scientific Analysis Based on Pareto Front Selection")
        prompt_parts.append("")
        
        
        if raw_structure_data and isinstance(raw_structure_data, dict):
            prompt_parts.append("## Mathematical Pattern Analysis Results")
            
            
            math_structures = raw_structure_data.get("mathematical_structures", {})
            if math_structures:
                prompt_parts.append("### Identified Mathematical Structures")
                
                functions = math_structures.get("recurring_functions", [])
                if functions:
                    prompt_parts.append(f"**Recurring Functions Found**: {', '.join(functions)}")
                
                patterns = math_structures.get("variable_interaction_patterns", [])
                if patterns:
                    prompt_parts.append(f"**Variable Interaction Patterns**: {', '.join(patterns)}")
                
                coeffs = math_structures.get("coefficient_relationships", [])
                if coeffs:
                    prompt_parts.append(f"**Coefficient Relationships**: {', '.join(coeffs)}")
                
                prompt_parts.append("")
            
            
            shared_elements = raw_structure_data.get("shared_elements", {})
            if shared_elements:
                prompt_parts.append("### Common Mathematical Elements")
                
                terms = shared_elements.get("common_terms", [])
                if terms:
                    prompt_parts.append(f"**Common Terms**: {', '.join(terms)}")
                
                nested = shared_elements.get("nested_structures", [])
                if nested:
                    prompt_parts.append(f"**Nested Structures**: {', '.join(nested)}")
                
                fractions = shared_elements.get("fraction_structures", [])
                if fractions:
                    prompt_parts.append(f"**Fraction Patterns**: {', '.join(fractions)}")
                
                prompt_parts.append("")
            
            
            interpretation = raw_structure_data.get("interpretation", {})
            if interpretation:
                prompt_parts.append("### Physical and Mathematical Interpretation")
                
                physical = interpretation.get("physical_relationships", [])
                if physical:
                    prompt_parts.append(f"**Physical Context**: {', '.join(physical)}")
                
                constraints = interpretation.get("mathematical_constraints", [])
                if constraints:
                    prompt_parts.append(f"**Mathematical Constraints**: {', '.join(constraints)}")
                
                characteristics = interpretation.get("optimal_solution_characteristics", [])
                if characteristics:
                    prompt_parts.append(f"**Solution Characteristics**: {', '.join(characteristics)}")
                
                prompt_parts.append("")
        
        
        prompt_parts.append("## Core Generation Guidance")
        prompt_parts.append(base_prompt)
        prompt_parts.append("")
        
        
        if analysis_tools and analysis_results:
            prompt_parts.append("## Data Analysis Results")
            prompt_parts.append(f"Applied {len(analysis_tools)} analysis tools to all variable pairs:")
            prompt_parts.append("")
            
            
            prompt_parts.append(f"**Tools used**: {', '.join(analysis_tools)}")
            prompt_parts.append("")
            
            
            metadata = analysis_results.get('_metadata', {})
            if metadata:
                n_variables = metadata.get('n_variables', 0)
                total_samples = metadata.get('total_samples', 0)
                prompt_parts.append(f"**Dataset overview**:")
                prompt_parts.append(f"- {n_variables} independent variables analyzed")
                prompt_parts.append(f"- {total_samples} total data points")
                prompt_parts.append("")
            
            
            for result_key, result_value in analysis_results.items():
                if result_key.startswith('_'):  
                    continue
                
                if isinstance(result_value, dict) and 'error' not in result_value:
                    
                    if '_' in result_key:
                        var_name, analysis_tool = result_key.split('_', 1)
                        var_name = var_name + "_y"
                    else:
                        continue
                    
                    
                    metric_value = result_value.get('metric', 0.0)
                    
                    if analysis_tool in ["pearson_corr", "spearman_corr","corr"]:
                        prompt_parts.append(f"## Correlation Analysis: {var_name}")
                        if abs(metric_value) > 0.7:
                            prompt_parts.append(f" **Strong correlation detected** (r={metric_value:.3f})")
                            prompt_parts.append(f"Consider incorporating a strong relationship between {var_name} and output in your equation.")
                            if metric_value > 0:
                                prompt_parts.append("The relationship appears to be positive - as one increases, the other increases.")
                            else:
                                prompt_parts.append("The relationship appears to be negative - as one increases, the other decreases.")
                        elif abs(metric_value) > 0.3:
                            prompt_parts.append(f" **Moderate correlation detected** (r={metric_value:.3f})")
                            prompt_parts.append(f"There is a moderate relationship between {var_name} and output.")
                        else:
                            prompt_parts.append(f" **Weak correlation detected** (r={metric_value:.3f})")
                            prompt_parts.append(f"The relationship between {var_name} and output appears weak or non-linear.")
                            
                    elif analysis_tool == "lin_reg":
                        r_squared = metric_value
                        slope = result_value.get("slope", 0)
                        intercept = result_value.get("intercept", 0)
                        
                        prompt_parts.append(f"## Linear Regression Analysis: {var_name}")
                        if r_squared > 0.8:
                            prompt_parts.append(f" **Strong linear relationship** (R²={r_squared:.3f})")
                            prompt_parts.append(f"Linear equation: output ≈ {slope:.3f} × {var_name} + {intercept:.3f}")
                            prompt_parts.append("Consider incorporating linear terms in your equation.")
                        elif r_squared > 0.5:
                            prompt_parts.append(f" **Moderate linear relationship** (R²={r_squared:.3f})")
                            prompt_parts.append("Linear terms may be partially effective, but consider non-linear relationships.")
                        else:
                            prompt_parts.append(f" **Non-linear relationship suggested** (R²={r_squared:.3f})")
                            prompt_parts.append("Linear fit is poor - explore non-linear terms (polynomial, exponential, trigonometric).")
                            
                    elif analysis_tool == "residual_var":
                        prompt_parts.append(f"## Residual Variance Analysis: {var_name}")
                        prompt_parts.append(f"**Residual variance**: {metric_value:.6f}")
                        if metric_value > 1.0:
                            prompt_parts.append("High residual variance suggests significant unexplained variation.")
                            prompt_parts.append("Consider more complex functional forms or additional variables.")
                        else:
                            prompt_parts.append("Low residual variance suggests the linear relationship captures most variation.")

                    elif analysis_tool == "mutual_info":
                        bins = result_value.get("details", {}).get("bins", "unknown")
                        prompt_parts.append(f"## Mutual Information Analysis: {var_name}")
                        prompt_parts.append(f"**Mutual information**: {metric_value:.6f} (bins={bins})")
                        prompt_parts.append("Mutual information measures the dependency between variables. Higher values indicate stronger dependency.")

                    elif analysis_tool == "fft_cross_freq":
                        details = result_value.get("details", {})
                        x_freq = details.get("x_dominant_freq", "unknown")
                        y_freq = details.get("y_dominant_freq", "unknown")
                        prompt_parts.append(f"## FFT Cross Frequency Analysis: {var_name}")
                        prompt_parts.append(f"**Frequency difference**: {metric_value:.6f} (x_freq={x_freq}, y_freq={y_freq})")
                        prompt_parts.append("FFT cross frequency measures the difference in dominant frequencies between two signals.")

                    elif analysis_tool == "wavelet_corr":
                        prompt_parts.append(f"## Wavelet Correlation Analysis: {var_name}")
                        prompt_parts.append(f"**Wavelet correlation**: {metric_value:.6f}")
                        prompt_parts.append("Wavelet correlation measures the energy similarity across scales between two signals.")

                    elif analysis_tool == "pca_mapping":
                        explained_variance = result_value.get("details", {}).get("explained_variance_ratio", [])
                        if explained_variance:
                            explained_variance = explained_variance[0]
                        else:
                            explained_variance = "unknown"
                        prompt_parts.append(f"## PCA Mapping Analysis: {var_name}")
                        prompt_parts.append(f"**Explained variance ratio**: {metric_value:.6f} (explained_variance={explained_variance})")
                        prompt_parts.append("PCA mapping shows the proportion of variance explained by the principal component.")

                    elif analysis_tool == "lyapunov_relation":
                        details = result_value.get("details", {})
                        lyap_x = details.get("lyap_x", "unknown")
                        lyap_y = details.get("lyap_y", "unknown")
                        prompt_parts.append(f"## Lyapunov Relation Analysis: {var_name}")
                        prompt_parts.append(f"**Lyapunov difference**: {metric_value:.6f} (lyap_x={lyap_x}, lyap_y={lyap_y})")
                        prompt_parts.append("Lyapunov relation measures the difference in chaos between two signals.")

                    elif analysis_tool == "corr_dim_relation":
                        details = result_value.get("details", {})
                        dim_x = details.get("corr_dim_x", "unknown")
                        dim_y = details.get("corr_dim_y", "unknown")
                        prompt_parts.append(f"## Correlation Dimension Relation Analysis: {var_name}")
                        prompt_parts.append(f"**Dimension difference**: {metric_value:.6f} (dim_x={dim_x}, dim_y={dim_y})")
                        prompt_parts.append("Correlation dimension relation quantifies the complexity difference between two signals.")

                    elif analysis_tool == "ks_test_diff":
                        p_value = result_value.get("p_value", "unknown")
                        prompt_parts.append(f"## KS Test Difference Analysis: {var_name}")
                        prompt_parts.append(f"**KS statistic**: {metric_value:.6f} (p_value={p_value})")
                        prompt_parts.append("KS test difference measures the similarity between two distributions.")

                    elif analysis_tool == "dtw_distance":
                        prompt_parts.append(f"## DTW Distance Analysis: {var_name}")
                        prompt_parts.append(f"**DTW distance**: {metric_value:.6f}")
                        prompt_parts.append("DTW distance quantifies the similarity between two time series by aligning them.")

                    elif analysis_tool == "granger_causality":
                        details = result_value.get("details", {})
                        lag_1_p = details.get("lag_1", "unknown")
                        prompt_parts.append(f"## Granger Causality Analysis: {var_name}")
                        prompt_parts.append(f"**Minimum p-value**: {metric_value:.6f} (lag_1_p={lag_1_p})")
                        prompt_parts.append("Granger causality tests whether one time series can predict another.")

                    elif analysis_tool == "mutual_info_regression_score":
                        prompt_parts.append(f"## Mutual Information Regression Analysis: {var_name}")
                        prompt_parts.append(f"**Mutual information score**: {metric_value:.6f}")
                        prompt_parts.append("Mutual information regression measures the dependency between variables in a regression context.")

                    elif analysis_tool == "ccm_causality":
                        details = result_value.get("details", {})
                        x_y = details.get("x:y", "unknown")
                        y_x = details.get("y:x", "unknown")
                        prompt_parts.append(f"## CCM Causality Analysis: {var_name}")
                        prompt_parts.append(f"**CCM causality score**: {metric_value:.6f} (x->y={x_y}, y->x={y_x})")
                        prompt_parts.append("CCM causality assesses the causal influence between two time series.")
                    
                    prompt_parts.append("")
            
            
            # prompt_parts.append("**Key findings summary:**")
            # significant_findings = []
            
            # for result_key, result_value in analysis_results.items():
            #     if result_key.startswith('_') or 'error' in result_value:
            #         continue
                    
            #     if '_' in result_key:
            #         var_name, tool_name = result_key.rsplit('_', 1)
            #     else:
            #         continue
                    
            #     metric_value = result_value.get('metric', 0.0)
                
            #     if tool_name in ["pearson_corr", "spearman_corr"] and abs(metric_value) > 0.5:
            #         significant_findings.append(f"- {var_name}: Strong correlation ({metric_value:.3f})")
            #     elif tool_name == "lin_reg" and metric_value > 0.3:
            #         significant_findings.append(f"- {var_name}: Good linear fit (R² = {metric_value:.3f})")
            #     elif tool_name == "mutual_info" and metric_value > 0.5:
            #         significant_findings.append(f"- {var_name}: High mutual information ({metric_value:.3f})")
            
            # if significant_findings:
            #     prompt_parts.extend(significant_findings)
            # else:
            #     prompt_parts.append("- No strongly significant patterns detected in current analysis")
            
            # prompt_parts.append("")
        
        
        prompt_parts.append("## Generation Strategy")
        prompt_parts.append("Based on the analysis of Pareto front samples and data relationships:")
        prompt_parts.append("")
        
        
        if len(sample_data_list) > 1:
            prompt_parts.append("1. **Leverage diversity**: Multiple high-quality solutions have been identified")
            prompt_parts.append("2. **Combine strengths**: Look for common patterns across successful samples")
            prompt_parts.append("3. **Address weaknesses**: Focus on regions where residuals are consistently high")
        else:
            prompt_parts.append("1. **Build upon success**: Use the selected high-quality solution as foundation")
            prompt_parts.append("2. **Targeted improvement**: Focus on reducing residuals in identified problem areas")
        
        if analysis_results:
            prompt_parts.append("4. **Data-driven insights**: Incorporate findings from variable relationship analysis")
            prompt_parts.append("5. **Variable interactions**: Consider the significant correlations and patterns identified")
        
        prompt_parts.append("6. **Maintain balance**: Preserve the multi-objective balance that led to Pareto selection")
        prompt_parts.append("7. **Consider trade-offs**: Balance accuracy, generalization, and computational efficiency")
        prompt_parts.append("")
        
        return "\n".join(prompt_parts)

    def _run_scientist_analysis_fallback(self) -> Optional[str]:
        try:
            
            if hasattr(self._database, '_population') and self._database._population:
                
                try:
                    selected_function = self.select_parent()  
                    
                    residual_data = getattr(selected_function, 'residual_data', None)
                    residuals_for_analysis = []
                    
                    if isinstance(residual_data, dict):
                        for dataset_name in ("dataset_id", "dataset_ood"):
                            res_data = residual_data.get(dataset_name)
                            if isinstance(res_data, dict) and 'worst_region_analysis' in res_data:
                                worst_region = res_data['worst_region_analysis']
                                residuals_for_analysis.append(worst_region)
                            else:
                                residuals_for_analysis.append(0.0)
                    else:
                        residuals_for_analysis = [0.0, 0.0]
                    
                    
                    guidance = self._scientist.analyze_equations(
                        best_eqs=[str(selected_function)],
                        scores=[selected_function.score if hasattr(selected_function, 'score') else [0.0, 0.0, 0.0]],
                        residuals=[residuals_for_analysis],
                        context=f"{self._problem_context}\n\nAnalyzing function selected through diversity-based parent selection."
                    )
                    
                    
                    analysis_tools = guidance.get('analysis_tools', [])
                    structure_insight_prompt_for_generator_LLM = guidance.get('structure_insight_prompt_for_generator_LLM', '')
                    
                    
                    analysis_results = None
                    if analysis_tools and self._dataset:
                        supervisor_output = {
                            "analysis_tools": analysis_tools
                        }
                        
                        try:
                            
                            analysis_results = data_analysis_runner.run_supervisor_tool(
                                supervisor_output=supervisor_output,
                                dataset=self._dataset
                            )
                        except Exception as e:
                            print(f"[Scientist] Error running analysis tools: {e}")
                            analysis_results = None
                    
                    if structure_insight_prompt_for_generator_LLM:
                        enhanced_prompt = f"""# Analysis Based on Selected Parent Function

    {structure_insight_prompt_for_generator_LLM}

    ## Selection Context
    - Function selected using diversity-based parent selection from population
    - {'Residual data available' if residual_data else 'No residual data available'}
    - Scores: {selected_function.score if hasattr(selected_function, 'score') else 'No scores available'}

    ## Data Analysis Results
    {f'Applied {len(analysis_tools)} tools to analyze all variable relationships' if analysis_tools else 'No data analysis tools applied'}

    Focus on building upon the selected function's strengths while addressing any identified weaknesses.
    """
                        return enhanced_prompt
                    
                except Exception as e:
                    print(f"[Scientist] Error using parent selection: {e}")
            
            return None
            
        except Exception as e:
            print(f"[Scientist] Error in fallback analysis: {e}")
            return None

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1


def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        # print("code:", code)
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = f"http://127.0.0.1:{Port}/completions"
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:
                    # print("111222")
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    # print("222333")
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    conn = http.client.HTTPSConnection("api.openai.com")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': f"Bearer {os.environ['API_KEY']}",
                        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']
                    
                    if self._trim:
                        response = _extract_body(response, config)
                    
                    all_samples.append(response)
                    break

                except Exception:
                    continue
        
        return all_samples
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            # print("===========================方程生成大模型的响应============================\n",response)
            return response if self._batch_inference else response[0]
