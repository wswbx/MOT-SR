""" Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing

from mot_sr import code_manipulation
from mot_sr import buffer
from mot_sr import evaluator_accelerate
from mot_sr import evaluate_on_problems

class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.
        
        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility). 
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate


    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.
        
        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """
        fit_dataset = inputs['dataset_id']
        eval_dataset = inputs[test_input]

        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, fit_dataset, eval_dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        # if the process is not finished in time, terminate
        if process.is_alive():
            process.terminate()
            process.join()
            results = None, None,False
            equation_time = None  
        else:
            results = self._get_results(result_queue)
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)

        if results and len(results) == 4:
            grade, res, equation_time, runs_ok = results
            results = (grade, runs_ok)
            # print(f"拆分后的grade: {grade}, res: {res}, runs_ok: {runs_ok}, remark: {remark}")
        else:
            res = None
            equation_time = None  
            results = (None, False)  
        return results,res,equation_time


    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, None, None,False


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(
            self,
            program,
            function_to_run,
            function_to_evolve,
            fit_dataset,
            eval_dataset,
            numba_accelerate,
            result_queue,
    ):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            evolved_function = all_globals_namespace[function_to_evolve]
            results, res, equation_time = evaluate_on_problems.evaluate(
                fit_dataset,
                eval_dataset,
                evolved_function,
            )
            if not isinstance(results, (int, float)):
                result_queue.put((None, None, False))
                return
            # print("results:", results)
            result_queue.put((results,res ,equation_time, True))
        # if raise any exception, execution is failed
        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, None, None, False))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    # def analyse(
    #         self,
    #         sample: str,
    #         island_id: int | None,
    #         version_generated: int | None,
    #         **kwargs 
    # ) -> None:
    #     """ Compile the hypothesis sample into a program and executes it on test inputs. """
    #     new_function, program = _sample_to_program(
    #         sample, version_generated, self._template, self._function_to_evolve)
    #     scores_per_test = {}

    #     time_reset = time.time()
        
    #     for current_input in self._inputs:
    #         results, res = self._sandbox.run(
    #             program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
    #             self._timeout_seconds
    #         )
    #         print("--------*******----------/////////")

            
    #         print(program)
    #         print("1111111111111111")
    #         print(self._function_to_run)
    #         print("2222222222222222")
    #         print(self._function_to_evolve)
    #         print("3333333333333333")
    #         print(self._inputs)
    #         print("4444444444444444")
    #         print(current_input)
    #         print("5555555555555555")
    #         print(self._timeout_seconds)
    #         print("--------*******----------/////////")


    #         test_output, runs_ok = results
    #         # if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
    #         #     if not isinstance(test_output, (int, float)):
    #         #         print(f'Error: test_output is {test_output}')
    #         #         raise ValueError('@function.run did not return an int/float score.')
    #         #     scores_per_test[current_input] = test_output
            
    #         print("--------*******----------/////////")
    #         print("test_output")
    #         print(test_output)
    #         print(runs_ok)


    
    #         if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
    #             if isinstance(test_output, list) and all(isinstance(score, (int, float)) for score in test_output):
    #                 scores_per_test[current_input] = test_output


    #             elif isinstance(test_output, (int, float)):
    
    #             else:
    #                 print(f'Error: test_output is {test_output}')
    #                 raise ValueError('@function.run did not return a valid score.')
    #             scores_per_test[current_input] = test_output



    #     evaluate_time = time.time() - time_reset

    #     print(f"scores_per_test: {scores_per_test}")

    #     if scores_per_test:
    #         self._database.register_program(
    #             new_function,
    
    #             # island_id,
    #             scores_per_test,
    #             **kwargs,
    #             evaluate_time=evaluate_time
    
        
    #     else:
    #         profiler: profile.Profiler = kwargs.get('profiler', None)
    #         if profiler:
    #             global_sample_nums = kwargs.get('global_sample_nums', None)
    #             sample_time = kwargs.get('sample_time', None)
    #             new_function.global_sample_nums = global_sample_nums
    #             new_function.score = None
    #             new_function.sample_time = sample_time
    #             new_function.evaluate_time = evaluate_time

    #             print("new_function------------------")
    #             print(new_function)

    #             profiler.register_function(new_function)
    
    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    ) -> None:
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        residual_data_per_test = {}  

        time_reset = time.time()
        
        target_datasets = ['dataset_id', 'dataset_ood']
        for current_input in target_datasets:
            if current_input not in self._inputs:
                continue
            results, res, equation_time = self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            # print("=========================测试刚生成的res=========================")
            # print(res)
            if equation_time is None:
                equation_time = 9999.0  
            
            test_output, runs_ok = results
            
            
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if isinstance(test_output, list) and all(isinstance(score, (int, float)) for score in test_output):
                    scores_per_test[current_input] = test_output
                elif isinstance(test_output, (int, float)):
                    scores_per_test[current_input] = [test_output]  
                else:
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return a valid score.')
                
                scores_per_test[current_input] = test_output
                
                
                if res is not None:
                    residual_data_per_test[current_input] = res

        evaluate_time = time.time() - time_reset

        if scores_per_test:
            
            self._database.register_program(
                new_function,
                scores_per_test,
                residual_data=residual_data_per_test,  
                **kwargs,
                evaluate_time=evaluate_time,
                equation_time=equation_time
            )
            
            
            return new_function, residual_data_per_test
        
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                profiler.register_function(new_function)
            
            return None, None
