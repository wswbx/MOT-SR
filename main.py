import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from mot_sr import pipeline
from mot_sr import config
from mot_sr import sampler
from mot_sr import evaluator
from mot_sr.scientist_defaults import DEFAULT_SCIENTIST_TOOLS


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str,default="./specs/specification_oscillator1_numpy.txt")
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)


parser.add_argument('--enable_scientist', type=bool, default=True, help='Enable scientist LLM analysis')
parser.add_argument('--scientist_port', type=int, default=5000, help='Port for scientist LLM')
parser.add_argument('--scientist_model', type=str, default="deepseek-coder", help='Model name for scientist LLM')


def find_best_split(X, y, search_range=range(10, 50), lambda_weight=10):
    best_loss = float('inf')
    best_id_mask = None
    best_bounds = (25, 75)
    best_low = None
    best_up = None

    for p in search_range:
        lower_percentile = p
        upper_percentile = 100 - p

        lower_bounds = np.percentile(X, lower_percentile, axis=0)
        upper_bounds = np.percentile(X, upper_percentile, axis=0)

        id_mask = np.all((X >= lower_bounds) & (X <= upper_bounds), axis=1)

        num_id = np.sum(id_mask)
        num_ood = len(X) - num_id

        symmetry_loss = np.sum(np.abs(lower_percentile - (100 - upper_percentile)))
        loss = abs(num_id - num_ood) + lambda_weight * symmetry_loss

        if loss < best_loss:
            best_loss = loss
            best_id_mask = id_mask
            best_bounds = (lower_percentile, upper_percentile)
            best_low = lower_bounds
            best_up = upper_bounds

    print(
        "[DataSplit] Selected percentile range: "
        f"{best_bounds[0]}%-{best_bounds[1]}%; "
        f"ID={np.sum(best_id_mask)}, OOD={len(X) - np.sum(best_id_mask)}"
    )

    X_id = X[best_id_mask]
    y_id = y[best_id_mask]
    X_ood = X[~best_id_mask]
    y_ood = y[~best_id_mask]

    return {'inputs': X_id, 'outputs': y_id}, {'inputs': X_ood, 'outputs': y_ood}, (best_low, best_up)


def build_scientist_config(args, X, y, dataset, problem_name):
    num_variables = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    var_index_map = {f"x{i}": i for i in range(num_variables)}
    problem_context = (
        f"Symbolic regression problem: {problem_name}. "
        f"Dataset has {num_variables} input variables and 1 target variable. "
        f"Training data shape: "
        f"{X.shape if hasattr(X, 'shape') else f'({len(X)}, {len(X[0])})'}"
        f" -> {y.shape if hasattr(y, 'shape') else len(y)}"
    )

    return {
        "endpoint": f"http://127.0.0.1:{args.scientist_port}/completions",
        "model_name": args.scientist_model,
        "api_key": "sk-no-key-required",
        "meta_prompt": """You are a scientific analyst AI specialized in symbolic regression.
Your role is to analyze equations and their performance data to provide guidance for generating better equations.
You should suggest specific data analysis tools and provide clear, actionable insights.""",
        "temperature": 0.7,
        "max_tokens": 512,
        "allowed_tools": list(DEFAULT_SCIENTIST_TOOLS),
        "allowed_variables": [f"x{i}" for i in range(num_variables)],
        "dataset": dataset,
        "var_index_map": var_index_map,
        "problem_context": problem_context,
        "min_islands_for_analysis": 3,
    }


if __name__ == '__main__':
    args = parser.parse_args()
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,)
    global_max_sample_num = 2000
    

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 
    
    id_data, ood_data, (lower_bounds, upper_bounds) = find_best_split(X, y)

    dataset = {
        'id': id_data,
        'ood': ood_data
}

    
    scientist_config = None
    if args.enable_scientist:
        scientist_config = build_scientist_config(args, X, y, dataset, problem_name)
        print(
            "[Scientist] Enabled: "
            f"model={scientist_config['model_name']}, "
            f"tools={len(scientist_config['allowed_tools'])}"
        )
    else:
        print("[Scientist] Scientist LLM disabled")
    
    pipeline.main(
        specification=specification,
        inputs={
        "dataset_id": id_data,
        "dataset_ood": ood_data
        },
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        scientist_config=scientist_config,  
        log_dir=args.log_path,
    )
