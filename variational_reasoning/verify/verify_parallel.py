"""
starting command:
python -m variational_reasoning.verify.verify_parallel --dataset_idx 
"""

import os
import random
from tqdm.auto import tqdm
import datasets
from concurrent.futures import ProcessPoolExecutor, as_completed

from variational_reasoning.utils.code_execution_apps import process_dataset_parallel as verify_code_apps
from variational_reasoning.utils.code_execution_apps import process_dataset_parallel as verify_code_taco
from skythought.evals.util.math_parsing_util import extract_answer, math_equal
from skythought.evals.util.common import TimeoutException, timeout
import argparse 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_idx", type=int, help="")
parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
parser.add_argument('--initial_reasoning_model', type=str, default="zhouxiangxin/Initial-Reasoning-4B", required=True, help='Model path of initial reasoning model')
args = parser.parse_args()
    



def push_dataset_to_hf(dataset, repo_id):
    datadict = datasets.DatasetDict({'train': dataset})
    datadict.push_to_hub(repo_id)


@timeout(5)  # Add timeout of 5 seconds
def _check_numina_correctness(reference_solution, generated_solution):
    solution = extract_answer(reference_solution)
    pred = extract_answer(generated_solution)
    correctness = math_equal(pred, solution)
    return correctness


def check_numina_correctness(reference_solution, generated_solution):
    try:
        correctness = _check_numina_correctness(reference_solution, generated_solution)
    except Exception as e:
        print(f"Exception: {e}")
        correctness = False
    return correctness


def check_numina_correctness_parallel(args_list):
    results = []
    for reference_solution, generated_solution in args_list:
        results.append(check_numina_correctness(reference_solution, generated_solution))
    return results


# {'numina': 10500, 'apps': 3299, 'taco': 2096, 'ruc': 815}
    
    
bespoke_dataset = datasets.load_dataset("zhouxiangxin/Bespoke-Stratos-17k-Source")['train']
## source datasets 
numina_dataset = datasets.load_dataset("zhouxiangxin/numina_all_subsets_formatted")['train']
# dict_keys(['source', 'problem', 'solution', 'messages', 'gpt_difficulty', 'gpt_difficulty_parsed', 'prompt', 'answer'])
apps_dataset = datasets.load_dataset("zhouxiangxin/apps")['test']
# dict_keys(['problem_id', 'question', 'solutions', 'input_output', 'difficulty', 'url', 'starter_code'])
taco_dataset = datasets.load_dataset("zhouxiangxin/TACO_subset")['train']
# dict_keys(['question', 'solutions', 'starter_code', 'input_output', 'difficulty', 'raw_tags', 'name', 'source', 'tags', 'skill_types', 'url', 'Expected Auxiliary Space', 'time_limit', 'date', 'picture_num', 'memory_limit', 'Expected Time Complexity'])
ruc_dataset = datasets.load_dataset("RUC-AIBOX/long_form_thought_data_5k")['train']
# dict_keys(['question', 'combined_text', 'domain'])


if '/' in args.posterior_name:
    clean_posterior_name = args.posterior_name.split('/')[-1]
else:
    clean_posterior_name = args.posterior_name
dataset_idx = args.dataset_idx
dataset_name = f"{clean_posterior_name}-{dataset_idx}"
initial_dataset = datasets.load_dataset(f"zhouxiangxin/{dataset_name}-{dataset_idx}-initial_response")['train']


apps_tasks = []
taco_tasks = []
numina_tasks = []  
ruc_tasks = []     

## collect apps & taco tasks 
for initial_data in tqdm(initial_dataset):
    prompt_idx = initial_data['prompt_idx'] # int 
    bespoke_data = bespoke_dataset[prompt_idx]

    source = bespoke_data['source']
    source_idx = int(source.split('/')[1])
    print(f'source: {source}')

    generated_solution = initial_data['initial_responses']
    
    if source.startswith('apps'):
        generated_response_dict = {
            'deepseek_solution': generated_solution,
            'input_output': apps_dataset[source_idx]['input_output'],
            'solutions': apps_dataset[source_idx]['solutions'],
        }
        apps_tasks.append(generated_response_dict)
    elif source.startswith('taco'):
        generated_response_dict = {
            'deepseek_solution': generated_solution,
            'input_output': taco_dataset[source_idx]['input_output'],
        }
        taco_tasks.append(generated_response_dict)
    elif source.startswith('numina'):
        gt_solution = numina_dataset[source_idx]['solution']
        reference_solution = bespoke_data['solution_content']
        numina_tasks.append((gt_solution, reference_solution, generated_solution))
    elif source.startswith('ruc'):
        gt_solution = ruc_dataset[source_idx]['combined_text']
        reference_solution = bespoke_data['solution_content']
        ruc_tasks.append((gt_solution, reference_solution, generated_solution))
        
apps_tasks_dataset = datasets.Dataset.from_list(apps_tasks)
print(f'num apps_tasks: {len(apps_tasks_dataset)}')
print(f'num taco_tasks: {len(taco_tasks)}')
print(f'num numina_tasks: {len(numina_tasks)}')
print(f'num ruc_tasks: {len(ruc_tasks)}')

## processing apps tasks 
verification_results_apps = verify_code_apps(apps_tasks_dataset, num_proc=64)
verification_results_taco = verify_code_taco(taco_tasks, num_cpus=64, batch_size=2048)


def process_numina_ruc_tasks(tasks, num_workers=64):
    results = []
    
    args_list = []
    for gt_solution, reference_solution, generated_solution in tasks:
        args_list.append((gt_solution, generated_solution))
        args_list.append((reference_solution, generated_solution))
    
    # batch process
    batch_size = 100
    batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_numina_correctness_parallel, batch) for batch in batches]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing numina/ruc tasks"):
            results.extend(future.result())
    
    # reorganize
    final_results = []
    for i in range(0, len(results), 2):
        correctness_1 = results[i]
        correctness_2 = results[i + 1]
        final_results.append(correctness_1 or correctness_2)
    
    return final_results

if numina_tasks:
    numina_results = process_numina_ruc_tasks(numina_tasks)
else:
    numina_results = []

if ruc_tasks:
    ruc_results = process_numina_ruc_tasks(ruc_tasks)
else:
    ruc_results = []

print(f'numina_results: {len(numina_results)}')
print(f'ruc_results: {len(ruc_results)}')
print(f'verification_results_apps: {verification_results_apps}')
print(f'verification_results_taco: {verification_results_taco}')

apps_cnt = 0 
taco_cnt = 0
numina_cnt = 0
ruc_cnt = 0

correctness_list = []

for initial_data in tqdm(initial_dataset):
    prompt_idx = initial_data['prompt_idx'] # int 
    bespoke_data = bespoke_dataset[prompt_idx]

    source = bespoke_data['source']
    source_idx = int(source.split('/')[1])
    print(f'source: {source}')

    generated_solution = initial_data['initial_responses']

    if source.startswith('numina'):
        correctness = numina_results[numina_cnt] if numina_results else False
        correctness_list.append(correctness)
        numina_cnt += 1
    elif source.startswith('apps'):
        correctness = verification_results_apps[apps_cnt]['correctness']
        correctness_list.append(correctness)
        apps_cnt += 1
    elif source.startswith('taco'):
        correctness = verification_results_taco[taco_cnt]['correctness']
        correctness_list.append(correctness)
        taco_cnt += 1
    elif source.startswith('ruc'):
        correctness = ruc_results[ruc_cnt] if ruc_results else False
        correctness_list.append(correctness)
        ruc_cnt += 1
    else:
        raise ValueError(f'Unknown source: {source}')


assert len(correctness_list) == len(initial_dataset)
initial_dataset = initial_dataset.add_column('correctness', correctness_list)
push_dataset_to_hf(initial_dataset, f"zhouxiangxin/{dataset_name}-initial_response-verified")

