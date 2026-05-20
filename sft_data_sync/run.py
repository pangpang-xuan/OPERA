import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import gpt_usage



def write_to_file(file_path, data):
    with open(file_path, 'a+', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def init_file_if_needed(file_path, remove_cache):
    if os.path.exists(file_path) and remove_cache:
        os.remove(file_path)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass

def run(args):
    task = get_task(args.task)
    if args.naive_run:
        file = f'/logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        tmp_path = f'logs/{args.task}/Arena_ThinkingWriter_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    
    os.makedirs(os.path.dirname(file), exist_ok=True)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    init_file_if_needed(tmp_path, args.remove_cache)

    with open(tmp_path, 'r', encoding='utf-8') as f: # cache
        outputs = [json.loads(line) for line in f if line.strip()]

    start_index = args.task_start_index + len(outputs)
    end_index = args.task_end_index
    print(f"start index is {str(start_index)}, end index is {str(end_index)}")
                 
    for i in range(start_index, end_index):
        # solve
        if args.naive_run:
            output, ys, info = naive_solve(args, task, i) 
        else:
            output, ys, info = solve(args, task, i)
        
        results=[]
        results.append(output)
        with open(tmp_path, 'a+', encoding='utf-8') as f:
            for line in results:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=0.6)

    args.add_argument('--task', type=str, required=True)
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=-1)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run
    args.add_argument('--voting_agents', type=str, choices=['one', 'multi'])
    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    # thought generator, sample independent thoughts (used in Creative Writing)

    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    # state evaluator, whether to use the vote on states together (used in Creative Writing)
    
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')

    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--num_threads', type=int, default=1)
    args.add_argument("--remove-cache", default=False, action='store_true', help="remove cache")


    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
    print("process done")