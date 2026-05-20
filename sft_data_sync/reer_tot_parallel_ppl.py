import os
import openai
from openai import OpenAI
import requests
import torch
from transformers import AutoTokenizer
import vllm
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from utils import *
import logging
import argparse
import requests
import json
import math
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('openai').setLevel(logging.WARNING)


MAX_MODEL_LEN = 40960
MAX_NUM_SEQS = 16
VLLM_GPU_MEMORY_UTILIZATION = 0.75

inference_prompt = """You are an expert in many fields. Suppose you will give a specific final response, I need you to also write down the thought process behind this solution.
Here is a question:
{prompt}

Now, you need to think aloud and brainstorm in the mind. The thinking process involves thoroughly exploring questions through a systematic long thinking process. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Present your complete thought process within a single and unique `<think></think>` tag.

Your thought process must adhere to the following requirements:

1.  **Narrate in the first-person as if you are thinking aloud and brainstorming**
    Stick to the narrative of "I". Imagine you are brainstorming and thinking in the mind. Use verbalized, simple language.

2.  **Unify the thinking process and the final solution:**
    Your thought process must precisely correspond to a part of the final solution. Your thoughts progressively "grew" into the finished solution, making the solution feel like the inevitable product of your thinking.

3.  **Tone of Voice: Planning, Sincere, Natural, and Accessible**
    Imagine you are analyzing and planning what to do before you start to give the solution.  Your language should be plain and easy to understand, avoiding obscure professional jargon to explain complex thought processes clearly.

4.  **Logical Flow: Clear and Progressive**

5.  **Thinking Framework for deep thinking**
    To ensure your thinking is clear and deep, to showcase your thinking and planning to fulfill the task, below is what you might cover when you are thinking aloud and brainstorming.

    Understanding the user intent and the task: Before giving the solution, I need to thoroughly consider the fundamental purpose of the question.
    
    Establishing the content: I need to brainstorm a core creative idea and communication strategy centered around my objective.
    
6. Throughout the thinking process, I want to involve deep thinking and planning, and use deliberate self-critique/self-reflection in my thinking process. Trigger these by frequently using patterns such as `wait`, `maybe`, `let me`, etc. For example:
    - Hmm, maybe .. (other concrete thinking regarding the given request)
    - Let me think .. 
    - Wait no ..
    - But wait ..(might find something wrong with your previous thoughts)
    - Wait, that's a bit ..(reflections about previous decisions). Let me think .. (are thinking of other possibilities)
    - Wait, the user said ..(backtracking of previous information). So ..
    - Hmm...Alternatively, maybe ..(branching on other possibilities)
    - But ..

Now record your clear, complete, and logical thinking process within `<think></think>` tags. 
In the thinking process, make sure NO PAST TENSES, NO PAST TENSES, because this is the thought process before you are to write a final solution. You are planning what you will and you need to do.
Imagine you're thinking aloud and brainstorming. Write it as an internal monologue or a stream of consciousness. Do not use bullet points, numbers, or formal section headings.
"""

logging.info(f"GPU Number: {torch.cuda.device_count()}")
llm = vllm.LLM(
    "models/huggingface.co/Qwen/Qwen3-32B",
    tensor_parallel_size = torch.cuda.device_count(),
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True
)

tokenizer = llm.get_tokenizer()

def generate(prompt, thinking_process, n, stop):
    if stop is None:
        stop = []
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    logging.info(text+thinking_process)
    logging.info("-"*100)
    responses = llm.generate(
        [text + thinking_process],
        sampling_params=vllm.SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            n=n,
            max_tokens=32768,
            skip_special_tokens=True,
            stop = stop
        )
    )
    return responses

def calculate_rollout_and_ppl(prompt, thinking_process):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    combined_input = prompt_text + thinking_process
    try:
        res_obj = llm.generate(
            [combined_input],
            sampling_params=vllm.SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=8192,
                logprobs=1,
                skip_special_tokens=True,
                prompt_logprobs=1,
                stop = ["\nBut", "\nWait", "\nHmm", "\nLet me think", "\nHowever", "</think>", 
                        ". But", ". Wait", ". Hmm", ". Let me think", ". However"]
            )
        )[0]
    
    except Exception as e:
        # OOM
        logging.error(f"vLLM generate fatal error: {e}")
        res_obj = llm.generate(
            [combined_input],
            sampling_params=vllm.SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=8192,
                skip_special_tokens=True,
                stop = ["\nBut", "\nWait", "\nHmm", "\nLet me think", "\nHowever", "</think>", 
                        ". But", ". Wait", ". Hmm", ". Let me think", ". However"]
            )
        )[0]
        return res_obj.outputs[0].text, 100.0
    
    rollout_text = res_obj.outputs[0].text
    rollout_step = rollout_text.split("\n\n")[0].strip() # rethinking step
    gen_logprobs = []
    if res_obj.outputs[0].logprobs:
        for lp_dict in res_obj.outputs[0].logprobs:
            token_id = list(lp_dict.keys())[0]
            gen_logprobs.append(lp_dict[token_id].logprob)
    try:
        input_logprobs_struct = res_obj.prompt_logprobs
        input_logprobs = []
        if input_logprobs_struct:
            for item in input_logprobs_struct:
                if item is not None:
                    token_id = list(item.keys())[0]
                    input_logprobs.append(item[token_id].logprob)
        input_logprobs = input_logprobs[1:]
    except torch.cuda.OutOfMemoryError:
        logging.error("Input PPL calculation OOM. Using gen-only fallback.")
        input_logprobs = []
    
    all_logprobs = input_logprobs + gen_logprobs
    if not all_logprobs:
        return rollout_text, float('inf')
    avg_log_prob = np.mean(all_logprobs)
    total_ppl = np.exp(-avg_log_prob)
    logging.info(f"Input tokens: {len(input_logprobs)}, Gen tokens: {len(gen_logprobs)}, Total PPL: {total_ppl}")
    logging.info("-"*100)
    logging.info(f"all generate: {rollout_text}")
    logging.info(f"generate step is: {rollout_step}")
    return rollout_step, total_ppl

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=-1)
    args = args.parse_args()
    return args

def run(args):
    input_path = "thinkingwriter_training/addition_2k_math_data.jsonl"
    output_path = f"thinkingwriter_training/ppl/rethinkingmind_ppl_v0_addition_start{args.task_start_index}_end{args.task_end_index}.jsonl"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        inputs = [json.loads(line) for line in f if line.strip()]
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            outputs = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        outputs = []
    
    start_index = args.task_start_index + len(outputs)
    inputs = inputs[start_index:args.task_end_index]
    for idx, item in enumerate(inputs):
        ori_prompt = item["prompt"]
        logging.info(f"......start generating {idx}....."+ori_prompt)
        prompt = inference_prompt.format(prompt=ori_prompt)
        thinking_process = ""
        standard_stop = ["Wait", "But", "Hmm", "Let me think", "However"]
        reflection_count = 0
        while(1):
            responses = generate(prompt, thinking_process, 1, standard_stop)
            if "</think>" in responses[0].outputs[0].text:
                full_text = responses[0].outputs[0].text
                if "<think>" in thinking_process: # "<think>\n..\n\n" + "...</think>\n..."
                    best_rollout = thinking_process.split("<think>")[1].strip() + "\n\n" + full_text.split("</think>")[0].strip()
                else: # "" + "<think>\n...\n</think>\n..."
                    first_handle = full_text.split("</think>")[0].strip()
                    best_rollout = first_handle.split("<think>")[1].strip()
                break
            else:
                reflection_count += 1 # count the number of reflections
                thinking_process = thinking_process.strip() + "\n\n" + responses[0].outputs[0].text.strip()
                logging.info("start parallel")
                results = []
                for _ in range(4):
                    step, ppl = calculate_rollout_and_ppl(prompt, thinking_process.strip() + "\n\n")
                    results.append({"rollout_step": step, "ppl": ppl})
                best_result = min(results, key=lambda x: x["ppl"])
                logging.info(f"Selected: {best_result['rollout_step']}")
                thinking_process = thinking_process.strip() + "\n\n" + best_result['rollout_step'].strip()
        
        # reflection_count >= 20: or output over
        if reflection_count >= 20:
            standard_stop = ["</think>"] # output until stop thinking
            responses = generate(prompt, thinking_process, 1, standard_stop)
            full_text = responses[0].outputs[0].text
            best_rollout = thinking_process.split("<think>")[1].strip() + "\n\n" + full_text.strip()
        
        logging.info("............generate the output.........")
        responses = generate(ori_prompt, f"<think>\n{best_rollout}\n</think>\n", 1, None)
        full_text = responses[0].outputs[0].text
        output_rollout = f"<think>\n{best_rollout}\n</think>\n"+full_text
        item_result = [{
            "prompt":ori_prompt,
            "answer":item["answer"],
            "thinking_process":best_rollout,
            "output": output_rollout
        }]
        with open(output_path, 'a+', encoding='utf-8') as f:
            for item in item_result:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    args = parse_args()
    run(args)
    print("process done")
    
