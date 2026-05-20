import os
import re
import json
import random
import logging
from collections import Counter
import math


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def vote_outputs_unwrap(vote_outputs, n_candidates):
    votes = []
    for vote_output in vote_outputs.outputs:
        logging.info("voting result")
        logging.info(vote_output.text)
        logging.info("-"*100)
        matches = re.findall(r'\\boxed\{([^{}]+)\}', vote_output.text)
        if matches:
            last_boxed = matches[-1]
            try:
                vote = int(last_boxed)-1
                if 0<=vote<len(vote_outputs.outputs):
                    votes.append(vote)
            except ValueError:
                print(f"Invalid boxed answer: {last_boxed}")
        else:
            print(f"vote no match: [{vote_output.text}]")
    if not votes:
        return 0  # random choice
    counter = Counter(votes)
    max_votes = max(counter.values())
    max_candidates = [k for k, v in counter.items() if v == max_votes]
    final_vote = random.choice(max_candidates)
    return final_vote

def vote_prompt_wrap(x: str, ys: list, thinking_process:str) -> str:
    prompt = f"Question:\n{x}\n\nPrevious Reasoning Process:\n{thinking_process}\n\nCandidate Choices:\n"
    for i, y in enumerate(ys, 1):
        # y = y.replace('Plan:\n', '')
        # TODO: truncate the plan part?
        prompt += f'Choice {i}:\n{y}\n'
    return prompt[:-1] # remove the last \n

def entropy_from_logprobs(logprobs_dict):
    if not logprobs_dict:
        return 0.0
    logps = list(logprobs_dict.values())
    max_logp = max(logps)
    exp_ps = [math.exp(lp - max_logp) for lp in logps]
    Z = sum(exp_ps)
    if Z <= 0 or math.isnan(Z):
        return 0.0
    probs = [p / Z for p in exp_ps]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return entropy