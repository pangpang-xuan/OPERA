# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated
from tensordict import TensorDict
import re
import math

class GlobalEMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def update(self, current_batch_logps):
        if not current_batch_logps:
            return 0.0
        batch_mean = np.mean(current_batch_logps)
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.alpha * batch_mean + (1 - self.alpha) * self.value
        return self.value


logp_ema_tracker = GlobalEMA(alpha=0.1)

@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val


REFLECTION_KEYWORDS = ["But", "Wait", "Hmm", "However", "Wait a minute"]

@torch.no_grad()
def compute_gt_ppl_batch(actor_rollout_wg, context_ids_list, gt_ids_list, tokenizer):
    original_size = len(context_ids_list)
    try:
        world_size = actor_rollout_wg.world_size 
    except:
        world_size = 32
    remainder = original_size % world_size
    if remainder != 0:
        padding_size = world_size - remainder
        context_ids_list = context_ids_list + [context_ids_list[-1]] * padding_size
        gt_ids_list = gt_ids_list + [gt_ids_list[-1]] * padding_size
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    # device = next(actor_rollout_wg.actor.parameters()).device
    combined_ids = [torch.cat([c, g]) for c, g in zip(context_ids_list, gt_ids_list)]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        combined_ids, 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    ).to(device)
    bsz, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    response_mask = torch.zeros_like(input_ids)
    for i, (c, g) in enumerate(zip(context_ids_list, gt_ids_list)):
        response_mask[i, len(c):len(c)+len(g)] = 1
    ppl_batch = DataProto(batch=TensorDict({
        "input_ids": input_ids,
        "responses": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }, batch_size=[bsz]))
    output = actor_rollout_wg.compute_log_prob(ppl_batch)
    log_probs = output.batch["old_log_probs"].cpu() 
    batch_res = []
    for i in range(log_probs.size(0)):
        current_mask = response_mask[i][1:].bool() 
        current_logps = log_probs[i]
        if current_mask.shape[0] > current_logps.shape[0]:
            current_mask = current_mask[:current_logps.shape[0]]
        valid_logps = current_logps[current_mask]
        batch_res.append(valid_logps.mean().item() if valid_logps.numel() > 0 else 0.0)
    return batch_res[:original_size]

def get_reflection_segments(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not match:
        return []
    content = match.group(1).strip()
    return [s.strip() for s in content.split("\n\n") if s.strip()]


def prepare_extra_info_for_batch(batch, actor_rollout_wg, tokenizer, config):
    responses_ids = batch.batch["responses"]
    prompts_ids = batch.batch["prompts"]
    uids = batch.non_tensor_batch["uid"]
    reward_model_info = batch.non_tensor_batch.get("reward_model", [])
    
    responses_str = tokenizer.batch_decode(responses_ids, skip_special_tokens=True)
    prompts_str = tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
    gt_texts = [info.get('ground_truth', '') if isinstance(info, dict) else str(info) for info in reward_model_info]
    task_types = [info.get('task_type', 'writing') for info in reward_model_info]
    all_tasks = []
    is_correct_list = []

    for i in range(len(responses_str)):
        model_res = responses_str[i]
        model_output_only = re.sub(r'<think>.*?</think>', '', model_res, flags=re.DOTALL)

        gt_text = gt_texts[i]
        task_type = task_types[i]
        is_correct = False
        # step 1 Judge math or writing
        if task_type == 'math':
            model_ans_match = re.findall(r'\\boxed\{(.*?)\}', model_output_only)
            model_answer = model_ans_match[-1] if model_ans_match else model_output_only
            is_correct = (gt_text.strip() == model_answer.strip())
        is_correct_list.append(is_correct)

        # --- Reward 1.2:
        # gt_think_match = re.search(r'<think>(.*?)</think>', gt_text, re.DOTALL)
        # gt_think_content = f"<think>{gt_think_match.group(1)}</think>" if gt_think_match else ""
        # hybrid_text = prompts_str[i] + f"<think>\n{gt_think_content.strip()}\n</think>\n\n" + model_output_only.strip() 
         
        hybrid_text = prompts_str[i] + model_res 
        all_tasks.append({
            'type': 'hybrid_full',
            'sample_idx': i,
            'input_ids': tokenizer.encode(hybrid_text, return_tensors="pt")[0]
        })

        initial_prefix = prompts_str[i] + "<think>"
        all_tasks.append({
            'type': 'initial',
            'sample_idx': i,
            'input_ids': tokenizer.encode(initial_prefix, return_tensors="pt")[0]
        })
        
        segments = get_reflection_segments(model_res)
        curr_prefix = prompts_str[i] + "<think>"
        for seg_idx, seg in enumerate(segments):
            curr_prefix += seg + "\n\n"
            all_tasks.append({
                'type': 'segment',
                'sample_idx': i,
                'seg_idx': seg_idx,
                'input_ids': tokenizer.encode(curr_prefix, return_tensors="pt")[0]
            })

    if not all_tasks:
        return []

    task_gt_ids = [tokenizer.encode(gt_texts[t['sample_idx']], return_tensors="pt")[0] for t in all_tasks]
    all_context_ids = [t['input_ids'] for t in all_tasks]

    all_logps = compute_gt_ppl_batch(actor_rollout_wg, all_context_ids, task_gt_ids, tokenizer)
    raw_all_logps = all_logps
    
    current_ema = logp_ema_tracker.update(raw_all_logps)
    all_logps = [0.8 * lp + 0.2 * current_ema for lp in raw_all_logps]

    temp_results = {i: {'hybrid_logp': 0.0, 'initial_logp': 0.0, 'seg_logps': []} for i in range(len(responses_str))}
    for task, logp in zip(all_tasks, all_logps):
        idx = task['sample_idx']
        if task['type'] == 'hybrid_full':
            temp_results[idx]['hybrid_logp'] = float(logp)
        elif task['type'] == 'initial':
            temp_results[idx]['initial_logp'] = float(logp)
        else:
            temp_results[idx]['seg_logps'].append(float(logp))
    
    uid_to_hybrid_logps = {}
    for i in range(len(responses_str)):
        uid = uids[i]
        if uid not in uid_to_hybrid_logps: uid_to_hybrid_logps[uid] = []
        uid_to_hybrid_logps[uid].append(temp_results[i]['hybrid_logp'])

    all_extra_info = []
    for i in range(len(responses_str)):
        group_lps = sorted(uid_to_hybrid_logps[uids[i]])
        curr_lp = temp_results[i]['hybrid_logp']
        n = len(group_lps)
        ppl_relative_score = (np.searchsorted(group_lps, curr_lp) / (n - 1)) if n > 1 else 1.0

        curr_hybrid_lp = temp_results[i]['hybrid_logp']
        base_lp = temp_results[i]['initial_logp']
        diff_lp = curr_hybrid_lp - base_lp
        ppl_improvement_score = 1.0 / (1.0 + math.exp(-diff_lp)) if diff_lp != 0 else 0.5

        # group_lps = uid_to_hybrid_logps[uids[i]]
        # curr_lp = temp_results[i]['hybrid_logp']
        # max_lp_in_group = max(group_lps)
        # if curr_lp >= max_lp_in_group:
        #     ppl_relative_score = 1.0
        # else:
        #     ppl_relative_score = 0.0

        reflection_raw_score = 0.0
        seg_lps = temp_results[i]['seg_logps']
        last_lp = temp_results[i]['initial_logp']
        segments = get_reflection_segments(responses_str[i])
        
        for j in range(len(seg_lps)):
            has_kw = any(kw.lower() in segments[j].lower() for kw in REFLECTION_KEYWORDS)
            if has_kw and (seg_lps[j] - last_lp > 0):
                reflection_raw_score += 1.0
            last_lp = seg_lps[j]
        
        reflection_score = np.tanh(reflection_raw_score / 2.0)

        task_type = task_types[i]
        if task_type == 'math':
            is_correct = is_correct_list[i]
            if is_correct:
                # total_reward = 0.8 + (0.2 * ppl_relative_score)
                total_reward = 1.0
            else:
                total_reward = 0.0
                ppl_relative_score = 0.0
                reflection_score = 0.0
        else:
            total_reward = 0.3 * ppl_relative_score + 0.7 * reflection_score

            # total_reward = reflection_score
            # total_reward = ppl_relative_score

        all_extra_info.append({
            'ppl_score': float(ppl_relative_score),
            'reflection_score': float(reflection_score),
            'total_score': float(total_reward),
            'ppl_ema_baseline': current_ema
        })

    return all_extra_info
