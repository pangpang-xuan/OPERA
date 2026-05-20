import re
import torch
import numpy as np
from verl import DataProto
import logging

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    default_res = {'score': 0.0, 'ppl': 0.0, 'reflection': 0.0}
    
    if extra_info is None or not isinstance(extra_info, dict):
        return default_res

    try:
        ppl_s = extra_info.get('ppl_score', 0.0)
        refl_s = extra_info.get('reflection_score', 0.0)
        total_s = extra_info.get('total_score', ppl_s + refl_s)

        return {
            'score': float(total_s),
            'ppl': float(ppl_s),
            'reflection': float(refl_s)
        }
    except (TypeError, ValueError):
        return default_res