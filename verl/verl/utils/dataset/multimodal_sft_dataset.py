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
Multi-Modal SFT dataset
"""
import os
import json
from typing import Union, Dict, List

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from PIL import Image

from verl.utils import hf_processor
from verl.utils.dataset import SFTDataset
from verl.utils.model import compute_position_id_with_mask

# Standard practice in Hugging Face ecosystem for ignoring token IDs in loss calculation.
IGNORE_TOKEN_ID = -100


class MultiModalSFTDataset(SFTDataset):
    """
    This is an in-memory Multi-Modal SFTDataset for single-turn conversation.

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, ListConfig], tokenizer, config, processor=None, model_config=None):
        prompt_key = config.get("prompt_key", "prompt")
        response_key = config.get("response_key", "response")
        self.image_key = config.get("image_key", "image")
        self.max_length = config.get("max_length", 2048)
        self.image_root_dir = config.get("image_root_dir", "")
        self.model_config = model_config

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        if processor is None:
            raise ValueError("Processor must be provided for MultiModalSFTDataset")
        if isinstance(processor, str):
            self.processor: ProcessorMixin = hf_processor(processor, trust_remote_code=True)
        else:
            self.processor: ProcessorMixin = processor
        self.tokenizer: PreTrainedTokenizer = self.processor.tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key

        dataframes = [pd.read_parquet(f) for f in self.parquet_files]
        self.dataframe = pd.concat(dataframes).reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        prompt = row[self.prompt_key]
        response = row[self.response_key]
        image_file = row[self.image_key]
        
        # Extract ground truth bounding box from extra_info
        gt_bbox = None
        if "extra_info" in row:
            try:
                extra_info = json.loads(row["extra_info"])
                if "bbox_normalized" in extra_info:
                    gt_bbox = extra_info["bbox_normalized"]
            except (json.JSONDecodeError, TypeError):
                # Handle cases where extra_info might not be a valid JSON string
                pass

        image = Image.open(os.path.join(self.image_root_dir, image_file)).convert("RGB")
        image = image.resize((448, 448))

        # THE CRITICAL FIX: Manually inject original_size into the image's info dict.
        # The Qwen2VLProcessor will use this to correctly generate the necessary parameters internally.
        if "original_size" not in image.info:
            image.info["original_size"] = image.size

        # 可选：统一图像尺寸以避免 pixel_values 尺寸不一致的问题
        # 如果智能填充的 collate_fn 不工作，可以取消注释下面这行
        # image = image.resize((420, 420))  # 统一到固定尺寸

        # Format conversation for the processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]

        # Apply chat template to get the prompt part of the conversation
        prompt_str = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Add the assistant's response for the full conversation
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        full_str = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        
        # Tokenize the prompt string to get its length, passing image is important
        prompt_inputs = self.processor(text=[prompt_str], images=[image], return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        # Tokenize the full string with padding and truncation handled by the processor
        # For pixel_values consistency, we can try setting a fixed size or padding
        model_inputs = self.processor(
            text=[full_str],
            images=[image],
            return_tensors="pt",
            padding="max_length",  # 恢复padding来处理text
            truncation=True,
            max_length=self.max_length,
        )
        
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        
        # Create position_ids from the final attention mask
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create loss_mask (1 for response, 0 for prompt and padding)
        loss_mask = torch.zeros_like(attention_mask)
        response_end_index = attention_mask.sum()
        loss_mask[prompt_len:response_end_index] = 1
        
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            # Pass the ground truth bounding box to the trainer
            "gt_bbox": gt_bbox,
        }

        if "pixel_values" in model_inputs:
            data_dict["pixel_values"] = model_inputs.pixel_values
        if "image_grid_thw" in model_inputs:
            data_dict["image_grid_thw"] = model_inputs.image_grid_thw[0]

        return data_dict