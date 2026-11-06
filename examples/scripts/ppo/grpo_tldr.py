# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/grpo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir "./outputs/grpo_tldr" \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 625 \
    --num_generations 2 \
    --max_completion_length 53 \
    --beta 0.05 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --eval_strategy no \
    --report_to wandb \
    --run_name grpo_tldr \
    --save_strategy no
"""


import os
import shutil
from dataclasses import dataclass, field

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    sft_model_path: str | None = field(
        default=None,
        metadata={"help": "Path to the SFT policy model used as the initial GRPO policy."},
    )
    reward_model_path: str | None = field(
        default=None,
        metadata={"help": "Path or model ID of the TL;DR reward model."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    if script_args.reward_model_path is None:
        raise ValueError("`--reward_model_path` is required for TL;DR GRPO reward scoring.")

    policy_model_path = script_args.sft_model_path or model_args.model_name_or_path

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(
        script_args.reward_model_path,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    peft_config = get_peft_config(model_args)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def prepare_dataset(dataset, tokenizer):
        def add_length(element):
            input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
            return {"lengths": len(input_ids)}

        return dataset.map(
            add_length,
            num_proc=getattr(script_args, "dataset_num_proc", None),
        )

    dataset_num_proc = getattr(script_args, "dataset_num_proc", None)
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=dataset_num_proc)

        train_dataset = train_dataset.remove_columns([c for c in ["lengths"] if c in train_dataset.column_names])
        if eval_dataset is not None:
            eval_dataset = eval_dataset.remove_columns([c for c in ["lengths"] if c in eval_dataset.column_names])

    assert "prompt" in train_dataset.column_names, "GRPOTrainer requires a `prompt` column."

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=reward_model,
        reward_processing_classes=reward_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save final policy model only after training.
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()

    trainer.save_model(training_args.output_dir)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)