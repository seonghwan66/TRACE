# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal PPO training script for English OASST2 root prompts.

export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2_cpu_offload.yaml \
    --num_processes 1 \
    examples/scripts/ppo/ppo_oasst2.py \
    --dataset_name OpenAssistant/oasst2 \
    --dataset_train_split train \
    --dataset_test_split validation \
    --max_prompt_length 384 \
    --output_dir ./outputs/ppo_oasst2_test \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
    --sft_model_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
    --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --local_rollout_forward_batch_size 1 \
    --response_length 96 \
    --stop_token eos \
    --kl_coef 0.05 \
    --missing_eos_penalty 1.0 \
    --eval_strategy no \
    --save_strategy no \
    --report_to wandb \
    --run_name ppo_oasst2_test

    """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import gc

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)
from transformers.utils import ModelOutput

from trl import ModelConfig, ScriptArguments
from trl.experimental.ppo import PPOConfig, PPOTrainer


PROMPTER_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endoftext|>"


# The OpenAssistant reward checkpoint uses this custom architecture.
class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

    def __init__(self, pooling: Literal["mean", "last"] = "last", **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling or "last"


@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    logits: torch.FloatTensor | None = None


class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if type(config) is GPTNeoXConfig:
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())

        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

        self.post_init()

    @property
    def score(self):
        # TRL PPO uses this head for token-level reward/value extraction.
        return self.out_proj

    def get_input_embeddings(self):
        return self.gpt_neox.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.gpt_neox.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        head_mask=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.gpt_neox(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1]
            else:
                # Correct for both left- and right-padded batches.
                last_idx = attention_mask.long().cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states[
                    torch.arange(hidden_states.size(0), device=hidden_states.device),
                    last_idx,
                ]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)
        if not return_dict:
            return (logits,) + outputs[1:]
        return GPTNeoXRewardModelOutput(logits=logits)


AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(
    GPTNeoXRewardModelConfig,
    GPTNeoXRewardModel,
)


@dataclass
class OASST2Arguments(ScriptArguments):
    dataset_name: str = field(default="OpenAssistant/oasst2")
    dataset_config: str | None = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="validation")
    max_prompt_length: int = field(default=384)


def is_training_prompt(example: dict) -> bool:
    text = str(example.get("text") or "").strip()

    return (
        example.get("role") == "prompter"
        and example.get("parent_id") in (None, "")
        and str(example.get("lang") or "").lower() == "en"
        and not bool(example.get("deleted", False))
        and bool(text)
    )


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_prompt_length: int,
    dataset_num_proc: int | None,
) -> Dataset:
    dataset = dataset.filter(
        lambda example: is_training_prompt(example),
        num_proc=dataset_num_proc,
    )

    def tokenize(example):
        prompt = (
            f"{PROMPTER_TOKEN}{example['text'].strip()}"
            f"{END_TOKEN}{ASSISTANT_TOKEN}"
        )
        input_ids = tokenizer(
            prompt,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=dataset_num_proc,
    )
    return dataset.filter(
        lambda example: 0 < example["lengths"] <= max_prompt_length,
        num_proc=dataset_num_proc,
    )


def grow_embeddings_if_required(model, tokenizer) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and len(tokenizer) > embeddings.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


def save_policy(trainer, output_dir: str, tokenizer) -> None:
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wrapped_model = getattr(
        trainer,
        "model_wrapped",
        trainer.model,
    )

    # Gather the trainer state once.
    full_state_dict = accelerator.get_state_dict(
        wrapped_model
    )

    # Keep only policy parameters.
    policy_state_dict = {
        key.removeprefix("policy."): value
        for key, value in full_state_dict.items()
        if key.startswith("policy.")
    }

    # IMPORTANT:
    # Release value-model and other wrapper states before serialization.
    del full_state_dict
    gc.collect()

    policy = accelerator.unwrap_model(
        trainer.model
    ).policy

    if accelerator.is_main_process:
        policy.save_pretrained(
            output_dir,
            state_dict=policy_state_dict,
            save_function=accelerator.save,
            safe_serialization=True,
            max_shard_size="1GB",
        )

        tokenizer.save_pretrained(output_dir)

        del policy_state_dict
        gc.collect()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = HfArgumentParser((OASST2Arguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    dtype = (
        model_args.dtype
        if model_args.dtype in ("auto", None)
        else getattr(torch, model_args.dtype)
    )
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.sft_model_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        **model_kwargs,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        **model_kwargs,
    )

    for model in (policy, ref_policy, reward_model, value_model):
        grow_embeddings_if_required(model, tokenizer)

    policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.config.pad_token_id = tokenizer.pad_token_id

    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    load_kwargs = {}
    if script_args.dataset_config:
        load_kwargs["name"] = script_args.dataset_config
    dataset = load_dataset(script_args.dataset_name, **load_kwargs)

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(
            dataset[script_args.dataset_train_split],
            tokenizer,
            script_args.max_prompt_length,
            training_args.dataset_num_proc,
        )
        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_dataset = prepare_dataset(
                dataset[script_args.dataset_test_split],
                tokenizer,
                script_args.max_prompt_length,
                training_args.dataset_num_proc,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No OASST2 prompts remain after filtering.")

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    save_policy(trainer, training_args.output_dir, tokenizer)
