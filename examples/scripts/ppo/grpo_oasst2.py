# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal GRPO baseline for English OASST2 root prompts.

This script deliberately matches ppo_oasst2.py / trace_oaasst2.py in:

- OASST2 prompt filtering
- OpenAssistant prompt formatting
- policy initialization
- reward-model checkpoint
- maximum prompt/completion lengths
- KL coefficient
- missing-EOS penalty
- seed and generated-trajectory budget

GRPO is critic-free, so there is no value model. The reference model is created
internally by GRPOTrainer when --beta is non-zero.

Smoke test
----------

export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2_cpu_offload_grpo.yaml \
    --num_processes 1 \
    examples/scripts/ppo/grpo_oasst2.py \
    --dataset_name OpenAssistant/oasst2 \
    --dataset_train_split train \
    --dataset_test_split validation \
    --max_prompt_length 384 \
    --reward_max_length 480 \
    --output_dir ./outputs/grpo_oasst2_v1 \
    --learning_rate 3e-6 \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 625 \
    --num_generations 2 \
    --num_iterations 1 \
    --max_completion_length 96 \
    --temperature 1.0 \
    --top_p 1.0 \
    --beta 0.05 \
    --loss_type grpo \
    --scale_rewards group \
    --missing_eos_penalty 1.0 \
    --model_name_or_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
    --sft_model_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
    --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --eval_strategy no \
    --save_strategy no \
    --report_to wandb \
    --run_name grpo_oasst2_v1

Main run
--------
Use --max_steps 625. With one process, batch size 1, gradient accumulation 16,
and num_generations 2, each optimizer step consumes 16 generated trajectories;
625 steps therefore correspond to 10,000 trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer


PROMPTER_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endoftext|>"


# ---------------------------------------------------------------------------
# OpenAssistant Pythia reward-model architecture
# ---------------------------------------------------------------------------


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
                pooled = (
                    (hidden_states * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp_min(1.0)
                )
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


# ---------------------------------------------------------------------------
# Arguments and dataset wrapper
# ---------------------------------------------------------------------------


@dataclass
class OASST2GRPOArguments(ScriptArguments):
    dataset_name: str = field(default="OpenAssistant/oasst2")
    dataset_config: str | None = field(default=None)
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="validation")
    sft_model_path: str | None = field(default=None)
    reward_model_path: str | None = field(default=None)
    max_prompt_length: int = field(default=384)
    reward_max_length: int = field(
        default=480,
        metadata={
            "help": (
                "Maximum reward-model input length. For the fair OASST2 setup, "
                "use max_prompt_length + max_completion_length = 384 + 96 = 480."
            )
        },
    )
    missing_eos_penalty: float = field(default=1.0)


def is_training_prompt(example: dict) -> bool:
    """Use the exact same OASST2 subset as PPO and TRACE."""
    text = str(example.get("text") or "").strip()
    return (
        example.get("role") == "prompter"
        and example.get("parent_id") in (None, "")
        and str(example.get("lang") or "").lower() == "en"
        and not bool(example.get("deleted", False))
        and bool(text)
    )


def format_prompt(text: str) -> str:
    return (
        f"{PROMPTER_TOKEN}{str(text).strip()}"
        f"{END_TOKEN}{ASSISTANT_TOKEN}"
    )


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    max_prompt_length: int,
    dataset_num_proc: int | None,
) -> Dataset:
    """Filter and format OASST2 exactly as in the PPO/TRACE wrappers."""
    dataset = dataset.filter(
        is_training_prompt,
        num_proc=dataset_num_proc,
    )

    def add_prompt_and_length(example):
        prompt = format_prompt(example["text"])
        input_ids = tokenizer(
            prompt,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        return {
            "prompt": prompt,
            "lengths": len(input_ids),
        }

    dataset = dataset.map(
        add_prompt_and_length,
        remove_columns=dataset.column_names,
        num_proc=dataset_num_proc,
    )
    dataset = dataset.filter(
        lambda example: 0 < example["lengths"] <= max_prompt_length,
        num_proc=dataset_num_proc,
    )
    return dataset.remove_columns(["lengths"])


def grow_embeddings_if_required(model, tokenizer) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is not None and len(tokenizer) > embeddings.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


# ---------------------------------------------------------------------------
# Reward scorer
# ---------------------------------------------------------------------------


class OASST2RewardScorer:
    """Score exact prompt/completion token sequences with the frozen OASST RM."""

    def __init__(
        self,
        reward_model_path: str,
        tokenizer,
        reward_max_length: int,
        missing_eos_penalty: float,
        dtype,
    ):
        self.tokenizer = tokenizer
        self.reward_max_length = reward_max_length
        self.missing_eos_penalty = missing_eos_penalty
        self.device = PartialState().device

        model_kwargs = {}
        if dtype not in (None, "auto"):
            model_kwargs["dtype"] = dtype

        self.model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            **model_kwargs,
        )
        grow_embeddings_if_required(self.model, tokenizer)
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _clean_completion_ids(self, token_ids) -> tuple[list[int], bool]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        cleaned: list[int] = []
        has_eos = False
        for token_id in token_ids:
            token_id = int(token_id)

            # GRPOTrainer normally supplies unpadded completion lists. This
            # branch only prevents an unexpected trailing PAD from entering RM.
            if token_id == self.tokenizer.pad_token_id:
                continue

            cleaned.append(token_id)
            if token_id == self.tokenizer.eos_token_id:
                has_eos = True
                break

        return cleaned, has_eos

    def _pad_sequences(
        self,
        sequences: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(len(sequence) for sequence in sequences)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        for sequence in sequences:
            pad_length = max_length - len(sequence)
            input_ids.append([pad_id] * pad_length + sequence)
            attention_masks.append([0] * pad_length + [1] * len(sequence))

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
                device=self.device,
            ),
        }

    def __call__(
        self,
        prompts=None,
        completions=None,
        completion_ids=None,
        **kwargs,
    ) -> list[float]:
        if prompts is None or completions is None:
            raise RuntimeError(
                "GRPOTrainer must pass both prompts and completions."
            )
        if completion_ids is None:
            raise RuntimeError(
                "This fair-comparison reward function requires completion_ids "
                "so EOS handling matches PPO and TRACE."
            )

        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        if not isinstance(completion_ids, (list, tuple)):
            completion_ids = [completion_ids]

        if len(prompts) != len(completion_ids):
            raise RuntimeError(
                "Prompt/completion batch-size mismatch: "
                f"{len(prompts)} prompts vs {len(completion_ids)} completions."
            )

        sequences: list[list[int]] = []
        eos_flags: list[bool] = []

        for prompt, generated_ids in zip(
            prompts,
            completion_ids,
            strict=True,
        ):
            prompt_ids = self.tokenizer(
                str(prompt),
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            response_ids, has_eos = self._clean_completion_ids(generated_ids)
            sequence = prompt_ids + response_ids

            if len(sequence) > self.reward_max_length:
                raise RuntimeError(
                    "Reward input exceeded --reward_max_length: "
                    f"{len(sequence)} > {self.reward_max_length}. "
                    "This should not occur when max_prompt_length=384 and "
                    "max_completion_length=96."
                )
            if not sequence:
                raise RuntimeError("Encountered an empty reward-model input.")

            sequences.append(sequence)
            eos_flags.append(has_eos)

        encoded = self._pad_sequences(sequences)
        with torch.inference_mode():
            scores = self.model(**encoded).logits.squeeze(-1).float()

        penalties = torch.tensor(
            [
                0.0 if has_eos else self.missing_eos_penalty
                for has_eos in eos_flags
            ],
            dtype=torch.float32,
            device=scores.device,
        )
        return (scores - penalties).detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = HfArgumentParser(
        (OASST2GRPOArguments, GRPOConfig, ModelConfig)
    )
    script_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    dataset_num_proc = getattr(script_args, "dataset_num_proc", None)

    if script_args.sft_model_path is None:
        script_args.sft_model_path = model_args.model_name_or_path
    if script_args.reward_model_path is None:
        raise ValueError("--reward_model_path is required.")

    # Prevent the current TRL default (DAPO) from silently replacing GRPO.
    if training_args.loss_type != "grpo":
        raise ValueError(
            "Use --loss_type grpo for the clean GRPO baseline. "
            f"Received: {training_args.loss_type!r}"
        )

    expected_reward_length = (
        script_args.max_prompt_length
        + training_args.max_completion_length
    )
    if script_args.reward_max_length < expected_reward_length:
        raise ValueError(
            "--reward_max_length must be at least "
            "--max_prompt_length + --max_completion_length "
            f"({expected_reward_length})."
        )

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
        script_args.sft_model_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    policy = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    grow_embeddings_if_required(policy, tokenizer)
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.eos_token_id

    reward_scorer = OASST2RewardScorer(
        reward_model_path=script_args.reward_model_path,
        tokenizer=tokenizer,
        reward_max_length=script_args.reward_max_length,
        missing_eos_penalty=script_args.missing_eos_penalty,
        dtype=dtype,
    )

    # A closure gives older/newer TRL versions a stable __name__ for logging.
    def oasst2_reward_model(
        prompts=None,
        completions=None,
        completion_ids=None,
        **kwargs,
    ):
        return reward_scorer(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids,
            **kwargs,
        )

    load_kwargs = {}
    if script_args.dataset_config:
        load_kwargs["name"] = script_args.dataset_config
    dataset = load_dataset(script_args.dataset_name, **load_kwargs)

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(
            dataset[script_args.dataset_train_split],
            tokenizer,
            script_args.max_prompt_length,
            dataset_num_proc,
        )
        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_dataset = prepare_dataset(
                dataset[script_args.dataset_test_split],
                tokenizer,
                script_args.max_prompt_length,
                dataset_num_proc,
            )

    if len(train_dataset) == 0:
        raise RuntimeError("No OASST2 prompts remain after filtering.")

    trainer = GRPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=oasst2_reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
