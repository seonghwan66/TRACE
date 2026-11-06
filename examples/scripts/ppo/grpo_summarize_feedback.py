# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
GRPO baseline on OpenAI summarize_from_feedback using an OpenAssistant DeBERTa
reward model.

This script mirrors the summarize_from_feedback PPO/TRACE scripts for prompt
construction and reward scoring, while keeping the GRPO-specific training path
minimal. It intentionally avoids extra algorithmic additions so that the run can
serve as a clean GRPO baseline.

Recommended smoke test:

export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2_cpu_offload_grpo.yaml \
    --num_processes 1 \
    examples/scripts/ppo/grpo_summarize_feedback.py \
    --dataset_name openai/summarize_from_feedback \
    --dataset_config comparisons \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir ./outputs/grpo_summarize_feedback_v2 \
    --learning_rate 3e-6 \
    --seed 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 625 \
    --num_generations 2 \
    --max_completion_length 64 \
    --beta 0.05 \
    --missing_eos_penalty 1.0 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path OpenAssistant/reward-model-deberta-v3-base \
    --eval_strategy no \
    --save_strategy no \
    --report_to wandb \
    --run_name grpo_summarize_feedback_v2

For roughly 10k prompt updates with the same effective prompt batch size as the
PPO/TRACE summarize_from_feedback runs, use --max_steps 625.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

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


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class OASummarizeGRPOScriptArguments(ScriptArguments):
    dataset_name: str = field(
        default="openai/summarize_from_feedback",
        metadata={"help": "Dataset name for Hugging Face datasets.load_dataset."},
    )
    dataset_config: str | None = field(
        default="comparisons",
        metadata={"help": "Dataset config. Use `comparisons` for train/validation preference data."},
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Training split name."})
    dataset_test_split: str = field(default="validation", metadata={"help": "Evaluation split name."})
    sft_model_path: str | None = field(
        default=None,
        metadata={"help": "Path or model ID of the SFT policy used as the initial GRPO policy."},
    )
    reward_model_path: str | None = field(
        default=None,
        metadata={"help": "Path or model ID of the OpenAssistant reward model."},
    )
    reward_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum tokenized length for the OpenAssistant reward model."},
    )
    reward_prompt_marker: str = field(
        default="TL;DR:",
        metadata={"help": "Marker used to identify the generated summary boundary."},
    )
    missing_eos_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty applied when a completion does not contain EOS."},
    )


def build_prompt(example: dict) -> str:
    """Build a TL;DR-style prompt from summarize_from_feedback rows."""
    info = example.get("info") or {}
    if not isinstance(info, dict):
        return ""

    post = str(info.get("post") or "").strip()
    if post:
        subreddit = str(info.get("subreddit") or "unknown").strip()
        subreddit = subreddit[2:] if subreddit.startswith("r/") else subreddit
        title = str(info.get("title") or "").strip()
        if title:
            return f"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
        return f"POST: {post}\n\nTL;DR:"

    article = str(info.get("article") or info.get("document") or info.get("text") or "").strip()
    if article:
        title = str(info.get("title") or "").strip()
        if title:
            return f"TITLE: {title}\n\nARTICLE: {article}\n\nTL;DR:"
        return f"ARTICLE: {article}\n\nTL;DR:"

    return ""


def add_prompt_column(dataset, dataset_num_proc: int | None):
    dataset = dataset.map(lambda example: {"prompt": build_prompt(example)}, num_proc=dataset_num_proc)
    return dataset.filter(lambda x: isinstance(x.get("prompt"), str) and len(x["prompt"].strip()) > 0, num_proc=dataset_num_proc)


def prepare_grpo_dataset(dataset, tokenizer, dataset_num_proc: int | None):
    """Keep only the prompt column required by GRPOTrainer, with length filtering."""

    def add_length(element):
        input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
        return {"lengths": len(input_ids)}

    dataset = dataset.map(add_length, num_proc=dataset_num_proc)
    dataset = dataset.filter(lambda x: x["lengths"] <= 512, num_proc=dataset_num_proc)
    columns_to_remove = [c for c in dataset.column_names if c not in {"prompt"}]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def _completion_to_text(completion: Any) -> str:
    """Support both plain-text and chat-style completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


class OpenAssistantGRPOReward:
    """Callable reward scorer for GRPO using an OpenAssistant sequence classifier.

    GRPOTrainer calls reward functions with generated completions. This callable
    scores each (prompt, completion) pair with a separate DeBERTa reward model,
    matching the pair-tokenization behavior used in the PPO/TRACE summarize
    scripts while avoiding any value-model or TRACE-specific logic.
    """

    def __init__(
        self,
        reward_model_path: str,
        policy_eos_token_id: int | None,
        missing_eos_penalty: float = 1.0,
        reward_max_length: int = 1024,
        prompt_marker: str = "TL;DR:",
        trust_remote_code: bool = False,
        dtype="auto",
    ):
        if reward_model_path is None:
            raise ValueError("`--reward_model_path` is required for GRPO reward scoring.")

        self.reward_max_length = reward_max_length
        self.prompt_marker = prompt_marker
        self.policy_eos_token_id = policy_eos_token_id
        self.missing_eos_penalty = float(missing_eos_penalty)
        self.device = PartialState().device

        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            padding_side="right",
            trust_remote_code=trust_remote_code,
        )
        if self.reward_tokenizer.pad_token is None:
            if self.reward_tokenizer.eos_token is not None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            else:
                self.reward_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        kwargs = {"trust_remote_code": trust_remote_code, "num_labels": 1}
        if dtype not in [None, "auto"]:
            kwargs["dtype"] = dtype
        self.rank_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, **kwargs)
        if len(self.reward_tokenizer) > self.rank_model.get_input_embeddings().num_embeddings:
            self.rank_model.resize_token_embeddings(len(self.reward_tokenizer))
        self.rank_model.to(self.device)
        self.rank_model.eval()
        for param in self.rank_model.parameters():
            param.requires_grad = False

    def _normalize_prompt(self, prompt: Any) -> str:
        prompt = str(prompt or "").strip()
        if self.prompt_marker and self.prompt_marker not in prompt:
            prompt = f"{prompt}\n\n{self.prompt_marker}"
        return prompt

    def _normalize_answer(self, completion: Any) -> str:
        answer = _completion_to_text(completion).strip()
        if self.prompt_marker and self.prompt_marker in answer:
            _, answer = answer.rsplit(self.prompt_marker, 1)
            answer = answer.strip()
        return answer

    def __call__(
        self,
        prompts=None,
        completions=None,
        completion_ids=None,
        **kwargs,
    ) -> list[float]:
        # Some TRL versions pass `prompt` through kwargs; keep this fallback minimal.
        if completions is None:
            completions = prompts
            prompts = kwargs.get("prompts", kwargs.get("prompt", None))

        if completion_ids is None:
            completion_ids = kwargs.get("completion_ids", None)

        if completions is None:
            return []
        if not isinstance(completions, (list, tuple)):
            completions = [completions]

        if prompts is None:
            prompts = [""] * len(completions)
        elif not isinstance(prompts, (list, tuple)):
            prompts = [prompts] * len(completions)

        prompts = [self._normalize_prompt(p) for p in prompts]
        answers = [self._normalize_answer(c) for c in completions]

        encoded = self.reward_tokenizer(
            prompts,
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.reward_max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            rewards = self.rank_model(**encoded).logits.squeeze(-1).float().detach().cpu().tolist()

        # Match PPO/TRACE: subtract the same penalty when the generated token
        # sequence does not contain the policy EOS token.
        if completion_ids is None:
            raise RuntimeError(
                "GRPOTrainer did not provide `completion_ids`; EOS penalty cannot be applied fairly."
            )
        if torch.is_tensor(completion_ids):
            completion_ids = completion_ids.detach().cpu().tolist()

        if len(completion_ids) != len(rewards):
            raise RuntimeError(
                f"completion_ids/reward size mismatch: {len(completion_ids)} vs {len(rewards)}"
            )

        for i, token_ids in enumerate(completion_ids):
            if torch.is_tensor(token_ids):
                token_ids = token_ids.detach().cpu().tolist()
            has_eos = (
                self.policy_eos_token_id is not None
                and self.policy_eos_token_id in token_ids
            )
            if not has_eos:
                rewards[i] -= self.missing_eos_penalty

        return rewards


if __name__ == "__main__":
    parser = HfArgumentParser((OASummarizeGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

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
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if len(tokenizer) > policy.get_input_embeddings().num_embeddings:
        policy.resize_token_embeddings(len(tokenizer))

    reward_scorer = OpenAssistantGRPOReward(
        reward_model_path=script_args.reward_model_path,
        policy_eos_token_id=tokenizer.eos_token_id,
        missing_eos_penalty=script_args.missing_eos_penalty,
        reward_max_length=script_args.reward_max_length,
        prompt_marker=script_args.reward_prompt_marker,
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
    )

    def openassistant_grpo_reward(
        prompts,
        completions,
        completion_ids=None,
        **kwargs,
    ):
        return reward_scorer(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids,
            **kwargs,
        )

    peft_config = get_peft_config(model_args)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, trust_remote_code=True)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    dataset_num_proc = getattr(script_args, "dataset_num_proc", None)

    with PartialState().local_main_process_first():
        train_dataset = add_prompt_column(train_dataset, dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = add_prompt_column(eval_dataset, dataset_num_proc)

        train_dataset = prepare_grpo_dataset(train_dataset, tokenizer, dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = prepare_grpo_dataset(eval_dataset, tokenizer, dataset_num_proc)

    assert len(train_dataset) > 0, "No training examples remain after prompt construction and length filtering."
    assert "prompt" in train_dataset.column_names, "GRPOTrainer requires a `prompt` column."

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=[openassistant_grpo_reward],
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
