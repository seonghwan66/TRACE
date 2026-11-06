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
PPO training on OpenAI summarize_from_feedback using an OpenAssistant DeBERTa
reward model.

This script keeps the structure of examples/scripts/ppo/ppo_tldr.py, but uses
separate tokenizers for:
  1. the policy/value/reference models, and
  2. OpenAssistant/reward-model-deberta-v3-large-v2.

Recommended run:

export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2_cpu_offload.yaml \
    --num_processes 1 \
    examples/scripts/ppo/trace_summarize_feedback.py \
    --dataset_name openai/summarize_from_feedback \
    --dataset_config comparisons \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir ./outputs/trace_summarize_feedback_test \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 32 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --value_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --reward_model_path OpenAssistant/reward-model-deberta-v3-base \
    --local_rollout_forward_batch_size 1 \
    --response_length 64 \
    --stop_token eos \
    --save_strategy no \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    --run_name trace_summarize_feedback_test
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from types import SimpleNamespace

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
from trl.experimental.ppo import PPOConfig
from trl.experimental.ppo.trace_trainer_wo_value_weighted_allocation import TRACETrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class OASummarizeScriptArguments(ScriptArguments):
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
    value_model_path: str | None = field(
        default=None,
        metadata={"help": "Policy-tokenizer-compatible value model path. Defaults to --sft_model_path."},
    )
    reward_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum tokenized length for the OpenAssistant reward model."},
    )
    reward_prompt_marker: str = field(
        default="TL;DR:",
        metadata={"help": "Marker used to split decoded text into prompt and generated summary."},
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


def prepare_dataset(dataset, tokenizer, dataset_num_proc: int | None):
    """Pre-tokenize prompts before training; only collate during training."""

    def tokenize(element):
        input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    return dataset.map(tokenize, remove_columns=dataset.column_names, num_proc=dataset_num_proc)


class OpenAssistantRewardBackbone(torch.nn.Module):
    """Retokenize policy outputs with an OpenAssistant reward-model tokenizer.

    PPOTrainer passes policy-tokenized `input_ids` to `reward_model`. The
    OpenAssistant reward model is DeBERTa-based, so those token ids cannot be
    used directly. This backbone decodes the policy sequence, splits it into
    (prompt, generated_summary), retokenizes the pair with the reward tokenizer,
    and returns the scalar reward as a one-dimensional hidden state compatible
    with the PPO reward-model path used by TRL.
    """

    def __init__(
        self,
        policy_tokenizer,
        reward_model_path: str,
        reward_max_length: int = 1024,
        prompt_marker: str = "TL;DR:",
        trust_remote_code: bool = False,
        dtype="auto",
    ):
        super().__init__()
        self.policy_tokenizer = policy_tokenizer
        self.reward_max_length = reward_max_length
        self.prompt_marker = prompt_marker

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
        self.rank_model.eval()
        for param in self.rank_model.parameters():
            param.requires_grad = False

    def _split_prompt_answer(self, text: str) -> tuple[str, str]:
        text = str(text or "").strip()
        if self.prompt_marker and self.prompt_marker in text:
            prompt, answer = text.rsplit(self.prompt_marker, 1)
            prompt = f"{prompt.strip()}\n\n{self.prompt_marker}"
            answer = answer.strip()
            return prompt, answer
        return text, ""

    def _score_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        pairs = [self._split_prompt_answer(text) for text in texts]
        prompts = [p for p, _ in pairs]
        answers = [a for _, a in pairs]

        encoded = self.reward_tokenizer(
            prompts,
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.reward_max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            scores = self.rank_model(**encoded).logits.squeeze(-1).float()
        return scores

    def forward(self, input_ids, attention_mask=None, **kwargs):
        texts = self.policy_tokenizer.batch_decode(
            input_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        reward_device = next(self.rank_model.parameters()).device
        rewards = self._score_texts(texts, reward_device).to(input_ids.device, dtype=torch.float32)

        hidden = torch.zeros((*input_ids.shape, 1), device=input_ids.device, dtype=torch.float32)
        hidden[:, :, 0] = rewards.unsqueeze(1)
        return SimpleNamespace(hidden_states=(hidden,), logits=rewards)


class OpenAssistantRewardModel(torch.nn.Module):
    """PPOTrainer-compatible wrapper for a separate-tokenizer reward model."""

    base_model_prefix = "backbone"

    def __init__(self, policy_tokenizer, reward_model_path: str, reward_max_length: int, prompt_marker: str, trust_remote_code: bool, dtype):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.backbone = OpenAssistantRewardBackbone(
            policy_tokenizer=policy_tokenizer,
            reward_model_path=reward_model_path,
            reward_max_length=reward_max_length,
            prompt_marker=prompt_marker,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        self.score = torch.nn.Identity()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


if __name__ == "__main__":
    parser = HfArgumentParser((OASummarizeScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

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
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    value_model_path = script_args.value_model_path or training_args.sft_model_path
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )

    reward_model = OpenAssistantRewardModel(
        policy_tokenizer=tokenizer,
        reward_model_path=training_args.reward_model_path,
        reward_max_length=script_args.reward_max_length,
        prompt_marker=script_args.reward_prompt_marker,
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, trust_remote_code=True)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    with PartialState().local_main_process_first():
        train_dataset = add_prompt_column(train_dataset, training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = add_prompt_column(eval_dataset, training_args.dataset_num_proc)

        train_dataset = prepare_dataset(train_dataset, tokenizer, training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer, training_args.dataset_num_proc)

        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert len(train_dataset) > 0, "No training examples remain after prompt construction and length filtering."
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    trainer = TRACETrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save final policy model only after training.
    accelerator = trainer.accelerator
    accelerator.wait_for_everyone()

    model_to_save = getattr(trainer, "model_wrapped", trainer.model)
    full_state_dict = accelerator.get_state_dict(model_to_save)

    policy_state_dict = {
        k.replace("policy.", "", 1): v
        for k, v in full_state_dict.items()
        if k.startswith("policy.")
    }

    policy_to_save = accelerator.unwrap_model(trainer.model).policy

    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        policy_to_save.save_pretrained(
            training_args.output_dir,
            state_dict=policy_state_dict,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
