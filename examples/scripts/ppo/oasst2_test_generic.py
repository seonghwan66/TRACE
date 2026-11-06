# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generic OASST2 checkpoint evaluation.

Any number of trained checkpoints can be passed with --models NAME=PATH.
No model-to-model comparison is performed.

Example:

export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python examples/scripts/ppo/oasst2_test_generic.py \
  --models \
    PPO_1=./outputs/PPO/ppo_oasst2_1 \
    PPO_2=./outputs/PPO/ppo_oasst2_2 \
    PPO_4=./outputs/PPO/ppo_oasst2_4 \
    PPO_5=./outputs/PPO/ppo_oasst2_5 \
    TRACE_1=./outputs/TRACE/trace_oasst2_1 \
    TRACE_2=./outputs/TRACE/trace_oasst2_2 \
    TRACE_3=./outputs/TRACE/trace_oasst2_3 \
    TRACE_4=./outputs/TRACE/trace_oasst2_4 \
    GRPO_1=./outputs/GRPO/grpo_oasst2_1 \
    GRPO_2=./outputs/GRPO/grpo_oasst2_2 \
    GRPO_3=./outputs/GRPO/grpo_oasst2_3 \
    GRPO_4=./outputs/GRPO/grpo_oasst2_4 \
    GRPO_5=./outputs/GRPO/grpo_oasst2_5 \
  --reference_model_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
  --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
  --dataset_name OpenAssistant/oasst2 \
  --dataset_split validation \
  --max_prompt_length 384 \
  --max_new_tokens 96 \
  --reward_max_length 480 \
  --missing_eos_penalty 1.0 \
  --max_samples 500 \
  --samples_per_prompt 1 \
  --batch_size 1 \
  --score_batch_size 1 \
  --seed 42 \
  --do_sample \
  --temperature 1.0 \
  --output_dir ./outputs/eval/OASST2/TEST


export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python examples/scripts/ppo/oasst2_test_generic.py \
  --models \
    TRACE_WO_W_1=./outputs/TRACE/wo_weighted/trace_oasst2_wo_value_weighted_allocation_1 \
    TRACE_WO_W_2=./outputs/TRACE/wo_weighted/trace_oasst2_wo_value_weighted_allocation_2 \
    TRACE_WO_W_3=./outputs/TRACE/wo_weighted/trace_oasst2_wo_value_weighted_allocation_3 \
  --reference_model_path andreaskoepf/pythia-1.4b-gpt4all-pretrain \
  --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
  --dataset_name OpenAssistant/oasst2 \
  --dataset_split validation \
  --max_prompt_length 384 \
  --max_new_tokens 96 \
  --reward_max_length 480 \
  --missing_eos_penalty 1.0 \
  --max_samples 500 \
  --samples_per_prompt 1 \
  --batch_size 1 \
  --score_batch_size 1 \
  --seed 42 \
  --do_sample \
  --temperature 1.0 \
  --output_dir ./outputs/eval/OASST2/TEST
"""



from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)
from transformers.utils import ModelOutput


PROMPTER_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endoftext|>"


# OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 architecture.
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
        hidden = outputs[0]

        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden[:, -1]
            else:
                last = attention_mask.long().cumsum(1).argmax(1)
                pooled = hidden[
                    torch.arange(hidden.size(0), device=hidden.device), last
                ]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)
        if not return_dict:
            return (logits,) + outputs[1:]
        return GPTNeoXRewardModelOutput(logits=logits)


AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(
    GPTNeoXRewardModelConfig, GPTNeoXRewardModel
)


def parse_model_specs(specs):
    models = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid model specification: {spec}. Use NAME=PATH.")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid model specification: {spec}")
        models[name] = path
    return models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", required=True,
        help="Models in NAME=PATH format. Any number of checkpoints can be evaluated.",
    )
    p.add_argument(
        "--reference_model_path",
        default="andreaskoepf/pythia-1.4b-gpt4all-pretrain",
    )
    p.add_argument(
        "--reward_model_path",
        default="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    )
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--dataset_name", default="OpenAssistant/oasst2")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--dataset_split", default="validation")
    p.add_argument("--dataset_num_proc", type=int, default=None)
    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--reward_max_length", type=int, default=480)
    p.add_argument("--missing_eos_penalty", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--samples_per_prompt", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--score_batch_size", type=int, default=1)
    p.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--output_dir", default="./outputs/oasst2_eval")
    args = p.parse_args()

    positive = [
        args.max_prompt_length,
        args.max_new_tokens,
        args.samples_per_prompt,
        args.batch_size,
        args.score_batch_size,
    ]
    if any(value <= 0 for value in positive):
        p.error("Length, sample, and batch arguments must be positive.")
    required = args.max_prompt_length + args.max_new_tokens
    if args.reward_max_length < required:
        p.error(f"--reward_max_length must be at least {required}.")
    if args.do_sample and args.temperature <= 0:
        p.error("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        p.error("--top_p must be in (0, 1].")
    if args.top_k < 0:
        p.error("--top_k must be non-negative.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value != "auto":
        return getattr(torch, value)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def release(model) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def grow_embeddings(model, tokenizer) -> None:
    embedding = model.get_input_embeddings()
    if embedding is not None and len(tokenizer) > embedding.num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


def format_prompt(text: str) -> str:
    return f"{PROMPTER_TOKEN}{text.strip()}{END_TOKEN}{ASSISTANT_TOKEN}"


def valid_prompt(example: dict[str, Any]) -> bool:
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
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    dataset = dataset.filter(
        valid_prompt,
        num_proc=args.dataset_num_proc,
        desc="Filtering OASST2",
    )

    def convert(example):
        text = str(example["text"]).strip()
        prompt = format_prompt(text)
        ids = tokenizer(
            prompt, add_special_tokens=False, padding=False
        )["input_ids"]
        return {
            "sample_id": str(example.get("message_id") or ""),
            "prompt_text": text,
            "prompt": prompt,
            "prompt_ids": ids,
            "prompt_tokens": len(ids),
        }

    dataset = dataset.map(
        convert,
        remove_columns=dataset.column_names,
        num_proc=args.dataset_num_proc,
        desc="Formatting prompts",
    )
    dataset = dataset.filter(
        lambda x: 0 < x["prompt_tokens"] <= args.max_prompt_length,
        num_proc=args.dataset_num_proc,
        desc="Filtering prompt length",
    )
    if len(dataset) == 0:
        raise RuntimeError("No validation prompts remain after filtering.")

    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output = []
    for index, item in enumerate(dataset):
        item = dict(item)
        item["sample_index"] = index
        item["prompt_ids"] = [int(x) for x in item["prompt_ids"]]
        item["prompt_tokens"] = int(item["prompt_tokens"])
        output.append(item)
    return output


def load_policy(path, tokenizer, device, dtype, trust_remote_code):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    grow_embeddings(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.to(device).eval()
    return model


def clean_completion(ids, tokenizer) -> tuple[list[int], bool]:
    output, has_eos = [], False
    for token_id in ids:
        token_id = int(token_id)
        if token_id == tokenizer.pad_token_id:
            continue
        output.append(token_id)
        if token_id == tokenizer.eos_token_id:
            has_eos = True
            break
    return output, has_eos


def generate(
    model_name: str,
    model,
    prompts: list[dict[str, Any]],
    tokenizer,
    device,
    args,
) -> list[dict[str, Any]]:
    set_seed(args.seed)
    records = []
    iterator = batches(prompts, args.batch_size)

    for prompt_batch in tqdm(
        iterator,
        total=math.ceil(len(prompts) / args.batch_size),
        desc=f"Generating {model_name}",
    ):
        encoded = tokenizer(
            [x["prompt"] for x in prompt_batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].size(1)

        kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "num_return_sequences": args.samples_per_prompt,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if args.do_sample:
            kwargs.update(temperature=args.temperature, top_p=args.top_p)
            if args.top_k > 0:
                kwargs["top_k"] = args.top_k

        with torch.inference_mode():
            outputs = model.generate(**encoded, **kwargs)
        completions = outputs[:, prompt_width:].detach().cpu().tolist()

        for output_index, ids in enumerate(completions):
            prompt_pos = output_index // args.samples_per_prompt
            generation_index = output_index % args.samples_per_prompt
            source = prompt_batch[prompt_pos]
            completion_ids, has_eos = clean_completion(ids, tokenizer)
            records.append(
                {
                    "model": model_name,
                    "sample_index": source["sample_index"],
                    "sample_id": source["sample_id"],
                    "generation_index": generation_index,
                    "prompt_text": source["prompt_text"],
                    "prompt": source["prompt"],
                    "prompt_ids": source["prompt_ids"],
                    "completion_ids": completion_ids,
                    "response": tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip(),
                    "prompt_tokens": source["prompt_tokens"],
                    "response_tokens": len(completion_ids),
                    "has_eos": has_eos,
                    "truncated": (
                        not has_eos
                        and len(completion_ids) >= args.max_new_tokens
                    ),
                }
            )
    return records


def left_pad(sequences, pad_id, device):
    width = max(len(x) for x in sequences)
    ids, masks, pads = [], [], []
    for sequence in sequences:
        count = width - len(sequence)
        pads.append(count)
        ids.append([pad_id] * count + sequence)
        masks.append([0] * count + [1] * len(sequence))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        pads,
    )


def add_logprobs(records, model, tokenizer, device, batch_size, prefix):
    for record_batch in tqdm(
        batches(records, batch_size),
        total=math.ceil(len(records) / batch_size),
        desc=f"{prefix} log-probs",
    ):
        sequences = [
            x["prompt_ids"] + x["completion_ids"] for x in record_batch
        ]
        input_ids, attention_mask, pads = left_pad(
            sequences, tokenizer.pad_token_id, device
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1].float()

        targets = input_ids[:, 1:]
        logps = F.log_softmax(logits, -1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)

        for row, record in enumerate(record_batch):
            length = len(record["completion_ids"])
            start = pads[row] + len(record["prompt_ids"]) - 1
            selected = logps[row, start : start + length]
            if selected.numel() != length:
                raise RuntimeError("Response log-probability mask mismatch.")
            total = float(selected.sum().item())
            record[f"{prefix}_logprob_sum"] = total
            record[f"{prefix}_logprob_per_token"] = total / max(length, 1)


def load_reward_model(path, tokenizer, device, dtype):
    model = AutoModelForSequenceClassification.from_pretrained(
        path, dtype=dtype
    )
    grow_embeddings(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def add_rewards(records, model, tokenizer, device, args):
    for record_batch in tqdm(
        batches(records, args.score_batch_size),
        total=math.ceil(len(records) / args.score_batch_size),
        desc="Reward scoring",
    ):
        sequences = [
            x["prompt_ids"] + x["completion_ids"] for x in record_batch
        ]
        if max(map(len, sequences)) > args.reward_max_length:
            raise RuntimeError("A reward-model input exceeded reward_max_length.")
        input_ids, attention_mask, _ = left_pad(
            sequences, tokenizer.pad_token_id, device
        )
        with torch.inference_mode():
            scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits.squeeze(-1).float().cpu().tolist()

        for record, raw in zip(record_batch, scores, strict=True):
            penalty = 0.0 if record["has_eos"] else args.missing_eos_penalty
            record["raw_reward"] = float(raw)
            record["eos_penalty"] = float(penalty)
            record["final_reward"] = float(raw - penalty)


def summarize(records, model_names):
    summaries = {}
    for name in model_names:
        rows = [x for x in records if x["model"] == name]

        def stats(key):
            values = np.asarray([x[key] for x in rows], dtype=np.float64)
            return float(values.mean()), float(values.std(ddof=0))

        raw_mean, raw_std = stats("raw_reward")
        final_mean, final_std = stats("final_reward")
        len_mean, len_std = stats("response_tokens")
        kl_mean, kl_std = stats("sampled_kl_per_token")
        summaries[name] = {
            "num_generations": len(rows),
            "raw_reward_mean": raw_mean,
            "raw_reward_std": raw_std,
            "final_reward_mean": final_mean,
            "final_reward_std": final_std,
            "eos_rate": float(np.mean([x["has_eos"] for x in rows])),
            "truncated_ratio": float(np.mean([x["truncated"] for x in rows])),
            "response_tokens_mean": len_mean,
            "response_tokens_std": len_std,
            "sampled_kl_per_token_mean": kl_mean,
            "sampled_kl_per_token_std": kl_std,
        }
    return summaries


def write_outputs(output_dir, records, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "model",
        "sample_index",
        "sample_id",
        "generation_index",
        "prompt_text",
        "response",
        "prompt_tokens",
        "response_tokens",
        "has_eos",
        "truncated",
        "raw_reward",
        "eos_penalty",
        "final_reward",
        "policy_logprob_sum",
        "reference_logprob_sum",
        "sampled_kl_sum",
        "sampled_kl_per_token",
    ]
    with (output_dir / "oasst2_eval_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in columns})

    (output_dir / "oasst2_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_columns = ["model"] + list(
        next(iter(summary["models"].values())).keys()
    )
    with (output_dir / "oasst2_eval_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=model_columns)
        writer.writeheader()
        for name, values in summary["models"].items():
            writer.writerow({"model": name, **values})


def print_summary(summary):
    print("\nModel   Final reward   Raw reward   EOS rate   Truncated   Tokens   KL/token")
    for name, row in summary["models"].items():
        print(
            f"{name:<7} {row['final_reward_mean']:>12.4f} "
            f"{row['raw_reward_mean']:>12.4f} "
            f"{row['eos_rate']:>10.3f} "
            f"{row['truncated_ratio']:>11.3f} "
            f"{row['response_tokens_mean']:>8.2f} "
            f"{row['sampled_kl_per_token_mean']:>10.6f}"
        )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer_path = args.tokenizer_path or args.reference_model_path
    model_paths = parse_model_specs(args.models)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")
    missing = [
        token
        for token in (PROMPTER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)
        if token not in tokenizer.get_vocab()
    ]
    if missing:
        raise RuntimeError(f"Tokenizer is missing special tokens: {missing}")

    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset_name, **load_kwargs)
    if args.dataset_split not in dataset:
        raise KeyError(f"Unknown split: {args.dataset_split}")

    prompts = prepare_dataset(dataset[args.dataset_split], tokenizer, args)
    print(f"Selected {len(prompts)} prompts; device={device}; dtype={dtype}.")

    records = []
    for name, path in model_paths.items():
        print(f"\nLoading {name}: {path}")
        model = load_policy(
            path, tokenizer, device, dtype, args.trust_remote_code
        )
        rows = generate(name, model, prompts, tokenizer, device, args)
        add_logprobs(
            rows,
            model,
            tokenizer,
            device,
            args.score_batch_size,
            "policy",
        )
        records.extend(rows)
        release(model)
        model = None

    print(f"\nLoading reference: {args.reference_model_path}")
    reference = load_policy(
        args.reference_model_path,
        tokenizer,
        device,
        dtype,
        args.trust_remote_code,
    )
    add_logprobs(
        records,
        reference,
        tokenizer,
        device,
        args.score_batch_size,
        "reference",
    )
    release(reference)
    reference = None

    for record in records:
        record["sampled_kl_sum"] = (
            record["policy_logprob_sum"] - record["reference_logprob_sum"]
        )
        record["sampled_kl_per_token"] = (
            record["sampled_kl_sum"] / max(record["response_tokens"], 1)
        )

    print(f"\nLoading reward model: {args.reward_model_path}")
    reward_model = load_reward_model(
        args.reward_model_path, tokenizer, device, dtype
    )
    add_rewards(records, reward_model, tokenizer, device, args)
    release(reward_model)
    reward_model = None

    model_names = list(model_paths)
    models = summarize(records, model_names)
    summary = {"models": models}
    write_outputs(args.output_dir, records, summary)
    print_summary(summary)
    print(f"\nSaved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()