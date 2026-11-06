#!/usr/bin/env python3
"""

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)


def parse_model_specs(specs: List[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = os.path.basename(os.path.normpath(path))
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid model spec: {spec}")
        parsed.append((name, path))
    return parsed


def get_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    return mapping[dtype_arg]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_pad_token(tokenizer, fallback: str = "eos") -> None:
    if tokenizer.pad_token is not None:
        return
    if fallback == "eos" and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def batched(items: List[Tuple[int, str]], batch_size: int) -> Iterable[List[Tuple[int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def sample_std(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "median": float("nan"),
        }
    mean = float(statistics.fmean(values))
    std = float(sample_std(values))
    sem = float(std / math.sqrt(len(values))) if values else float("nan")
    ci = 1.96 * sem if not math.isnan(sem) else float("nan")
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "median": float(statistics.median(values)),
    }


def completion_lengths(completion_ids: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None) -> List[int]:
    lengths: List[int] = []
    ids_cpu = completion_ids.detach().cpu().tolist()
    for row in ids_cpu:
        length = 0
        for token_id in row:
            if pad_token_id is not None and token_id == pad_token_id:
                continue
            length += 1
            if eos_token_id is not None and token_id == eos_token_id:
                break
        lengths.append(length)
    return lengths


def has_eos(completion_ids: torch.Tensor, eos_token_id: int | None) -> List[bool]:
    if eos_token_id is None:
        return [False] * completion_ids.shape[0]
    return (completion_ids == eos_token_id).any(dim=1).detach().cpu().tolist()


def chosen_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def logprob_gap_on_generated_tokens(
    policy,
    ref_model,
    full_ids: torch.Tensor,
    pad_token_id: int | None,
    input_width: int,
) -> Tuple[List[float], List[float]]:
    attention_mask = torch.ones_like(full_ids)
    if pad_token_id is not None:
        attention_mask = (full_ids != pad_token_id).long()

    policy_lp = chosen_token_logprobs(policy, full_ids, attention_mask)
    ref_lp = chosen_token_logprobs(ref_model, full_ids, attention_mask)
    gap = policy_lp - ref_lp

    target_positions = torch.arange(1, full_ids.shape[1], device=full_ids.device).unsqueeze(0)
    gen_mask = target_positions >= input_width
    valid_mask = attention_mask[:, 1:].bool() & gen_mask
    if pad_token_id is not None:
        valid_mask = valid_mask & (full_ids[:, 1:] != pad_token_id)

    token_counts = valid_mask.sum(dim=1).clamp(min=1)
    gap_sum = (gap * valid_mask).sum(dim=1)
    gap_mean = gap_sum / token_counts
    return gap_sum.detach().cpu().float().tolist(), gap_mean.detach().cpu().float().tolist()


def load_eval_prompts(args, tokenizer) -> List[Tuple[int, str]]:
    dataset = load_dataset(args.dataset_name, name=args.dataset_config, split=args.dataset_split)
    prompts: List[Tuple[int, str]] = []
    for idx, row in enumerate(tqdm(dataset, desc="Filtering prompts")):
        if "prompt" not in row:
            raise KeyError("Dataset must contain a 'prompt' column for TL;DR evaluation.")
        prompt = row["prompt"]
        input_ids = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
        if len(input_ids) <= args.max_prompt_length:
            prompts.append((idx, prompt))
        if args.max_samples is not None and len(prompts) >= args.max_samples:
            break
    return prompts


def score_with_reward_model(reward_model, reward_tokenizer, texts: List[str], args, device) -> List[float]:
    encoded = reward_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.reward_max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = reward_model(**encoded)
    return outputs.logits.squeeze(-1).detach().cpu().float().tolist()


def evaluate_one_model(
    method: str,
    model_path: str,
    prompts: List[Tuple[int, str]],
    reward_model,
    reward_tokenizer,
    ref_model,
    args,
    device,
    dtype,
) -> Dict[str, object]:
    print(f"\n===== Evaluating {method}: {model_path} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer, fallback="eos")

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    policy.eval()
    policy.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id

    rows: List[Dict[str, object]] = []
    start_time = time.time()

    for batch in tqdm(list(batched(prompts, args.batch_size)), desc=f"Generating {method}"):
        indices = [x[0] for x in batch]
        batch_prompts = [x[1] for x in batch]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        input_width = encoded["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

        with torch.inference_mode():
            full_ids = policy.generate(**encoded, **generation_kwargs)
            completion_ids = full_ids[:, input_width:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            reward_scores = score_with_reward_model(
                reward_model,
                reward_tokenizer,
                [p + c for p, c in zip(batch_prompts, completions)],
                args,
                device,
            )
            kl_sum, kl_mean = logprob_gap_on_generated_tokens(
                policy,
                ref_model,
                full_ids,
                tokenizer.pad_token_id,
                input_width,
            )

        lengths = completion_lengths(completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        eos_flags = has_eos(completion_ids, tokenizer.eos_token_id)

        for idx, prompt, completion, score, length, eos_flag, gap_s, gap_m in zip(
            indices, batch_prompts, completions, reward_scores, lengths, eos_flags, kl_sum, kl_mean
        ):
            rows.append(
                {
                    "idx": idx,
                    "method": method,
                    "model_path": model_path,
                    "prompt": prompt,
                    "completion": completion,
                    "score": score,
                    "completion_tokens": length,
                    "has_eos": int(bool(eos_flag)),
                    "kl_proxy_sum": gap_s,
                    "kl_proxy_mean": gap_m,
                }
            )

    elapsed = time.time() - start_time
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"{method}_predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "method",
                "model_path",
                "prompt",
                "completion",
                "score",
                "completion_tokens",
                "has_eos",
                "kl_proxy_sum",
                "kl_proxy_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(r["score"]) for r in rows]
    lengths = [float(r["completion_tokens"]) for r in rows]
    eos_values = [float(r["has_eos"]) for r in rows]
    kl_sum_values = [float(r["kl_proxy_sum"]) for r in rows]
    kl_mean_values = [float(r["kl_proxy_mean"]) for r in rows]

    score_stats = summarize(scores)
    summary = {
        "method": method,
        "model_path": model_path,
        "n_eval": len(rows),
        "score_mean": score_stats["mean"],
        "score_std": score_stats["std"],
        "score_sem": score_stats["sem"],
        "score_ci95_low": score_stats["ci95_low"],
        "score_ci95_high": score_stats["ci95_high"],
        "score_median": score_stats["median"],
        "completion_tokens_mean": summarize(lengths)["mean"],
        "completion_tokens_std": summarize(lengths)["std"],
        "eos_rate": summarize(eos_values)["mean"],
        "kl_proxy_sum_mean": summarize(kl_sum_values)["mean"],
        "kl_proxy_mean": summarize(kl_mean_values)["mean"],
        "seconds": elapsed,
        "predictions_path": pred_path,
    }

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TL;DR checkpoints with a common score/KL/length schema.")
    parser.add_argument("--models", nargs="+", required=True, help="Model specs, e.g. ppo=./outputs/ppo_tldr trace=./outputs/trace_tldr")
    parser.add_argument("--dataset_name", default="trl-lib/tldr")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--ref_model_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/tldr_eval")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=53)
    parser.add_argument("--reward_max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick tests. Omit for full split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    dtype = get_dtype(args.dtype)
    model_specs = parse_model_specs(args.models)

    first_tokenizer = AutoTokenizer.from_pretrained(model_specs[0][1], padding_side="left", trust_remote_code=args.trust_remote_code)
    ensure_pad_token(first_tokenizer, fallback="eos")
    prompts = load_eval_prompts(args, first_tokenizer)
    print(f"Loaded {len(prompts)} evaluation prompts from {args.dataset_name}/{args.dataset_split}")

    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, padding_side="right", trust_remote_code=args.trust_remote_code)
    ensure_pad_token(reward_tokenizer, fallback="eos")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        torch_dtype=dtype,
        num_labels=1,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    reward_model.eval()
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    ref_model.eval()

    summaries: List[Dict[str, object]] = []
    with torch.inference_mode():
        for method, model_path in model_specs:
            summaries.append(
                evaluate_one_model(
                    method,
                    model_path,
                    prompts,
                    reward_model,
                    reward_tokenizer,
                    ref_model,
                    args,
                    device,
                    dtype,
                )
            )

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        row["method"]: {k: v for k, v in row.items() if k not in {"method", "model_path", "seconds", "predictions_path"}}
        for row in summaries
    }

    summary_path = os.path.join(args.output_dir, "summary.csv")
    rows = [{"model": name, **metrics} for name, metrics in result.items()]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n===== Summary =====")
    for name, row in result.items():
        print(
            f"{name}: score_mean={row['score_mean']:.6f}, "
            f"kl_proxy_mean={row['kl_proxy_mean']:.6f}, "
            f"eos_rate={row['eos_rate']:.4f}, "
            f"tokens={row['completion_tokens_mean']:.2f}"
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()