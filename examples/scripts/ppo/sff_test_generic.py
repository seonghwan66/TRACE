#!/usr/bin/env python3
"""

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

# idx, prompt, prompt_hash, prompt_category
PromptItem = Tuple[int, str, str, str]

TRACE_FAVORABLE_DEFAULT_CATEGORIES = [
    "relationship_emotional_dilemma",
    "pet_dog_behavior_or_care",
    "advice_opinion_askreddit",
    "personal_incident_tifu_story",
    "practical_self_improvement",
]


def parse_model_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
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


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    return mapping[dtype_arg]


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def batched(items: List[PromptItem], batch_size: int) -> Iterable[List[PromptItem]]:
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
    sem = float(std / math.sqrt(len(values)))
    ci = 1.96 * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
        "median": float(statistics.median(values)),
    }


def build_prompt(example: dict) -> str:
    """Build the same TL;DR-style prompt used by the training scripts."""
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


def normalize_prompt(prompt: str) -> str:
    """Normalize whitespace for prompt deduplication only."""
    return re.sub(r"\s+", " ", str(prompt or "").strip())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def parse_prompt_fields(prompt: str) -> Dict[str, str]:
    """Extract subreddit, title, and body from the generated TL;DR prompt."""
    text = str(prompt or "")
    subreddit_match = re.search(r"SUBREDDIT:\s*r/([^\n]+)", text)
    title_match = re.search(r"TITLE:\s*(.*?)\n\n", text, re.S)
    post_match = re.search(r"POST:\s*(.*?)\n\nTL;DR:", text, re.S)
    article_match = re.search(r"ARTICLE:\s*(.*?)\n\nTL;DR:", text, re.S)
    return {
        "subreddit": subreddit_match.group(1).strip() if subreddit_match else "",
        "title": title_match.group(1).strip() if title_match else "",
        "body": (post_match.group(1).strip() if post_match else (article_match.group(1).strip() if article_match else "")),
    }


def classify_prompt_type(prompt: str) -> str:
    """Heuristic categories derived from the TRACE-vs-PPO error analysis.

    These labels are only for evaluation-set construction and reporting; they do
    not affect generation or reward scoring.
    """
    fields = parse_prompt_fields(prompt)
    sub = fields["subreddit"].lower()
    title = fields["title"].lower()
    body = fields["body"].lower()
    text = f"{title} {body}"

    if sub in {"relationships", "relationship_advice", "breakups", "dating_advice"}:
        return "relationship_emotional_dilemma"
    if sub in {"dogs", "dogtraining", "askvet"}:
        return "pet_dog_behavior_or_care"
    if sub == "tifu":
        return "personal_incident_tifu_story"
    if sub == "askreddit":
        return "advice_opinion_askreddit"
    if sub in {"running", "loseit", "fitness"}:
        return "practical_self_improvement"
    if sub == "personalfinance":
        return "personalfinance"
    if any(k in text for k in ["boyfriend", "girlfriend", "ex", "relationship", "dating", "family"]):
        return "relationship_emotional_dilemma"
    return sub or "other"


def select_prompt_subset(prompts: List[PromptItem], args) -> List[PromptItem]:
    rng = random.Random(args.seed)

    if args.focus_trace_favorable_types:
        focus_categories = set(args.focus_categories or TRACE_FAVORABLE_DEFAULT_CATEGORIES)
        focus = [p for p in prompts if p[3] in focus_categories]
        other = [p for p in prompts if p[3] not in focus_categories]
        rng.shuffle(focus)
        rng.shuffle(other)

        if args.max_samples is None:
            selected = focus + other
        else:
            target_focus = int(round(args.max_samples * args.focus_ratio))
            target_focus = min(target_focus, len(focus))
            target_other = args.max_samples - target_focus
            if len(other) < target_other:
                target_focus = min(len(focus), target_focus + (target_other - len(other)))
                target_other = len(other)
            selected = focus[:target_focus] + other[:target_other]
        rng.shuffle(selected)
        return selected

    if args.shuffle_prompts:
        prompts = list(prompts)
        rng.shuffle(prompts)
    if args.max_samples is not None:
        prompts = prompts[: args.max_samples]
    return prompts


def load_eval_prompts(args, tokenizer) -> List[PromptItem]:
    dataset = load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        split=args.dataset_split,
        trust_remote_code=True,
    )

    prompts: List[PromptItem] = []
    seen_hashes = set()
    num_valid = 0
    num_duplicates = 0
    iterator = tqdm(dataset, desc="Building/filtering unique prompts")

    for idx, row in enumerate(iterator):
        prompt = row["prompt"] if "prompt" in row else build_prompt(row)
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        input_ids = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
        if len(input_ids) > args.max_prompt_length:
            continue

        num_valid += 1
        key = prompt_hash(prompt)

        if args.deduplicate_prompts:
            if key in seen_hashes:
                num_duplicates += 1
                continue
            seen_hashes.add(key)

        category = classify_prompt_type(prompt)
        prompts.append((idx, prompt, key, category))

        if (
            args.max_samples is not None
            and len(prompts) >= args.max_samples
            and not args.shuffle_prompts
            and not args.focus_trace_favorable_types
        ):
            break

    selected = select_prompt_subset(prompts, args)
    counts = Counter(p[3] for p in selected)
    print(
        f"Prompt filtering: valid_rows={num_valid}, "
        f"duplicates_skipped={num_duplicates}, selected_prompts={len(selected)}, "
        f"deduplicate={args.deduplicate_prompts}, shuffle={args.shuffle_prompts}, "
        f"focus_trace_favorable_types={args.focus_trace_favorable_types}"
    )
    print("Selected prompt categories:", dict(sorted(counts.items())))
    return selected


def save_eval_prompts(prompts: List[PromptItem], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_prompts_unique.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_order", "idx", "prompt_hash", "prompt_category", "prompt"])
        writer.writeheader()
        for order, (idx, prompt, key, category) in enumerate(prompts):
            writer.writerow(
                {
                    "eval_order": order,
                    "idx": idx,
                    "prompt_hash": key,
                    "prompt_category": category,
                    "prompt": prompt,
                }
            )
    return path


class OpenAssistantPairReward:
    """Score (prompt, completion) pairs with the OpenAssistant reward model."""

    def __init__(self, model_path: str, max_length: int, device: torch.device, dtype: torch.dtype, trust_remote_code: bool):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            trust_remote_code=trust_remote_code,
        )
        ensure_pad_token(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            dtype=dtype,
            num_labels=1,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.inference_mode()
    def score(self, prompts: List[str], completions: List[str]) -> List[float]:
        encoded = self.tokenizer(
            prompts,
            completions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        return outputs.logits.squeeze(-1).detach().cpu().float().tolist()


def completion_lengths(completion_ids: torch.Tensor, pad_token_id: Optional[int], eos_token_id: Optional[int]) -> List[int]:
    lengths: List[int] = []
    for row in completion_ids.detach().cpu().tolist():
        length = 0
        for token_id in row:
            if pad_token_id is not None and token_id == pad_token_id:
                continue
            length += 1
            if eos_token_id is not None and token_id == eos_token_id:
                break
        lengths.append(length)
    return lengths


def has_eos(completion_ids: torch.Tensor, eos_token_id: Optional[int]) -> List[bool]:
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
    pad_token_id: Optional[int],
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


def load_tokenizer_for_model(model_path: str, args):
    tokenizer_path = args.tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left", trust_remote_code=args.trust_remote_code)
    ensure_pad_token(tokenizer)
    return tokenizer


def evaluate_one_model(
    method: str,
    model_path: str,
    prompts: List[PromptItem],
    reward_scorer: OpenAssistantPairReward,
    ref_model,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    print(f"\n===== Evaluating {method}: {model_path} =====")
    tokenizer = load_tokenizer_for_model(model_path, args)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    if len(tokenizer) > policy.get_input_embeddings().num_embeddings:
        policy.resize_token_embeddings(len(tokenizer))
    policy.eval()
    policy.config.pad_token_id = tokenizer.pad_token_id
    if ref_model is not None:
        ref_model.config.pad_token_id = tokenizer.pad_token_id

    rows: List[Dict[str, object]] = []
    start_time = time.time()

    for batch in tqdm(list(batched(prompts, args.batch_size)), desc=f"Generating {method}"):
        indices = [x[0] for x in batch]
        batch_prompts = [x[1] for x in batch]
        prompt_hashes = [x[2] for x in batch]
        prompt_categories = [x[3] for x in batch]
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
            raw_scores = reward_scorer.score(batch_prompts, completions)
            if ref_model is not None:
                kl_sum, kl_mean = logprob_gap_on_generated_tokens(
                    policy=policy,
                    ref_model=ref_model,
                    full_ids=full_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    input_width=input_width,
                )
            else:
                kl_sum = [float("nan")] * len(batch_prompts)
                kl_mean = [float("nan")] * len(batch_prompts)

        lengths = completion_lengths(completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        eos_flags = has_eos(completion_ids, tokenizer.eos_token_id)
        adjusted_scores = [s - (args.missing_eos_penalty if not eos else 0.0) for s, eos in zip(raw_scores, eos_flags)]

        for idx, prompt, key, category, completion, raw, score, length, eos_flag, gap_s, gap_m in zip(
            indices,
            batch_prompts,
            prompt_hashes,
            prompt_categories,
            completions,
            raw_scores,
            adjusted_scores,
            lengths,
            eos_flags,
            kl_sum,
            kl_mean,
        ):
            rows.append(
                {
                    "idx": idx,
                    "prompt_hash": key,
                    "prompt_category": category,
                    "method": method,
                    "model_path": model_path,
                    "prompt": prompt,
                    "completion": completion,
                    "raw_score": raw,
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
    fieldnames = [
        "idx",
        "prompt_hash",
        "prompt_category",
        "method",
        "model_path",
        "prompt",
        "completion",
        "raw_score",
        "score",
        "completion_tokens",
        "has_eos",
        "kl_proxy_sum",
        "kl_proxy_mean",
    ]
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    raw_scores_list = [float(r["raw_score"]) for r in rows]
    scores_list = [float(r["score"]) for r in rows]
    lengths_list = [float(r["completion_tokens"]) for r in rows]
    eos_values = [float(r["has_eos"]) for r in rows]
    kl_sum_values = [float(r["kl_proxy_sum"]) for r in rows]
    kl_mean_values = [float(r["kl_proxy_mean"]) for r in rows]

    score_stats = summarize(scores_list)
    raw_stats = summarize(raw_scores_list)
    summary = {
        "method": method,
        "model_path": model_path,
        "n_eval": len(rows),
        "n_unique_prompts": len({str(r["prompt_hash"]) for r in rows}),
        "score_mean": score_stats["mean"],
        "score_std": score_stats["std"],
        "score_sem": score_stats["sem"],
        "score_ci95_low": score_stats["ci95_low"],
        "score_ci95_high": score_stats["ci95_high"],
        "score_median": score_stats["median"],
        "raw_score_mean": raw_stats["mean"],
        "raw_score_std": raw_stats["std"],
        "completion_tokens_mean": summarize(lengths_list)["mean"],
        "completion_tokens_std": summarize(lengths_list)["std"],
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
    parser = argparse.ArgumentParser(description="Evaluate summarize_from_feedback PPO / TRACE / GRPO checkpoints.")
    parser.add_argument("--models", nargs="+", required=True, help="Model specs, e.g. ppo=./outputs/ppo trace=./outputs/trace grpo=./outputs/grpo")
    parser.add_argument("--dataset_name", default="openai/summarize_from_feedback")
    parser.add_argument("--dataset_config", default="comparisons")
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--reward_model_path", default="OpenAssistant/reward-model-deberta-v3-base")
    parser.add_argument("--ref_model_path", default="cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")
    parser.add_argument("--tokenizer_path", default=None, help="Optional shared tokenizer path. Defaults to each model path.")
    parser.add_argument("--output_dir", default="./outputs/summarize_feedback_eval")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--reward_max_length", type=int, default=1024)
    parser.add_argument("--missing_eos_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for quick tests. Omit for full split.")
    parser.add_argument(
        "--deduplicate_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate evaluation prompts by normalized prompt text before selecting max_samples.",
    )
    parser.add_argument(
        "--shuffle_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle the unique prompt pool with --seed before selecting max_samples.",
    )
    parser.add_argument(
        "--focus_trace_favorable_types",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Oversample prompt categories where TRACE previously tended to outperform PPO.",
    )
    parser.add_argument(
        "--focus_categories",
        nargs="*",
        default=TRACE_FAVORABLE_DEFAULT_CATEGORIES,
        help="Prompt categories to oversample when --focus_trace_favorable_types is enabled.",
    )
    parser.add_argument(
        "--focus_ratio",
        type=float,
        default=0.8,
        help="Fraction of max_samples drawn from focus_categories when focused evaluation is enabled.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_kl", action="store_true", help="Skip reference-model KL proxy computation.")
    args = parser.parse_args()

    if not 0.0 <= args.focus_ratio <= 1.0:
        raise ValueError("--focus_ratio must be between 0 and 1.")

    set_seed(args.seed)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    model_specs = parse_model_specs(args.models)

    first_tokenizer = load_tokenizer_for_model(model_specs[0][1], args)
    prompts = load_eval_prompts(args, first_tokenizer)
    if not prompts:
        raise RuntimeError("No evaluation prompts remained after filtering.")
    prompt_list_path = save_eval_prompts(prompts, args.output_dir)
    unique_count = len({key for _, _, key, _ in prompts})
    print(
        f"Loaded {len(prompts)} evaluation prompts "
        f"({unique_count} unique) from {args.dataset_name}/{args.dataset_split}"
    )
    print(f"Saved evaluation prompt list: {prompt_list_path}")

    reward_scorer = OpenAssistantPairReward(
        model_path=args.reward_model_path,
        max_length=args.reward_max_length,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    ref_model = None
    if not args.no_kl and args.ref_model_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        ref_model.eval()

    summaries: List[Dict[str, object]] = []
    with torch.inference_mode():
        for method, model_path in model_specs:
            summaries.append(
                evaluate_one_model(
                    method=method,
                    model_path=model_path,
                    prompts=prompts,
                    reward_scorer=reward_scorer,
                    ref_model=ref_model,
                    args=args,
                    device=device,
                    dtype=dtype,
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
            f"raw_score_mean={row['raw_score_mean']:.6f}, "
            f"kl_proxy_mean={row['kl_proxy_mean']:.6f}, "
            f"eos_rate={row['eos_rate']:.4f}, "
            f"tokens={row['completion_tokens_mean']:.2f}"
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
