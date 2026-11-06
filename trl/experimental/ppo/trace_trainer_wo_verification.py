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

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, logging
from accelerate.utils import gather_object
from datasets import Dataset
from packaging.version import Version
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import ModelOutput, is_peft_available, is_rich_available

from ...models.utils import unwrap_model_for_generation
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import (
    disable_dropout_in_model,
    log_table_to_comet_experiment,
    pad,
    prepare_deepspeed,
    selective_log_softmax,
)
from ..utils import (
    create_reference_model,
    empty_cache,
    first_true_indices,
    get_reward,
    peft_module_casting_to_bf16,
)
from .ppo_config import PPOConfig


if is_rich_available():
    from rich.console import Console
    from rich.table import Table


logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


INVALID_LOGPROB = 1.0


def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config ([`~transformers.GenerationConfig`]):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


def print_rich_table(df: pd.DataFrame) -> None:
    if not is_rich_available():
        raise ImportError(
            "The function `print_rich_table` requires the `rich` library. Please install it with `pip install rich`."
        )
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor) -> torch.Tensor:
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.

    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.

    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> ModelOutput:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `ModelOutput`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


@dataclass
class OnlineTrainerState(TrainerState):
    """
    Training state for online/on-policy trainers.

    Extends [`~transformers.TrainerState`] with an `episode` counter to track the current rollout/episode.

    Args:
        episode (`int`, defaults to 0): Zero-based episode index.
    """

    episode: int = 0


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: bool | None = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)
        self.is_gradient_checkpointing = policy.is_gradient_checkpointing

    def gradient_checkpointing_enable(self, **kwargs):
        self.policy.gradient_checkpointing_enable(**kwargs)
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.policy.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class TRACETrainer(_BaseTrainer):
    """TRACE ablation trainer without critic-based pivot value verification.

    All surprisal-selected candidate pivots are retained directly, and the
    trajectory-level TRACE bonus is distributed uniformly across them. The
    reliability gate, candidate proposal, pivot-only injection, and PPO update
    path remain unchanged.

    Trainer for Proximal Policy Optimization (PPO).

    For details on PPO, see the paper: [Proximal Policy Optimization
    Algorithms](https://huggingface.co/papers/1707.06347).

    Args:
        args ([`experimental.ppo.PPOConfig`]):
            Training arguments.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`]):
            Class to process the data.
        model (`torch.nn.Module`):
            Model to be trained. This is the policy model.
        ref_model (`torch.nn.Module`, *optional*):
            Reference model used to compute the KL divergence. If `None`, a copy of the policy model is created.
        reward_model (`torch.nn.Module`):
            Reward model used to compute the rewards.
        train_dataset ([`~datasets.Dataset`]):
            Dataset for training.
        value_model (`torch.nn.Module`):
            Value model used to predict the value of a state.
        data_collator ([`~transformers.DataCollatorWithPadding`], *optional*):
            Data collator to batch and pad samples from the dataset. If `None`, a default data collator is created
            using the `processing_class`.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training. If `None`, the
            optimizer and the learning rate scheduler are created using the
            [`~transformers.Trainer.create_optimizer_and_scheduler`] method.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the policy `model`
            will be wrapped with the specified PEFT adapter.
    """

    _tag_names = ["trl", "ppo"]
    _name = "PPO"
    _paper = {
        "title": "Fine-Tuning Language Models from Human Preferences",
        "id": "1909.08593",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{mziegler2019fine-tuning,
                title        = {{Fine-Tuning Language Models from Human Preferences}},
                author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
                year         = 2019,
                eprint       = {arXiv:1909.08593}
            }"""),
    }

    def __init__(
        self,
        args: PPOConfig,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        model: nn.Module,
        ref_model: nn.Module | None,
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: nn.Module,
        data_collator: DataCollatorWithPadding | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: list[TrainerCallback] | None = None,
        peft_config: "PeftConfig | None" = None,
    ) -> None:
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # Check that the kl estimator is valid
        if self.args.kl_estimator not in {"k1", "k3"}:
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(self.policy_model, PeftModel):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first "
                    "merge and unload the existing adapter, save the resulting base model, and then pass that base "
                    "model along with the new `peft_config` to the trainer."
                )

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        # trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        # setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        backup_model = self.model
        if hasattr(self.model, "policy"):
            self.model = self.model.policy  # save only the policy for inference
        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model
        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def compute_token_trace_rewards(
        self,
        kl,
        logprobs, # 1st round screening 수정을 위해 추가
        values,
        scores,
        sequence_lengths,
        padding_mask_p1,
        selectivity_low=0.80,
        selectivity_high=1.00,
        total_bonus_scale=3.0,
        score_transform="softplus", # "softplus", "sigmoid", "raw"
        debug=False,
        debug_n_samples=2,
    ):
        """
        kl: [B, T]
        values: [B, T]
        scores: [B]
        sequence_lengths: [B]   # 마지막 유효 토큰 index
        padding_mask_p1: [B, T]

        W/o-verification ablation:
        - 모든 surprisal candidate를 critic-value 검증 없이 최종 pivot으로 유지한다.
        - trajectory-level TRACE bonus는 선택된 candidate들에 균등 분배한다.
        """

        # ---------------------------
        # 0) shape / dtype / finite 검사
        # ---------------------------
        assert kl.ndim == 2, f"kl must be [B,T], got {tuple(kl.shape)}"
        assert values.ndim == 2, f"values must be [B,T], got {tuple(values.shape)}"
        assert padding_mask_p1.ndim == 2, f"padding_mask_p1 must be [B,T], got {tuple(padding_mask_p1.shape)}"
        assert sequence_lengths.ndim == 1, f"sequence_lengths must be [B], got {tuple(sequence_lengths.shape)}"
        assert scores.ndim == 1, f"scores must be [B], got {tuple(scores.shape)}"

        B, T = kl.shape
        assert logprobs.ndim == 2, f"logprobs must be [B,T], got {tuple(logprobs.shape)}"
        assert logprobs.shape == (B, T), f"logprobs shape mismatch: {tuple(logprobs.shape)} vs {(B, T)}"
        assert torch.isfinite(logprobs).all(), "logprobs contains NaN/Inf"

        assert values.shape == (B, T), f"values shape mismatch: {tuple(values.shape)} vs {(B, T)}"
        assert padding_mask_p1.shape == (B, T), f"padding_mask_p1 shape mismatch: {tuple(padding_mask_p1.shape)} vs {(B, T)}"
        assert sequence_lengths.shape[0] == B, f"sequence_lengths batch mismatch: {sequence_lengths.shape[0]} vs {B}"
        assert scores.shape[0] == B, f"scores batch mismatch: {scores.shape[0]} vs {B}"

        assert torch.isfinite(kl).all(), "kl contains NaN/Inf"
        assert torch.isfinite(values).all(), "values contains NaN/Inf"
        assert torch.isfinite(scores).all(), "scores contains NaN/Inf"

        trace_rewards = torch.zeros_like(values)

        num_candidates = 0
        num_final_pivots = 0
        verified_bonus_bases = []
        verified_total_bonus = []

        debug_samples = []

        # 좋은 final reward trajectory만 후보 생성
        score_threshold = torch.median(scores).item()
        effective_score_threshold = max(score_threshold, 0.0) # negative reward trajectory는 후보 생성에 사용하지 않음

        # ---------------------------
        # 1) batch loop
        # ---------------------------
        for b in range(B):
            valid_len = int(sequence_lengths[b].item()) + 1 # 각 샘플마다 유효한 토큰 길이 (padding 제외, 0-based index -> length로 변환)

            if valid_len <= 0:
                if debug and len(debug_samples) < debug_n_samples:
                    debug_samples.append({
                        "sample_idx": b,
                        "skip_reason": "empty_valid_sequence < 0",
                        "valid_len": valid_len,
                        "score_raw": float(scores[b].item()),
                    })
                continue


            # ---------------------------
            # 1) COCOA-inspired 1st screening
            #    policy가 쉽게 낸 토큰이 아니라,
            #    good outcome trajectory에서 선택성이 있었던 토큰을 후보로 남김
            # ---------------------------

            # token selectivity = surprisal = -log pi(a_t | prefix_t)
            # logprobs는 rollout policy가 실제 생성 token에 부여한 log-prob
            token_selectivity = -logprobs[b, :valid_len]
            token_selectivity = torch.clamp(token_selectivity, min=0.0, max=50.0) # 안정성 처리

            valid_selectivity = token_selectivity


            if scores[b].item() <= effective_score_threshold:
                candidate_indices = []
                q_low = None
                q_high = None
            else:
                q_low = torch.quantile(valid_selectivity, selectivity_low)
                q_high = torch.quantile(valid_selectivity, selectivity_high)

                candidate_indices = []
                for idx in range(valid_len):
                    s = token_selectivity[idx].item()
                    if q_low.item() <= s <= q_high.item():
                        candidate_indices.append(idx)

                candidate_indices = sorted(
                    candidate_indices,
                    key=lambda idx: token_selectivity[idx].item(),
                    reverse=True,
                )

            num_candidates += len(candidate_indices)

            sample_debug = {
                "sample_idx": b,
                "valid_len": valid_len,
                "score_raw": float(scores[b].item()),
                "selectivity_q_low": float(q_low.item()) if q_low is not None else None,
                "selectivity_q_high": float(q_high.item()) if q_high is not None else None,
                "top_selectivity": [float(x) for x in token_selectivity[torch.argsort(token_selectivity, descending=True)[:5]].tolist()],
                "candidate_pivots": [],
                "final_pivots": [],
            }

            # ---------------------------
            # 2) candidate -> selected (w/o value verification ablation)
            # ---------------------------
            # critic segment 비교와 delta_v > 0 필터를 제거하고,
            # 모든 surprisal candidate를 최종 reward pivot으로 직접 유지한다.
            selected_pivots = []
            pivot_positions = sorted(candidate_indices)

            for pivot_idx in pivot_positions:
                cand_info = {
                    "pivot_idx": int(pivot_idx),
                    "selectivity": float(token_selectivity[pivot_idx].item()),
                    "delta_v": None,
                    "verified": False,
                    "verification_skipped": True,
                    "selected": True,
                }
                sample_debug["candidate_pivots"].append(cand_info)
                selected_pivots.append(cand_info)

            # ---------------------------
            # 3) shared pivot-only reward injection
            # ---------------------------
            score_b = scores[b]

            if score_transform == "raw":
                bonus_base = score_b
            elif score_transform == "sigmoid":
                bonus_base = torch.sigmoid(score_b)
            else:
                bonus_base = F.softplus(score_b)

            total_bonus = float(bonus_base.item()) * float(total_bonus_scale)

            if len(selected_pivots) > 0:
                # value verification과 delta_v 기반 weighting을 제거했으므로,
                # 동일한 trajectory bonus를 모든 candidate pivot에 균등 분배한다.
                uniform_share = 1.0 / len(selected_pivots)
                pivot_shares = [uniform_share for _ in selected_pivots]

                verified_bonus_bases.append(float(bonus_base.item()))
                verified_total_bonus.append(float(total_bonus))

                for cand, pivot_share in zip(selected_pivots, pivot_shares):
                    pivot_idx = cand["pivot_idx"]

                    num_final_pivots += 1

                    # 전체 trajectory bonus를 candidate pivot 수로 균등 분배
                    pivot_bonus = total_bonus * pivot_share

                    # reward_span 없이 pivot token에만 reward 지급
                    trace_rewards[b, pivot_idx] += pivot_bonus

                    cand.update({
                        "score_transformed": float(bonus_base.item()),
                        "total_bonus": float(total_bonus),
                        "pivot_share": float(pivot_share),
                        "pivot_bonus": float(pivot_bonus),
                        "reward_position": int(pivot_idx),
                        "token_bonus": float(pivot_bonus),
                    })

                    sample_debug["final_pivots"].append(cand)

            else:
                # candidate pivot이 없으면 reward injection을 수행하지 않는다.
                if debug:
                    sample_debug["skip_reward_injection_reason"] = (
                        f"num_selected_pivots={len(selected_pivots)}"
                    )
            if debug and len(debug_samples) < debug_n_samples:
                valid_reward_sum = float(trace_rewards[b, :valid_len].sum().item())
                invalid_reward_sum = float(trace_rewards[b, valid_len:].sum().item()) if valid_len < T else 0.0
                sample_debug["valid_reward_sum_before_mask"] = valid_reward_sum
                sample_debug["invalid_reward_sum_before_mask"] = invalid_reward_sum
                debug_samples.append(sample_debug)
        # ---------------------------
        # 4) padding mask 적용 + 후검증
        # ---------------------------
        trace_rewards = torch.masked_fill(trace_rewards, padding_mask_p1, 0.0)

        if debug:
            assert torch.isfinite(trace_rewards).all(), "trace_rewards contains NaN/Inf after masking"
            masked_vals = trace_rewards.masked_select(padding_mask_p1)
            if masked_vals.numel() > 0:
                assert float(masked_vals.abs().max().item()) == 0.0, "trace_rewards leaked into padded positions"

        # ---------------------------
        # 5) 요약 통계
        # ---------------------------
        valid_mask = ~padding_mask_p1
        avg_reward = (
            float(trace_rewards.masked_select(valid_mask).mean().item())
            if valid_mask.any()
            else 0.0
        )

        stats = {
            "trace/num_candidates": float(num_candidates),
            "trace/num_final_pivots": float(num_final_pivots),
            # delta_v is intentionally not computed in the w/o-verification ablation.
            "trace/avg_delta_v_final_pivots": 0.0,
            "trace/avg_total_bonus": float(sum(verified_total_bonus) / max(len(verified_total_bonus), 1)),
        }

        return trace_rewards, stats, debug_samples

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with (
                    unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                        generation_kwargs=generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                    ) as unwrapped_model
                ):
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
                non_score_reward = -args.kl_coef * kl
                ppo_rewards = non_score_reward.clone()
                actual_start = torch.arange(ppo_rewards.size(0), device=ppo_rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < ppo_rewards.size(1), sequence_lengths_p1, sequence_lengths)
                ppo_rewards[actual_start, actual_end] += scores

                def _safe_pearson(x, y):
                    x = x.float()
                    y = y.float()
                    x = x - x.mean()
                    y = y - y.mean()
                    denom = x.norm() * y.norm()
                    if denom <= 1e-8:
                        return torch.tensor(0.0, device=x.device)
                    return (x * y).sum() / denom

                terminal_value_idx = sequence_lengths.clamp(min=0, max=values.size(1) - 1)
                batch_idx = torch.arange(values.size(0), device=values.device)
                terminal_values = values[batch_idx, terminal_value_idx]

                critic_score_corr = _safe_pearson(
                    terminal_values.detach(),
                    scores.detach(),
                ).item()

                if not hasattr(self, "trace_critic_corr_ema"):
                    self.trace_critic_corr_ema = critic_score_corr
                else:
                    self.trace_critic_corr_ema = (
                        0.9 * self.trace_critic_corr_ema
                        + 0.1 * critic_score_corr
                    )

                trace_active = (
                    scores.mean().item() > 0.5
                    and self.trace_critic_corr_ema > 0.5
                )

                if not trace_active:
                    trace_rewards = torch.zeros_like(values)
                    trace_stats = {
                        "trace/active": 0.0,
                        "trace/critic_score_corr": float(critic_score_corr),
                        "trace/critic_score_corr_ema": float(self.trace_critic_corr_ema),
                        "trace/num_candidates": 0.0,
                        "trace/num_final_pivots": 0.0,
                        "trace/avg_delta_v_final_pivots": 0.0,
                        "trace/avg_total_bonus": 0.0,
                    }
                    trace_debug = []
                else:
                    trace_rewards, trace_stats, trace_debug = self.compute_token_trace_rewards(
                        kl=kl,
                        logprobs=logprobs,
                        values=values,
                        scores=scores,
                        sequence_lengths=sequence_lengths,
                        padding_mask_p1=padding_mask_p1,
                        score_transform="softplus",
                        total_bonus_scale=3.0,
                        selectivity_low=0.80,
                        selectivity_high=1.00,
                        debug=(update % 50 == 0),
                        debug_n_samples=2,
                    )
                    trace_stats["trace/active"] = 1.0
                    trace_stats["trace/critic_score_corr"] = float(critic_score_corr)
                    trace_stats["trace/critic_score_corr_ema"] = float(self.trace_critic_corr_ema)

                # 5. whiten rewards
                if args.whiten_rewards:
                    ppo_rewards = masked_whiten(ppo_rewards, mask=~padding_mask_p1, shift_mean=False)
                    ppo_rewards = torch.masked_fill(ppo_rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    # 벨만 오차(TD-error) 계산 시 TRACE 보상의 위치를 명확히 분리하여 결합함.
                    delta = ppo_rewards[:, t] + trace_rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode

                metrics.update(trace_stats)   # <- 여기 추가

                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                ppo_rewards,
                trace_rewards,
                trace_stats,
                trace_debug,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        if self.eval_dataset is None:
            return  # no eval set to sample from (pass eval_dataset and eval_strategy != "no" for sample generations)
        args = self.args
        processing_class = self.processing_class
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (0.01 + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        table = defaultdict(list)
        with (
            unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model
        ):
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
