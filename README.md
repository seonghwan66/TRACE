# TRACE: Critic-Verified Credit Localization for LLM Post-Training

Official implementation of:

> **Finding the Pivot: Critic-Verified Credit Localization for LLM Post-Training**  
> Seonghwan Kim, Hoyoung Song, Chanju Lim, and Hyunseok Kim  
> **Submitted to IEEE Access**

This repository contains the implementation and experimental code for **TRACE (Trajectory Reasoning Adaptive Credit Estimation)**, a selective credit-localization framework for reinforcement learning-based LLM post-training.

TRACE is implemented on top of [Hugging Face TRL](https://github.com/huggingface/trl).

---

## Overview

In reinforcement learning for large language models, a response is generated as a sequence of token-level decisions, while the reward is often provided only after the entire response is completed.

Such delayed sequence-level feedback evaluates the overall response quality, but does not explicitly identify **which intermediate token decisions contributed to the final outcome**.

TRACE addresses this temporal credit assignment problem by identifying a sparse set of important token positions, called **pivots**, and selectively assigning additional outcome credit only to critic-verified pivots.

Unlike methods that require external process reward models or step-level annotations, TRACE uses signals already available during PPO training.

---

## TRACE

TRACE consists of four main components:

### 1. Reliability Gate

TRACE first determines whether the current PPO critic is sufficiently aligned with observed trajectory-level outcomes.

Credit localization is activated only when the critic is considered reliable enough to support local credit estimation.

### 2. Pivot Detection

Within relatively successful trajectories, TRACE uses **sampled-token surprisal** to identify candidate token positions.

A token with relatively high surprisal may correspond to an important local decision, but surprisal alone is not considered sufficient evidence of contribution.

### 3. Critic-Based Value Verification

Each candidate pivot is verified using the PPO critic.

TRACE compares critic-estimated values across the segments before and after a candidate pivot and retains only candidates associated with a positive value improvement.

### 4. Value-Weighted Credit Allocation

The trajectory-level outcome reward is transformed into an additional non-negative bonus.

This bonus is distributed only among verified pivots in proportion to their critic-estimated value improvements.

The localized TRACE reward is incorporated before Generalized Advantage Estimation (GAE), while the original PPO objective remains unchanged.

---

## Experimental Setup

TRACE is evaluated on three delayed-feedback language-generation tasks:

| Dataset | Task |
|---|---|
| **TL;DR** | Reddit post summarization |
| **Summarize from Feedback (SFF)** | Human-feedback-based summarization |
| **OASST2** | Open-ended assistant response generation |

The following methods are compared under matched training budgets:

- **PPO**
- **GRPO**
- **TRACE**

Each method is trained using **10,000 generated trajectories per task**, and results are aggregated over **5 independent runs**.

---

## Main Results

TRACE achieved the highest observed mean evaluation reward among PPO, GRPO, and TRACE across all three tasks.

| Dataset | PPO | GRPO | TRACE |
|---|---:|---:|---:|
| TL;DR | 3.590 | 3.174 | **4.106** |
| SFF | 1.561 | 0.650 | **1.892** |
| OASST2 | 0.441 | -0.581 | **0.817** |

Mean reward differences between TRACE and PPO were:

- **TL;DR:** +0.516
- **SFF:** +0.331
- **OASST2:** +0.376

These values should be interpreted as **observed mean differences rather than evidence of statistical superiority**, since the run-level confidence intervals overlap.

---

## Ablation Study

The repository also includes two TRACE ablation variants:

### Without Value-Weighted Allocation

```text
trl/experimental/ppo/
└── trace_trainer_wo_value_weighted_allocation.py
```

This variant retains critic-based pivot verification but distributes the localized bonus uniformly across verified pivots.

### Without Value Verification

```text
trl/experimental/ppo/
└── trace_trainer_wo_verification.py
```

This variant removes critic-based filtering and allocates credit to surprisal-selected candidate pivots.

The ablation results indicate that the effects of critic verification and value-weighted allocation are task-dependent.

---

## Repository Structure

```text
TRACE/
│
├── trl/
│   └── experimental/
│       └── ppo/
│           ├── trace_trainer.py
│           ├── trace_trainer_wo_value_weighted_allocation.py
│           └── trace_trainer_wo_verification.py
│
├── examples/
│   ├── accelerate_configs/
│   │   ├── deepspeed_zero2_cpu_offload.yaml
│   │   ├── deepspeed_zero2_cpu_offload_grpo.yaml
│   │   └── deepspeed_zero3.yaml
│   │
│   └── scripts/
│       └── ppo/
│           ├── trace_tldr.py
│           ├── trace_summarize_feedback.py
│           ├── trace_oasst2.py
│           │
│           ├── ppo_tldr.py
│           ├── ppo_summarize_feedback.py
│           ├── ppo_oasst2.py
│           │
│           ├── grpo_tldr.py
│           ├── grpo_summarize_feedback.py
│           ├── grpo_oasst2.py
│           │
│           ├── tldr_test_generic.py
│           ├── sff_test_generic.py
│           └── oasst2_test_generic.py
│
└── README.md
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/seonghwan66/TRACE.git
cd TRACE
```

Install the repository in editable mode:

```bash
pip install -e .
```

The implementation is based on Hugging Face TRL and uses PyTorch, Transformers, Accelerate, and DeepSpeed for LLM post-training experiments.

GPU memory requirements depend on the task and model configuration.

The provided Accelerate configurations include CPU offloading for experiments conducted under limited GPU-memory settings.

---

## Training

### TRACE

Task-specific TRACE training entry points are provided for:

```text
examples/scripts/ppo/trace_tldr.py
examples/scripts/ppo/trace_summarize_feedback.py
examples/scripts/ppo/trace_oasst2.py
```

### PPO Baseline

```text
examples/scripts/ppo/ppo_tldr.py
examples/scripts/ppo/ppo_summarize_feedback.py
examples/scripts/ppo/ppo_oasst2.py
```

### GRPO Baseline

```text
examples/scripts/ppo/grpo_tldr.py
examples/scripts/ppo/grpo_summarize_feedback.py
examples/scripts/ppo/grpo_oasst2.py
```

Each script contains the task-specific configuration used for the corresponding experiment.

---

## Evaluation

Evaluation scripts are provided for all three tasks:

```text
examples/scripts/ppo/tldr_test_generic.py
examples/scripts/ppo/sff_test_generic.py
examples/scripts/ppo/oasst2_test_generic.py
```

The experiments evaluate outcome-level reward together with generation diagnostics such as:

- sampled response-token KL proxy
- EOS / termination behavior
- response length

These diagnostics are reported alongside reward because reward-model scores alone should not be interpreted as absolute measures of response quality.

---

## Implementation

The main TRACE implementation is located at:

```text
trl/experimental/ppo/trace_trainer.py
```

The core pipeline follows:

```text
Trajectory rollout
      ↓
Sequence-level outcome reward
      ↓
Reliability Gate
      ↓
Pivot Detection
(sampled-token surprisal)
      ↓
Value Verification
(PPO critic)
      ↓
Verified Pivots
      ↓
Value-Weighted Credit Allocation
      ↓
Localized TRACE reward
      ↓
GAE
      ↓
Standard PPO objective
```

TRACE does **not** require:

- external process reward models
- human step-level annotations
- a modified PPO optimization objective

---

## Paper Status

The manuscript

> **Finding the Pivot: Critic-Verified Credit Localization for LLM Post-Training**

has been **submitted to IEEE Access**.

The repository may be updated as the manuscript progresses through the review process.

---

## Authors

- **Seonghwan Kim**
- Hoyoung Song
- Chanju Lim
- Hyunseok Kim

Robotic Artificial Intelligence Lab (RAILab)  
Department of Computer Engineering  
Dong-A University, Republic of Korea

---

## Citation

The manuscript is currently under review. Citation information will be updated upon publication.

For now, please refer to the manuscript as:

```text
Seonghwan Kim, Hoyoung Song, Chanju Lim, and Hyunseok Kim,
"Finding the Pivot: Critic-Verified Credit Localization for LLM Post-Training,"
submitted to IEEE Access.
```

---

## Acknowledgment

This implementation is built upon the open-source
[Hugging Face TRL](https://github.com/huggingface/trl) library.

We thank the TRL contributors for providing the foundation-model post-training framework used in this research.

---

## Upstream TRL

This repository is derived from:

**TRL: Transformers Reinforcement Learning**  
https://github.com/huggingface/trl

If you are looking for the original TRL library, documentation, or general-purpose trainers, please refer to the official repository and documentation:

- TRL Repository: https://github.com/huggingface/trl
- TRL Documentation: https://huggingface.co/docs/trl/

---

## License

This repository is based on Hugging Face TRL and retains the original **Apache License 2.0**.

See [LICENSE](LICENSE) for details.

Original TRL citation:

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis
             and Beeching, Edward and Thrush, Tristan and Lambert, Nathan
             and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```
