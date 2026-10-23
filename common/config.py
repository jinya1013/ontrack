from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
from trl.trainer.grpo_config import GRPOConfig

POLICY_TYPES = {"dit_confidence", "learn2pd"}
POLICY_INPUT_TYPES = {"confidence", "hidden", "hidden_only"}
FEATURE_KV_CACHE_MODES = {"no_cache", "prefix_cache"}


@dataclass
class Config(GRPOConfig):
    """Policy architecture and offline teacher-forcing settings.

    The Hugging Face/TRL base class supplies the optimizer, schedule, batching
    and checkpointing arguments.
    """

    model_path: Optional[str] = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={"help": "Path or Hub id of the frozen diffusion LM."},
    )
    model_revision: Optional[str] = field(
        default=None, metadata={"help": "Pinned Hub revision of model_path."}
    )
    training_math_protocol: str = field(
        default="common-math-v1",
        metadata={"help": "Prompt/scorer contract the teacher data was built with."},
    )
    math_training_invalid_gold: str = field(
        default="exclude",
        metadata={"help": "How the teacher data treated unparseable MATH gold answers."},
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={"help": "Prompt truncation (left) used by the teacher data."},
    )
    block_length: int = field(default=32, metadata={"help": "Diffusion block length."})
    temperature: float = field(
        default=0.0, metadata={"help": "Token temperature of the denoiser (0 = greedy)."}
    )
    beta: float = field(default=0.0, metadata={"help": "Unused (no reference-model KL)."})

    # Policy architecture.
    policy_type: str = field(
        default="dit_confidence", metadata={"help": "dit_confidence or learn2pd."}
    )
    policy_input_type: str = field(
        default="confidence",
        metadata={
            "help": "Per-position policy feature: confidence (top-k denoiser "
            "probabilities), hidden (final hidden state; requires "
            "policy_hidden_residual) or hidden_only."
        },
    )
    policy_hidden_residual: bool = field(
        default=False,
        metadata={
            "help": "Joint policy: a zero-initialized hidden-state projection is "
            "added to the confidence stream; both branches are trained."
        },
    )
    policy_hidden_only_norm: bool = field(
        default=True, metadata={"help": "LayerNorm on hidden_only inputs."}
    )
    confidences_top_p: int = field(
        default=1, metadata={"help": "Number of top denoiser probabilities per position."}
    )
    use_top1_margin: bool = field(
        default=False,
        metadata={"help": "Replace top-2 probabilities by (top-1, top-1 minus top-2)."},
    )
    policy_hidden_dim: Optional[int] = field(default=128)
    policy_feedforward_dim: Optional[int] = field(default=None)
    policy_num_heads: int = field(default=2)
    policy_num_blocks: int = field(default=1)
    policy_dropout: float = field(default=0.0)
    policy_time_embed_dim: int = field(default=128)
    policy_time_period: float = field(default=1)
    policy_smart_init: Optional[float] = field(
        default=None,
        metadata={"help": "Initial output bias (identity AdaLN); None keeps PyTorch init."},
    )

    # Offline teacher forcing.
    g_trace_teacher_data: Optional[str] = field(
        default=None,
        metadata={"help": "Directory or Parquet glob of teacher trajectories."},
    )
    g_trace_feature_kv_cache: str = field(
        default="no_cache",
        metadata={
            "help": "Denoiser cache used to recompute teacher-state policy features: "
            "no_cache, or prefix_cache to reproduce Fast-dLLM prefix-cache features "
            "(the cache-aware policy)."
        },
    )
    teacher_forcing_weight: float = field(default=1.0)
    timestep_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Teacher states per policy forward; None processes all at once."},
    )
    loglikelihood_dtype: Optional[str] = field(
        default="float32", metadata={"help": "Torch dtype of the Bernoulli log-likelihood."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.beta != 0.0:
            raise ValueError("teacher forcing uses no reference-model KL; set beta: 0.0")
        if self.policy_type not in POLICY_TYPES:
            raise ValueError(f"policy_type must be one of {sorted(POLICY_TYPES)}")
        if self.policy_input_type not in POLICY_INPUT_TYPES:
            raise ValueError(f"policy_input_type must be one of {sorted(POLICY_INPUT_TYPES)}")
        if self.policy_hidden_residual != (self.policy_input_type == "hidden"):
            raise ValueError("policy_input_type=hidden is the joint policy (policy_hidden_residual=true)")
        if self.policy_hidden_residual and self.confidences_top_p != 1:
            raise ValueError("the joint policy uses top-1 confidence")
        if not self.policy_hidden_only_norm and self.policy_input_type != "hidden_only":
            raise ValueError("policy_hidden_only_norm=false requires policy_input_type=hidden_only")
        if self.use_top1_margin and (
            self.policy_input_type != "confidence" or self.confidences_top_p != 2
        ):
            raise ValueError("use_top1_margin requires confidence input with confidences_top_p=2")
        if self.policy_type == "learn2pd" and (
            self.policy_input_type != "confidence" or self.confidences_top_p != 1
        ):
            raise ValueError("learn2pd uses top-1 confidence")
        if self.g_trace_feature_kv_cache not in FEATURE_KV_CACHE_MODES:
            raise ValueError(f"g_trace_feature_kv_cache must be one of {sorted(FEATURE_KV_CACHE_MODES)}")
        if self.g_trace_feature_kv_cache != "no_cache" and "Dream" in (self.model_path or ""):
            raise ValueError("prefix-cache teacher features support LLaDA only")
        if self.math_training_invalid_gold not in {"error", "exclude"}:
            raise ValueError("math_training_invalid_gold must be error or exclude")
        if self.loglikelihood_dtype is not None and isinstance(self.loglikelihood_dtype, str):
            if self.loglikelihood_dtype.lower() in {"none", "null"}:
                self.loglikelihood_dtype = None
            else:
                dtype = getattr(torch, self.loglikelihood_dtype, None)
                if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                    raise TypeError(f"invalid loglikelihood_dtype {self.loglikelihood_dtype!r}")
                self.loglikelihood_dtype = dtype
