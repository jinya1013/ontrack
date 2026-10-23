"""Train an unmasking policy by offline teacher forcing.

    python -m train.train --config configs/llada/ontrack.yaml

``--g_trace_teacher_data``, ``--output_dir``, ``--seed`` etc. override the YAML.
The final policy is written to ``<output_dir>/final``.
"""

import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.math_protocol import protocol_metadata
from common.models.factory import build_policy
from data.data_utils import set_random_seed
from train.trainer import Trainer

torch.set_float32_matmul_precision("high")

MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}


def load_teacher_dataset(config: Config):
    """Load teacher trajectories and check they match the configured prompt protocol."""
    data_files = config.g_trace_teacher_data
    if not data_files:
        raise ValueError("g_trace_teacher_data is required")
    if os.path.isdir(data_files):
        data_files = os.path.join(data_files, "*.parquet")
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    math_rows = dataset.filter(lambda row: row["dataset"] in {"gsm8k", "math"})
    if len(math_rows):
        for column, expected in (
            ("training_math_protocol", config.training_math_protocol),
            ("math_training_invalid_gold", config.math_training_invalid_gold),
            ("math_protocol_metadata", json.dumps(protocol_metadata(config.max_prompt_length), sort_keys=True)),
        ):
            found = set(math_rows[column]) if column in math_rows.column_names else set()
            if found != {expected}:
                raise ValueError(
                    f"teacher data {column} differs from the current prompt/scorer "
                    f"implementation: data={sorted(found)} expected={expected!r}"
                )
    print(f"Teacher dataset: {len(dataset)} trajectories from {config.g_trace_teacher_data}")
    return dataset


def main(config: Config):
    set_random_seed(config.seed)
    dataset = load_teacher_dataset(config).shuffle(seed=config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "LLaDA" in config.model_path:
        model_class = AutoModel
        if config.g_trace_feature_kv_cache != "no_cache":
            # Fast-dLLM v1 modeling with KV-cache support; same weights and config.
            from common.models.modeling_llada_fastdllm import LLaDAModelLM as model_class
        config.model_type = "LLaDA"
    elif "Dream" in config.model_path:
        model_class = AutoModel
        config.model_type = "Dream"
    else:
        raise ValueError(f"Model path {config.model_path} not supported")
    model = model_class.from_pretrained(
        config.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        revision=config.model_revision,
    ).to(device)
    config.mask_id = MASK_TOKENS_MAP[config.model_type]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True, revision=config.model_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.requires_grad_(False)

    policy = build_policy(config, model.config.hidden_size).to(device)
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {config.policy_type}/{config.policy_input_type}; "
          f"parameters={total:,} trainable={trainable:,}")

    os.makedirs(config.output_dir, exist_ok=True)
    trainer = Trainer(
        model=policy,
        dllm=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = os.path.join(config.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved final policy to {final_dir} (step {trainer.state.global_step})")


if __name__ == "__main__":
    (config,) = TrlParser((Config,)).parse_args_and_config()
    main(config)
