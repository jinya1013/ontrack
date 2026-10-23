# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
"""Training prompts (GSM8K and MATH train splits) under the common-math-v1 protocol."""
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk

from common.parsing.answers import extract_hash_answer

# Optional directory of datasets saved with ``save_to_disk``; Hub otherwise.
LOCAL_DATASETS_DIR = Path(os.environ.get("DLLM_DATASETS_DIR", Path(__file__).resolve().parent / "datasets"))


def _load_dataset_with_fallback(
    repo_id: str, config: str | None = None, local_name: str | None = None
) -> Dataset:
    """Load dataset from local disk if available, otherwise from HuggingFace."""
    if local_name:
        local_path = LOCAL_DATASETS_DIR / local_name
        if local_path.exists():
            return load_from_disk(str(local_path))
    if config:
        return load_dataset(repo_id, config)
    return load_dataset(repo_id)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = _load_dataset_with_fallback(
        "openai/gsm8k", config="main", local_name="gsm8k"
    )
    data = data[split]
    return data.map(
        lambda x, index: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
            "dataset_type": "gsm8k",
            "g_trace_sample_id": f"{split}:{index}",
        },
        with_indices=True,
    )


def get_math_questions(split="train") -> Dataset:
    data = _load_dataset_with_fallback("ankner/math-500", local_name="math-500")
    data = data[split]
    data = data.map(
        lambda x, index: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
            "dataset_type": "math",
            "g_trace_sample_id": f"{split}:{index}",
        },
        with_indices=True,
    )
    return data


def apply_common_math_protocol(dataset):
    """Convert online math rows without changing IDs, split, or order."""
    from common.math_protocol import PROTOCOL, prompt_messages
    from common.parsing.answers import extract_math_answer

    def convert(row):
        name = row["dataset_type"]
        if name == "gsm8k":
            question, gold = row["question"], row["answer"]
        elif name == "math":
            question = row["problem"]
            # Same solution-to-gold conversion as MATH500Dataset.__getitem__.
            gold = extract_math_answer(row["solution"])
            if gold is None:
                raise ValueError(f"Missing explicit MATH gold: {row['g_trace_sample_id']}")
        else:
            raise ValueError(f"{PROTOCOL} does not support {name}")
        return {"prompt": prompt_messages(question), "answer": gold,
                "math_protocol": PROTOCOL}

    return dataset.map(convert)


def validate_common_math_gold(dataset, *, invalid_gold="error", report_path=None):
    """Check source labels before rollout; never alter the response scorer."""
    import json
    from common.math_protocol import score_answer, protocol_metadata
    if invalid_gold not in {"error", "exclude"}:
        raise ValueError("invalid_gold must be error or exclude")
    invalid = []
    valid_indices = []
    for index, row in enumerate(dataset):
        try:
            score_answer("", row["answer"])
        except ValueError as exc:
            invalid.append({"dataset": row["dataset_type"], "sample_id": row["g_trace_sample_id"],
                            "gold": row["answer"], "error": str(exc)})
        else:
            valid_indices.append(index)
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"protocol": protocol_metadata(), "invalid_gold_policy": invalid_gold,
                                   "original_count": len(dataset), "valid_count": len(valid_indices),
                                   "invalid": invalid}, indent=2, ensure_ascii=False) + "\n")
    if invalid and invalid_gold == "error":
        keys = ", ".join(f"{r['dataset']}/{r['sample_id']}" for r in invalid)
        raise ValueError(f"Unparseable training gold ({len(invalid)}); review labels before training: {keys}")
    return dataset.select(valid_indices) if invalid else dataset
