"""Generate (and optionally score) one dataset under a pinned evaluation protocol.

Examples::

    # learned policy at threshold 0.9
    python -m eval.eval --dataset gsm8k --remasking policy \
        --config configs/llada/ontrack.yaml --policy_path outputs/llada_ontrack/final \
        --policy_accept_threshold 0.9 --output_dir results/ontrack/thr0.9
    # Fast-dLLM threshold baseline with the prefix KV cache
    python -m eval.eval --dataset math --remasking fastdllm --thres 0.9 \
        --kv_cache prefix_cache --output_dir results/fastdllm/thr0.9

``--defer_scoring`` saves the unscored generations; score them later on CPU
with ``python -m eval.scoring <file>``.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.models.factory import build_policy
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.humaneval import HumanEvalDataset
from data.loaders.math500 import MATH500Dataset
from data.loaders.mbpp import MBPPDataset
from data.sanitize import sanitize_humaneval
from data.sanitize import sanitize_mbpp
from eval.protocols import DATASET_SIZES
from eval.protocols import FOUR_SHOT_PROTOCOLS
from eval.protocols import LENGTH_TRANSFER_PROTOCOL
from eval.protocols import MATH500_TPS_PROTOCOL
from eval.protocols import PROTOCOLS
from eval.protocols import code_provenance
from eval.protocols import configure
from eval.scoring import evaluate_code
from eval.scoring import score_results

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}
MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}
LLADA_TOKENIZER = "GSAI-ML/LLaDA-8B-Instruct"


def stop_sequences(dataset_name):
    """lm-eval 0.4.9.1 stop strings applied to every decoded generation."""
    return {
        "gsm8k": ["Question:", "</s>", "<|im_end|>"],
        "math": ["Problem:"],
        "humaneval": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
        "mbpp": ["[DONE]"],
    }[dataset_name]


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def selected_set_summary(sizes):
    nonzero = [v for v in sizes if v > 0]
    return {
        "selected_set_sizes_by_macro_step": sizes,
        "mean_selected_set_size": float(np.mean(nonzero)) if nonzero else None,
        "median_selected_set_size": float(np.median(nonzero)) if nonzero else None,
        "max_selected_set_size": max(sizes, default=None),
    }


def evaluate(model, tokenizer, dataloader, dataset_name, *, generation_kwargs, gen_length, mask_id):
    """Decode every batch; record text, NFE, committed set sizes and timing."""
    model.eval()
    all_generations = []
    device = model.device
    wrapper_start = time.perf_counter()
    counted_tokens_total = 0
    is_code_dataset = dataset_name in ["humaneval", "mbpp"]

    with torch.no_grad():
        for batch in tqdm(dataloader):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].bool().to(device)
            prompts = batch["prompts"]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_start = time.perf_counter()
            result = generate_unified(
                model=model, prompt=input_ids, attention_mask=attention_mask,
                gen_length=gen_length, mask_id=mask_id, **generation_kwargs,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_start
            out = result.sequences
            selected_set_sizes = result.selected_set_sizes.tolist()
            steps_taken = result.steps_taken.reshape(-1).tolist()

            # As in Learn2PD's lm-eval wrapper: decode with special tokens, cut
            # at the stop strings, re-tokenize and count, then strip specials.
            generated_ids = out[:, -gen_length:]
            generated_texts = []
            counted_tokens = []
            for token_row in generated_ids:
                text = tokenizer.decode(token_row, skip_special_tokens=False)
                for stop_sequence in stop_sequences(dataset_name):
                    if stop_sequence in text:
                        text = text.split(stop_sequence, 1)[0]
                ids = tokenizer(text)["input_ids"]
                counted_tokens.append(len(ids))
                generated_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
            counted_tokens_total += sum(counted_tokens)
            wall_time_per_sample = (time.time() - start_time) / len(generated_texts)

            if is_code_dataset:
                raw_prompts = batch["raw_prompts"] if dataset_name == "humaneval" else batch["texts"]
                entry_points = batch["entry_points"] if dataset_name == "humaneval" else [None] * len(raw_prompts)
                rows = []
                for j, gen_text in enumerate(generated_texts):
                    try:
                        sanitized = (
                            sanitize_humaneval(raw_prompts[j] + gen_text, entry_points[j])
                            if dataset_name == "humaneval" else sanitize_mbpp(gen_text)
                        )
                    except Exception as error:
                        print(f"Warning: failed to sanitize {batch['task_ids'][j]}: {error}")
                        sanitized = raw_prompts[j] + gen_text if dataset_name == "humaneval" else gen_text
                    rows.append({
                        "task_id": batch["task_ids"][j],
                        "prompt": raw_prompts[j],
                        "prompt_input": prompts[j],
                        "generation_raw": gen_text,
                        "generation_sanitized": sanitized,
                        "test_cases": batch["test_cases"][j],
                        "entry_point": entry_points[j],
                    })
            else:
                rows = [
                    {
                        "question": batch["questions"][j],
                        "prompt_input": prompts[j],
                        "generations": generated_texts[j],
                        "ground_truth": (
                            batch["answers"][j].item()
                            if hasattr(batch["answers"][j], "item") else batch["answers"][j]
                        ),
                    }
                    for j in range(len(generated_texts))
                ]
            for j, row in enumerate(rows):
                token_ids = out[j, -gen_length:].tolist()
                row.update({
                    "steps": steps_taken[j],
                    "wall_time": wall_time_per_sample,
                    **selected_set_summary(selected_set_sizes[j]),
                    "generated_token_ids": token_ids,
                    "remaining_masks": token_ids.count(mask_id),
                    "complete": mask_id not in token_ids,
                    "generation_seconds": generation_seconds / len(rows),
                    "effective_tokens": sum(t not in tokenizer.all_special_ids for t in token_ids),
                    "learn2pd_counted_tokens": counted_tokens[j],
                })
            all_generations.extend(rows)

    effective_tokens = sum(row["effective_tokens"] for row in all_generations)
    generation_seconds = sum(row["generation_seconds"] for row in all_generations)
    wrapper_seconds = time.perf_counter() - wrapper_start
    return {
        "generations": all_generations,
        "throughput": {
            "effective_tokens": effective_tokens,
            "generation_seconds": generation_seconds,
            "sampler_tokens_per_second": effective_tokens / generation_seconds if generation_seconds else None,
            "token_definition": "generated token ids excluding tokenizer special ids",
            "sampler_timing_scope": "CUDA-synchronized sampler, including policy; excluding tokenization, decode, and scoring",
            "learn2pd_wrapper_tokens": counted_tokens_total,
            "learn2pd_wrapper_seconds": wrapper_seconds,
            "learn2pd_wrapper_tokens_per_second": counted_tokens_total / wrapper_seconds if wrapper_seconds else None,
        },
    }


def result_path(args, model_name):
    parts = [args.dataset, model_name, args.gen_length, args.diffusion_steps,
             args.block_length, args.remasking, 0, "generations"]
    return Path(args.output_dir) / ("_".join(map(str, parts)) + ".json")


def save_results(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(results, handle, indent=2, sort_keys=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved results to {path}")


def load_backbone(model_path, revision, kv_cache):
    """Load LLaDA (optionally with Fast-dLLM KV-cache modeling) or Dream."""
    if "Dream" in model_path:
        if kv_cache != "no_cache":
            raise ValueError("the prefix KV cache supports LLaDA only")
        from huggingface_hub import snapshot_download

        local = Path(model_path) if Path(model_path).is_dir() else Path(
            snapshot_download(model_path, revision=revision)
        )
        model = AutoModel.from_pretrained(
            local, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        # Dream's remote from_pretrained cannot return loading info.
        index = json.loads((local / "model.safetensors.index.json").read_text())
        if set(index["weight_map"]) != set(model.state_dict()):
            raise ValueError("Dream checkpoint architecture mismatch")
        if getattr(model.config, "model_type", "").lower() != "dream":
            raise ValueError("Dream model path did not load a Dream model")
        return model, "Dream", str(local)
    model_class = AutoModel
    if kv_cache != "no_cache":
        # Fast-dLLM v1 modeling with KV-cache support; same weights and config.
        from common.models.modeling_llada_fastdllm import LLaDAModelLM as model_class
    model, loading_info = model_class.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        revision=revision, output_loading_info=True,
    )
    if loading_info["missing_keys"] or loading_info["unexpected_keys"]:
        raise ValueError(f"Checkpoint architecture mismatch: {loading_info}")
    if getattr(model.config, "model_type", "").lower() != "llada":
        raise ValueError("expected a LLaDA or Dream model")
    return model, "LLaDA", LLADA_TOKENIZER


def build_dataset(args, tokenizer):
    kwargs = {"tokenizer": tokenizer, "num_examples": args.few_shot}
    if args.eval_protocol in FOUR_SHOT_PROTOCOLS:
        kwargs.update(prompt_protocol=args.eval_protocol, fewshot_seed=1234)
    if args.dataset == "mbpp":
        kwargs["dataset_config"] = "sanitized"
    return DATASET_MAP[args.dataset](**kwargs), kwargs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASET_MAP), required=True)
    parser.add_argument("--eval_protocol", choices=PROTOCOLS, default=MATH500_TPS_PROTOCOL)
    parser.add_argument("--remasking", choices=["policy", "fastdllm", "eb_sampler", "low_confidence"],
                        required=True)
    parser.add_argument("--config", default=None,
                        help="Policy training config (required for --remasking policy).")
    parser.add_argument("--policy_path", default=None,
                        help="Policy directory (with model.safetensors) or safetensors file.")
    parser.add_argument("--model_path", default=None,
                        help="Backbone (default: the policy config's model_path, else LLaDA-8B-Instruct).")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--policy_accept_threshold", type=float, default=None,
                        help="Commit positions with sigmoid(policy logit) > threshold.")
    parser.add_argument("--thres", type=float, default=None, help="Fast-dLLM confidence threshold.")
    parser.add_argument("--eb_gamma", type=float, default=None, help="EB-Sampler entropy budget.")
    parser.add_argument("--eb_error_proxy", choices=["confidence", "entropy"], default="confidence")
    parser.add_argument("--diffusion_steps", type=int, default=0,
                        help="Steps for low_confidence (= gen_length for one-by-one decoding).")
    parser.add_argument("--kv_cache", choices=["no_cache", "prefix_cache"], default="no_cache")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--few_shot", type=int, default=-1)
    parser.add_argument("--truncate_prompt", type=int, default=None,
                        help="Keep only the last N prompt tokens (zero-shot protocols). "
                             "Default: full prompts, as in the paper's evaluation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--test_offset", type=int, default=0)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--defer_scoring", action="store_true",
                        help="Save unscored generations for python -m eval.scoring.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure(args)
    init_seed(args.seed)

    config = None
    if args.remasking == "policy":
        if not args.config or not args.policy_path:
            raise ValueError("--remasking policy requires --config and --policy_path")
        (config,) = TrlParser((Config,)).parse_args_and_config(
            args=["--config", args.config], fail_with_unknown_args=False
        )
    if args.remasking == "fastdllm" and args.thres is None:
        raise ValueError("--thres is required for Fast-dLLM")
    if args.remasking == "eb_sampler" and (args.eb_gamma is None or args.eb_gamma < 0):
        raise ValueError("--eb_gamma must be nonnegative for EB-Sampler")
    if args.remasking == "low_confidence" and not 0 < args.diffusion_steps <= args.gen_length:
        raise ValueError("--diffusion_steps must be in (0, gen_length] for low_confidence")
    if args.model_path is None:
        args.model_path = config.model_path if config is not None else LLADA_TOKENIZER
        if config is not None and args.model_revision is None:
            args.model_revision = config.model_revision

    args.output_dir = str(Path(args.output_dir) / args.eval_protocol)
    suffix = f"{args.suffix}_{args.eval_protocol}_seed{args.seed}"
    model_name = "instruct" + (f"_fs{args.few_shot}" if args.few_shot > 0 else "") + f"_{suffix}"
    path = result_path(args, model_name)
    if path.exists():
        raise FileExistsError(f"{path} exists; choose a new output_dir or suffix")
    source_provenance = code_provenance()

    model, model_type, tokenizer_path = load_backbone(args.model_path, args.model_revision, args.kv_cache)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    mask_id = MASK_TOKENS_MAP[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    policy = None
    generation_kwargs = {
        "remasking": args.remasking,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "model_type": model_type,
        "kv_cache": args.kv_cache,
    }
    if args.remasking == "policy":
        policy = build_policy(config, model.config.hidden_size)
        policy_file = Path(args.policy_path)
        if policy_file.is_dir():
            policy_file = policy_file / "model.safetensors"
        policy.load_state_dict(load_file(str(policy_file)))
        policy = policy.to(device).eval()
        generation_kwargs.update(
            policy=policy, sampling_mode="bernoulli-deterministic",
            policy_accept_threshold=args.policy_accept_threshold,
            confidences_top_p=config.confidences_top_p,
        )
    elif args.remasking == "fastdllm":
        generation_kwargs["thres"] = args.thres
    elif args.remasking == "eb_sampler":
        generation_kwargs.update(eb_gamma=args.eb_gamma, eb_error_proxy=args.eb_error_proxy)
    else:
        generation_kwargs["steps"] = args.diffusion_steps

    dataset, dataset_kwargs = build_dataset(args, tokenizer)
    full_dataset_size = len(dataset)
    if full_dataset_size != DATASET_SIZES[args.dataset]:
        raise ValueError("Unexpected benchmark size; verify the cached evaluation dataset")
    collate_fn = dataset.collate_fn
    prompt_truncation_tokens = None
    if (args.eval_protocol in {MATH500_TPS_PROTOCOL, LENGTH_TRANSFER_PROTOCOL}
            and args.truncate_prompt is not None):
        prompt_truncation_tokens = args.truncate_prompt
        base_collate_fn = collate_fn

        def collate_fn(batch):
            values = base_collate_fn(batch)
            values["input_ids"] = values["input_ids"][:, -args.truncate_prompt:]
            values["attention_mask"] = values["attention_mask"][:, -args.truncate_prompt:]
            # Record the actual model input, not the pre-truncation display prompt.
            values["prompts"] = tokenizer.batch_decode(values["input_ids"], skip_special_tokens=False)
            return values
    if not 0 <= args.test_offset < full_dataset_size:
        raise ValueError(f"test_offset must be in [0, {full_dataset_size})")
    test_stop = full_dataset_size if args.n_test is None else min(
        full_dataset_size, args.test_offset + args.n_test
    )
    if args.test_offset or test_stop < full_dataset_size:
        dataset = torch.utils.data.Subset(dataset, range(args.test_offset, test_stop))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Seed sampling after all loading/initialization is complete.
    init_seed(args.seed)
    results = evaluate(
        model, tokenizer, dataloader, args.dataset,
        generation_kwargs=generation_kwargs, gen_length=args.gen_length, mask_id=mask_id,
    )

    if args.defer_scoring:
        results["scoring_status"] = {"state": "deferred", "mode": "cpu_pipeline"}
    elif args.dataset in {"gsm8k", "math"}:
        results = score_results(results, protocol=args.eval_protocol, dataset=args.dataset)
    else:
        results["code_eval_results"] = evaluate_code(results["generations"], args.dataset)
    results["metrics_complete_rate"] = (
        sum(r["complete"] for r in results["generations"]) / len(results["generations"])
    )
    results.update({
        "eval_protocol": args.eval_protocol,
        "prompt_truncation_tokens": prompt_truncation_tokens,
        "source_provenance": source_provenance,
        "evaluation_arguments": vars(args),
        "policy_configuration": (
            {k: v for k, v in vars(config).items() if k.startswith("policy_")} if config else None
        ),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest() if args.config else None,
        "policy_parameter_count": sum(p.numel() for p in policy.parameters()) if policy else 0,
        "model_dtype": str(model.dtype),
        "dataset_config": dataset_kwargs.get("dataset_config"),
        "full_dataset_size": full_dataset_size,
        "evaluated_dataset_size": len(dataset),
        "partial_evaluation": len(dataset) != full_dataset_size,
    })
    save_results(results, path)
    print(f"Processed {len(results['generations'])}/{len(dataset)} examples of {args.dataset}")


if __name__ == "__main__":
    main()
