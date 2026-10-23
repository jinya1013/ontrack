"""Collect terminal-match oracle states using Fast-dLLM terminal responses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel

from teacher.convert import TEACHER_SCHEMA
from teacher.learn2pd_oracle import collect_egp_teacher_rollout, offline_teacher_row


def collect(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-state collection requires a GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    model_load_seconds = time.perf_counter() - load_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    count = transition_count = 0
    timings = []
    writer = pq.ParquetWriter(temporary, TEACHER_SCHEMA, compression="zstd")
    try:
        for input_path in args.input:
            with input_path.open() as source:
                lines = list(source)
            for line in lines:
                example = json.loads(line)
                sample_started = time.perf_counter()
                prompt = torch.tensor([example["prompt_ids"]], device="cuda")
                attention = torch.tensor(
                    [example["prompt_attention_mask"]], device="cuda"
                )
                reference = torch.tensor(
                    [example["terminal_token_ids"]], device="cuda"
                )
                rollout = collect_egp_teacher_rollout(
                    model, prompt,
                    attention_mask=attention,
                    reference_tokens=reference,
                    gen_length=int(example["generation_length"]),
                    block_length=int(example["block_length"]),
                    temperature=0.0,
                    model_type="LLaDA",
                )
                row = offline_teacher_row(
                    rollout,
                    dataset=example["dataset"],
                    sample_id=example["sample_id"],
                    prompt_length=prompt.shape[1],
                    block_length=int(example["block_length"]),
                    source_shard=int(example["source_shard"]),
                    training_math_protocol=example["training_math_protocol"],
                    math_training_invalid_gold=example["math_training_invalid_gold"],
                    math_protocol_metadata=example["math_protocol_metadata"],
                )
                writer.write_table(pa.Table.from_pylist([row], schema=TEACHER_SCHEMA))
                timings.append({
                    "dataset": example["dataset"],
                    "sample_id": example["sample_id"],
                    "nfe": int(rollout.generation.steps_taken.item()),
                    "seconds": time.perf_counter() - sample_started,
                })
                count += 1
                transition_count += len(row["transitions"])
                if count % 10 == 0:
                    print(f"{input_path.name}: {count} samples", flush=True)
    finally:
        writer.close()
    temporary.replace(args.output)
    args.output.with_suffix(".timing.json").write_text(json.dumps({
        "model_load_seconds": model_load_seconds,
        "samples": timings,
    }, indent=2) + "\n")
    print(json.dumps({"samples": count, "transitions": transition_count}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, nargs="+", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    return result


if __name__ == "__main__":
    collect(parser().parse_args())
