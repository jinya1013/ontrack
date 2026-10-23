"""Build a matched 2x2 teacher matrix from recorded Fast-dLLM traces.

The two axes are supervision states (student or teacher trajectory) and oracle
actions (terminal-token agreement or shortest-NFE exact rejoin). All four cells
use the same Fast-dLLM terminal response and selected question identities.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from teacher.convert import (
    TEACHER_SCHEMA,
    iter_json_samples,
    read_artifact_metadata,
    teacher_sample_row,
    teacher_transitions_for_path,
)
from teacher.rejoin import shortest_suffix_first_jumps

VARIANTS = ("student_terminal", "student_rejoin", "teacher_rejoin")


def selected_ids(path: Path) -> dict[str, set[str]]:
    manifest = json.loads(path.read_text())
    return {
        dataset: {f"train:{int(index)}" for index in indices}
        for dataset, indices in manifest["dataset_indices"].items()
    }


def terminal_transition(sample: dict, metadata: dict, source_step: int) -> dict:
    """Label one observed student state by final-token agreement."""
    step = sample["steps"][source_step]
    source = step["state_before_forward"]
    final = sample["terminal_state"]
    generation_length = int(metadata["generation_length"])
    prompt_length = len(source["token_ids"]) - generation_length
    eligible = [int(position) for position in step["eligible_positions"]]
    top1 = step["top1_token_id_per_position"]
    confidence = step["top1_probability_per_position"]
    committed = [
        position for position in eligible
        if source["mask_bitmap"][position]
        and not final["mask_bitmap"][position]
        and top1[position] == final["token_ids"][position]
    ]
    committed.sort(key=lambda position: (-confidence[position], position))
    relative = [position - prompt_length for position in committed]
    return {
        "source_step": source_step,
        "target_step": source_step + 1,
        "active_block": int(source["active_block"]),
        "input_ids": source["token_ids"],
        "attention_mask": source["attention_mask"],
        "mask_bitmap": source["mask_bitmap"],
        "eligible_positions": [position - prompt_length for position in eligible],
        "target_commit_positions": relative,
        "target_action_sequence": relative + [generation_length],
        "target_token_ids": [top1[position] for position in committed],
        "target_confidences": [confidence[position] for position in committed],
    }


def matrix_rows(sample: dict, metadata: dict) -> dict[str, dict]:
    """Return three CPU-only cells with identical prompt and terminal response."""
    base = teacher_sample_row(sample, metadata, int(metadata["shard_index"]))
    reaches = [int(reach) for reach in sample["max_reach"]]
    if len(reaches) != len(sample["steps"]):
        raise ValueError(f"incomplete rejoin graph for {sample['sample_id']}")
    _, first = shortest_suffix_first_jumps(reaches)
    teacher_path = [0]
    while teacher_path[-1] < len(first):
        teacher_path.append(first[teacher_path[-1]])
    if len(teacher_path) != len(sample["compressed_path"]):
        raise ValueError(f"shortest-NFE path length mismatch for {sample['sample_id']}")
    student_terminal = [
        terminal_transition(sample, metadata, index)
        for index in range(len(sample["steps"]))
    ]
    student_rejoin = [
        next(teacher_transitions_for_path(sample, metadata, (index, target)))
        for index, target in enumerate(first)
    ]
    variants = {
        "student_terminal": student_terminal,
        "student_rejoin": student_rejoin,
        "teacher_rejoin": list(teacher_transitions_for_path(sample, metadata, teacher_path)),
    }
    return {name: {**base, "transitions": transitions} for name, transitions in variants.items()}


def prepare(args: argparse.Namespace) -> None:
    selected = selected_ids(args.selection_manifest) if args.selection_manifest else None
    args.output.mkdir(parents=True, exist_ok=False)
    for name in VARIANTS:
        (args.output / name).mkdir()
    prompt_dir = args.output / "teacher_terminal_inputs"
    prompt_dir.mkdir()
    seen: dict[str, set[str]] = {}
    expected_from_sources: dict[str, set[str]] = {}
    source_shards: dict[str, set[int]] = {}
    shard_counts: dict[str, int] = {}
    counts = {name: {} for name in VARIANTS}
    transition_counts = {name: 0 for name in VARIANTS}
    buffers = {name: [] for name in VARIANTS}
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                pq.ParquetWriter(
                    args.output / name / "teacher.parquet", TEACHER_SCHEMA,
                    compression="zstd",
                )
            )
            for name in VARIANTS
        }
        prompt_files = [
            stack.enter_context((prompt_dir / f"shard{index}.jsonl").open("w"))
            for index in range(args.teacher_shards)
        ]
        for raw_path in sorted(args.raw_dir.glob("*_train_full_shard*.json*")):
            metadata = read_artifact_metadata(raw_path)
            dataset = metadata["dataset"]
            if selected is not None and dataset not in selected:
                continue
            seen.setdefault(dataset, set())
            expected_from_sources.setdefault(dataset, set()).update(
                f"train:{int(index)}" for index in metadata["dataset_indices"]
            )
            source_shards.setdefault(dataset, set()).add(int(metadata["shard_index"]))
            shard_counts[dataset] = int(metadata["num_shards"])
            for sample in iter_json_samples(raw_path):
                sample_id = str(sample["sample_id"])
                if selected is not None and sample_id not in selected[dataset]:
                    continue
                if sample_id in seen[dataset]:
                    raise ValueError(f"duplicate selected sample {dataset}:{sample_id}")
                seen[dataset].add(sample_id)
                if any(sample["terminal_state"]["mask_bitmap"]):
                    raise ValueError(f"incomplete terminal response: {sample_id}")
                rows = matrix_rows(sample, metadata)
                for name, row in rows.items():
                    buffers[name].append(row)
                    if len(buffers[name]) >= 64:
                        writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=TEACHER_SCHEMA))
                        buffers[name].clear()
                    counts[name][dataset] = counts[name].get(dataset, 0) + 1
                    transition_counts[name] += len(row["transitions"])
                first = sample["steps"][0]["state_before_forward"]
                prompt_length = rows["teacher_rejoin"]["prompt_length"]
                ordinal = int(sample_id.split(":")[-1])
                prompt_shard = ordinal % args.teacher_shards
                prompt_files[prompt_shard].write(json.dumps({
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "source_shard": int(metadata["shard_index"]),
                    "prompt_ids": first["token_ids"][:prompt_length],
                    "prompt_attention_mask": first["attention_mask"][:prompt_length],
                    "terminal_token_ids": sample["terminal_state"]["token_ids"][prompt_length:],
                    "generation_length": int(metadata["generation_length"]),
                    "block_length": int(metadata["block_length"]),
                    "training_math_protocol": metadata["training_math_protocol"],
                    "math_training_invalid_gold": metadata["math_training_invalid_gold"],
                    "math_protocol_metadata": metadata["math_protocol_metadata"],
                }) + "\n")
            print(f"scanned {raw_path.name}", flush=True)
        for name in VARIANTS:
            if buffers[name]:
                writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=TEACHER_SCHEMA))
    if selected is not None and seen != selected:
        missing = {name: len(selected[name] - seen[name]) for name in selected}
        raise RuntimeError(f"missing selected trajectories: {missing}")
    if selected is None:
        for dataset, observed in seen.items():
            if source_shards[dataset] != set(range(shard_counts[dataset])):
                raise RuntimeError(f"missing source shards for {dataset}")
            if observed != expected_from_sources[dataset]:
                raise RuntimeError(f"missing source trajectories for {dataset}")
    manifest = {
        "schema_version": 1,
        "complete": False,
        "selection_manifest": str(args.selection_manifest) if args.selection_manifest else None,
        "raw_dir": str(args.raw_dir),
        "teacher_shards": args.teacher_shards,
        "counts": counts,
        "transitions": transition_counts,
        "action_definitions": {
            "terminal": "eligible source top1 equals Fast-dLLM terminal token",
            "rejoin": "first edge on a shortest-NFE exact-rejoin suffix; farthest tie break",
        },
        "state_definitions": {
            "student": "every recorded Fast-dLLM state",
            "teacher_rejoin": "recorded states on the shortest-NFE path",
            "teacher_terminal": "rollout under terminal-match oracle with reference fallback",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": counts, "transitions": transition_counts}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--selection-manifest", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--teacher-shards", type=int, default=128)
    return result


if __name__ == "__main__":
    prepare(parser().parse_args())
