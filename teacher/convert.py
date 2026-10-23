"""Convert measured G_trace shards into compact offline teacher examples.

The measurement artifacts are very large, single-object JSON files.  This
converter streams the ``samples`` array instead of loading an entire shard in
memory and writes one Parquet row per trajectory.  Optional soft commit-budget
state graph can be retained so training samples M and constructs its teacher
trajectory online without extra denoiser forwards.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq

TRANSITION_TYPE = pa.struct(
    [
        ("source_step", pa.int32()),
        ("target_step", pa.int32()),
        ("active_block", pa.int16()),
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
        ("mask_bitmap", pa.list_(pa.bool_())),
        ("eligible_positions", pa.list_(pa.int32())),
        ("target_commit_positions", pa.list_(pa.int32())),
        ("target_action_sequence", pa.list_(pa.int32())),
        ("target_token_ids", pa.list_(pa.int32())),
        ("target_confidences", pa.list_(pa.float32())),
    ]
)

ARTIFACT_STEP_TYPE = pa.struct(
    [
        ("source_step", pa.int32()),
        ("active_block", pa.int16()),
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
        ("mask_bitmap", pa.list_(pa.bool_())),
        ("eligible_positions", pa.list_(pa.int32())),
        ("top1_token_ids", pa.list_(pa.int32())),
        ("top1_probabilities", pa.list_(pa.float32())),
    ]
)


TEACHER_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("source_shard", pa.int16()),
        ("sample_id", pa.string()),
        ("prompt_length", pa.int32()),
        ("generation_length", pa.int32()),
        ("block_length", pa.int32()),
        ("training_math_protocol", pa.string()),
        ("math_training_invalid_gold", pa.string()),
        ("math_protocol_metadata", pa.string()),
        ("transitions", pa.list_(TRANSITION_TYPE)),
        ("max_reach", pa.list_(pa.int32())),
        ("artifact_steps", pa.list_(ARTIFACT_STEP_TYPE)),
        ("terminal_mask_bitmap", pa.list_(pa.bool_())),
    ]
)


def _find_samples_array(handle: TextIO, chunk_size: int) -> str:
    """Consume through the opening ``samples`` array and return trailing bytes."""
    needle = '"samples"'
    buffer = ""
    while chunk := handle.read(chunk_size):
        buffer += chunk
        key = buffer.find(needle)
        if key >= 0:
            opening = buffer.find("[", key + len(needle))
            if opening >= 0:
                return buffer[opening + 1 :]
        # Retain enough overlap for a key split across chunks.
        buffer = buffer[-(len(needle) + 64) :]
    raise ValueError("G_trace artifact has no samples array")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_json_samples(path: Path, chunk_size: int = 8 << 20) -> Iterator[dict]:
    """Yield objects from the top-level ``samples`` array with bounded memory."""
    decoder = json.JSONDecoder()
    with _open_text(path) as handle:
        buffer = _find_samples_array(handle, chunk_size)
        eof = False
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    if eof:
                        raise ValueError(f"truncated samples array in {path}")
                    eof = True
                else:
                    buffer += chunk
                    continue
                raise ValueError(f"truncated JSON object in {path}")
            if not isinstance(value, dict):
                raise TypeError(f"expected sample object in {path}, got {type(value)}")
            yield value
            buffer = buffer[end:]


def read_artifact_metadata(path: Path, chunk_size: int = 1 << 20) -> dict[str, Any]:
    """Read metadata preceding ``samples`` without touching the large array."""
    needle = '"samples"'
    with _open_text(path) as handle:
        prefix = ""
        while chunk := handle.read(chunk_size):
            prefix += chunk
            key = prefix.find(needle)
            if key >= 0:
                # Turn the prefix into a complete object with an empty samples array.
                synthetic = prefix[:key] + '"samples": []}'
                return json.loads(synthetic)
    raise ValueError(f"G_trace artifact has no samples field: {path}")


def _state_at(sample: dict, step: int) -> dict:
    steps = sample["steps"]
    if step == len(steps):
        return sample["terminal_state"]
    return steps[step]["state_before_forward"]


def _sample_max_reach(sample: dict) -> list[int]:
    if "max_reach" in sample:
        return [int(value) for value in sample["max_reach"]]
    reaches = [index for index in range(len(sample["steps"]))]
    for source, target in sample.get("admissible_edges", []):
        source = int(source)
        target = int(target)
        if 0 <= source < len(reaches):
            reaches[source] = max(reaches[source], target)
    if any(reach <= source for source, reach in enumerate(reaches)):
        raise ValueError(
            f"sample {sample['sample_id']} lacks a complete exact-rejoin graph"
        )
    return reaches


def teacher_transitions_for_path(
    sample: dict, metadata: dict, path: list[int] | tuple[int, ...]
) -> Iterator[dict]:
    """Teacher transitions along an exact-rejoin path (commits in descending confidence)."""
    steps = sample["steps"]
    generation_length = int(metadata["generation_length"])
    block_length = int(metadata["block_length"])
    for source_step, target_step in zip(path, path[1:]):
        source_record = steps[source_step]
        source = source_record["state_before_forward"]
        target = _state_at(sample, target_step)
        prompt_length = len(source["token_ids"]) - generation_length
        committed_full = [
            position
            for position, (masked, target_masked) in enumerate(
                zip(source["mask_bitmap"], target["mask_bitmap"])
            )
            if masked and not target_masked
        ]
        if not committed_full:
            raise ValueError(
                f"empty compressed edge {source_step}->{target_step} in "
                f"sample {sample['sample_id']}"
            )
        confidences = source_record["top1_probability_per_position"]
        committed_full.sort(key=lambda position: (-confidences[position], position))
        committed = [position - prompt_length for position in committed_full]
        if not all(0 <= position < generation_length for position in committed):
            raise ValueError(f"commit position outside response in {sample['sample_id']}")
        expected_tokens = source_record["top1_token_id_per_position"]
        eligible = [
            position - prompt_length
            for position in source_record["eligible_positions"]
        ]
        yield {
            "source_step": source_step,
            "target_step": target_step,
            "active_block": int(source["active_block"]),
            "input_ids": source["token_ids"],
            "attention_mask": source["attention_mask"],
            "mask_bitmap": source["mask_bitmap"],
            "eligible_positions": eligible,
            "target_commit_positions": committed,
            # Ordered commit sequence terminated by a STOP index (= response length).
            "target_action_sequence": committed + [generation_length],
            "target_token_ids": [expected_tokens[position] for position in committed_full],
            "target_confidences": [confidences[position] for position in committed_full],
        }


def teacher_transitions(
    sample: dict, metadata: dict, original_path: bool = False
) -> Iterator[dict]:
    """Build teachers for the minimum-NFE exact-rejoin path."""
    path = (
        list(range(len(sample["steps"]) + 1))
        if original_path
        else sample["compressed_path"]
    )
    yield from teacher_transitions_for_path(sample, metadata, path)


def teacher_sample_row(
    sample: dict,
    metadata: dict,
    source_shard: int,
    include_online_graph: bool = False,
    original_path: bool = False,
) -> dict:
    """Return one training row containing the sample's complete teacher trajectory."""
    generation_length = int(metadata["generation_length"])
    first_state = sample["steps"][0]["state_before_forward"]
    prompt_length = len(first_state["token_ids"]) - generation_length
    transitions = list(teacher_transitions(sample, metadata, original_path))
    if not transitions:
        raise ValueError(f"sample {sample['sample_id']} has no teacher transitions")
    return {
        "schema_version": 3 if include_online_graph else 2,
        "dataset": metadata["dataset"],
        "split": metadata.get("split", "test"),
        "source_shard": source_shard,
        "sample_id": str(sample["sample_id"]),
        "prompt_length": prompt_length,
        "generation_length": generation_length,
        "block_length": int(metadata["block_length"]),
        "training_math_protocol": metadata.get("training_math_protocol", "legacy"),
        "math_training_invalid_gold": metadata.get(
            "math_training_invalid_gold", "error"
        ),
        "math_protocol_metadata": json.dumps(
            metadata.get("math_protocol_metadata", {"name": "legacy"}),
            sort_keys=True,
        ),
        "transitions": transitions,
        "max_reach": _sample_max_reach(sample) if include_online_graph else [],
        "artifact_steps": (
            [
                {
                    "source_step": index,
                    "active_block": int(
                        record["state_before_forward"]["active_block"]
                    ),
                    "input_ids": record["state_before_forward"]["token_ids"],
                    "attention_mask": record["state_before_forward"]["attention_mask"],
                    "mask_bitmap": record["state_before_forward"]["mask_bitmap"],
                    "eligible_positions": [
                        position - prompt_length
                        for position in record["eligible_positions"]
                    ],
                    "top1_token_ids": record["top1_token_id_per_position"],
                    "top1_probabilities": record[
                        "top1_probability_per_position"
                    ],
                }
                for index, record in enumerate(sample["steps"])
            ]
            if include_online_graph
            else []
        ),
        "terminal_mask_bitmap": (
            sample["terminal_state"]["mask_bitmap"] if include_online_graph else []
        ),
    }


def convert_shard(
    input_path: Path,
    output_path: Path,
    row_group_size: int = 1024,
    include_online_graph: bool = False,
    original_path: bool = False,
) -> tuple[int, int]:
    metadata = read_artifact_metadata(input_path)
    source_shard = int(metadata.get("shard_index", 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = row_count = 0
    writer = pq.ParquetWriter(output_path, TEACHER_SCHEMA, compression="zstd")
    batch: list[dict] = []
    try:
        for sample in iter_json_samples(input_path):
            sample_count += 1
            row = teacher_sample_row(
                sample, metadata, source_shard, include_online_graph, original_path
            )
            batch.append(row)
            row_count += len(row["transitions"])
            if len(batch) >= row_group_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=TEACHER_SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=TEACHER_SCHEMA))
    finally:
        writer.close()
    return sample_count, row_count


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for input_path in args.inputs:
        stem = input_path.name.removesuffix(".gz").removesuffix(".json")
        output_path = args.output_dir / f"{stem}.parquet"
        samples, rows = convert_shard(
            input_path,
            output_path,
            args.row_group_size,
            args.include_online_graph,
            args.original_path,
        )
        manifest.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "samples": samples,
                "rows": rows,
                "include_online_graph": args.include_online_graph,
            }
        )
        print(f"converted {samples} samples / {rows} decisions to {output_path}")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument(
        "--include-online-graph",
        action="store_true",
        help="Retain original states and max_reach for online M sampling.",
    )
    parser.add_argument(
        "--original-path",
        action="store_true",
        help="Store every original Fast-dLLM step instead of G_trace jumps.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
