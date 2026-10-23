"""Filter offline G_trace teacher Parquet files to correct terminal outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.math_protocol import score_answer
from teacher.convert import iter_json_samples
from teacher.convert import read_artifact_metadata


def correct_ids(raw_path: Path) -> tuple[str, set[str]]:
    """Sample ids whose Fast-dLLM final answer is correct (common-math-v1 scorer)."""
    metadata = read_artifact_metadata(raw_path)
    dataset = str(metadata["dataset"])
    if metadata.get("training_math_protocol") != "common-math-v1":
        raise ValueError(f"{raw_path} was not collected with common-math-v1")
    return dataset, {
        str(sample["sample_id"])
        for sample in iter_json_samples(raw_path)
        if score_answer(sample["final_text"], sample["reference_answer"])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    manifest = []
    for raw_path in sorted(args.raw_dir.glob("*_train_full_shard*.json*")):
        dataset, selected = correct_ids(raw_path)
        # Same stem rule as teacher.convert (handles .json.gz).
        stem = raw_path.name.removesuffix(".gz").removesuffix(".json")
        input_path = args.input_dir / f"{stem}.parquet"
        table = pq.read_table(input_path)
        keep = [
            row_dataset == dataset and sample_id in selected
            for row_dataset, sample_id in zip(
                table.column("dataset").to_pylist(),
                table.column("sample_id").to_pylist(),
            )
        ]
        filtered = table.filter(pa.array(keep))
        output_path = args.output_dir / input_path.name
        pq.write_table(filtered, output_path, compression="zstd")
        total += filtered.num_rows
        manifest.append(
            {
                "input": str(input_path),
                "output": str(output_path),
                "dataset": dataset,
                "samples": filtered.num_rows,
            }
        )
        print(f"{output_path}: {filtered.num_rows}", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"total correct trajectories: {total}")


if __name__ == "__main__":
    main()
