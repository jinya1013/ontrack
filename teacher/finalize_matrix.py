"""Validate the matched teacher datasets and create correct-only views.

The four cells of the teacher matrix share prompts and Fast-dLLM terminal
responses; each ``<cell>_correct`` directory keeps the trajectories whose
response is correct.  ``--variants student_rejoin`` finalizes only the
student-state / exact-rejoin teacher (used for Dream, where the
teacher-terminal cell is not collected).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from teacher.convert import TEACHER_SCHEMA

VARIANTS = ("student_terminal", "student_rejoin", "teacher_terminal", "teacher_rejoin")


def identities(paths: list[Path]) -> tuple[set[tuple[str, str]], Counter]:
    observed: set[tuple[str, str]] = set()
    transitions: Counter = Counter()
    for path in paths:
        table = pq.read_table(path, columns=["dataset", "sample_id", "transitions"])
        datasets = table["dataset"].to_pylist()
        sample_ids = table["sample_id"].to_pylist()
        lengths = pc.list_value_length(table["transitions"]).to_pylist()
        for dataset, sample_id, length in zip(datasets, sample_ids, lengths):
            key = (dataset, str(sample_id))
            if key in observed:
                raise ValueError(f"duplicate sample {key} in {path}")
            observed.add(key)
            transitions[dataset] += int(length)
    return observed, transitions


def normalize_teacher_terminal(paths: list[Path]) -> None:
    """Use the same confidence-descending action order as the other cells."""
    for path in paths:
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            stop = int(row["generation_length"])
            for transition in row["transitions"]:
                positions = transition["target_commit_positions"]
                tokens = transition["target_token_ids"]
                confidence = transition["target_confidences"]
                if len(positions) != len(tokens) or len(positions) != len(confidence):
                    raise ValueError(f"incomplete teacher action in {path}")
                order = sorted(
                    range(len(positions)),
                    key=lambda index: (-confidence[index], positions[index]),
                )
                transition["target_commit_positions"] = [positions[index] for index in order]
                transition["target_token_ids"] = [tokens[index] for index in order]
                transition["target_confidences"] = [confidence[index] for index in order]
                transition["target_action_sequence"] = (
                    transition["target_commit_positions"] + [stop]
                )
        temporary = path.with_suffix(path.suffix + ".normalized.tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=TEACHER_SCHEMA),
            temporary,
            compression="zstd",
        )
        temporary.replace(path)


def first_actions(paths: list[Path]) -> dict[tuple[str, str], tuple]:
    result = {}
    for path in paths:
        table = pq.read_table(path, columns=["dataset", "sample_id", "transitions"])
        datasets = table["dataset"].to_pylist()
        sample_ids = table["sample_id"].to_pylist()
        first_rows = pc.list_element(table["transitions"], 0).to_pylist()
        for dataset, sample_id, first in zip(datasets, sample_ids, first_rows):
            result[(dataset, str(sample_id))] = (
                first["input_ids"],
                first["mask_bitmap"],
                first["eligible_positions"],
                first["target_commit_positions"],
                first["target_token_ids"],
            )
    return result


def finalize(args: argparse.Namespace) -> None:
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    correct_paths = sorted(args.correct_source.glob("*.parquet"))
    if not correct_paths:
        raise FileNotFoundError(f"no correct-source Parquet files in {args.correct_source}")
    correct_ids, _ = identities(correct_paths)
    variants = tuple(args.variants)
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"variants must be among {VARIANTS}")
    paths = {
        variant: sorted((args.root / variant).glob("*.parquet"))
        for variant in variants
    }
    if any(not group for group in paths.values()):
        raise FileNotFoundError("one or more teacher variants have no Parquet files")
    if "teacher_terminal" in paths:
        normalize_teacher_terminal(paths["teacher_terminal"])
    ids_and_transitions = {name: identities(group) for name, group in paths.items()}
    reference = ids_and_transitions[variants[0]][0]
    for name, (ids, _) in ids_and_transitions.items():
        if ids != reference:
            raise ValueError(f"{name} identities differ from {variants[0]}: missing={len(reference-ids)} extra={len(ids-reference)}")
    if not correct_ids <= reference:
        raise ValueError(f"correct-source IDs outside matrix: {len(correct_ids-reference)}")
    expected = sum(manifest["counts"][variants[0]].values())
    if len(reference) != expected:
        raise ValueError(f"expected {expected} identities, found {len(reference)}")
    if {"student_terminal", "teacher_terminal"} <= set(paths) and (
        first_actions(paths["student_terminal"]) != first_actions(paths["teacher_terminal"])
    ):
        raise ValueError("student and teacher terminal labels disagree at the common initial states")
    for name, group in paths.items():
        output = args.root / f"{name}_correct"
        output.mkdir(exist_ok=False)
        for path in group:
            table = pq.read_table(path)
            keep = [
                (dataset, str(sample_id)) in correct_ids
                for dataset, sample_id in zip(
                    table["dataset"].to_pylist(), table["sample_id"].to_pylist()
                )
            ]
            filtered = table.filter(pa.array(keep))
            pq.write_table(filtered, output / path.name, compression="zstd")
    manifest["complete"] = True
    manifest["full_samples"] = len(reference)
    manifest["correct_samples"] = len(correct_ids)
    manifest["full_transitions"] = {
        name: dict(counts) for name, (_, counts) in ids_and_transitions.items()
    }
    manifest["correct_source"] = str(args.correct_source)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "full_samples": len(reference), "correct_samples": len(correct_ids),
        "full_transitions": manifest["full_transitions"],
    }), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--correct-source", type=Path, required=True)
    result.add_argument("--variants", nargs="+", default=list(VARIANTS))
    return result


if __name__ == "__main__":
    finalize(parser().parse_args())
