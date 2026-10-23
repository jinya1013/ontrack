"""Record deterministic Fast-dLLM trajectories on training prompts (GPU).

For every prompt of one dataset shard this stores each decoding state, the
denoiser's top-1 tokens and confidences, and the exact-rejoin reach of every
state: the farthest later state of the same trajectory that is reproduced
exactly by committing, in one step, the source state's own top-1 predictions
within its active block (see ``teacher.g_trace``).  ``teacher.convert``
and ``teacher.build_matrix`` turn these traces into teacher datasets.

    python -m teacher.collect_traces --dataset gsm8k --num-shards 8 --shard-index 0 \
        --output traces/llada/gsm8k_train_full_shard0.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from teacher.g_trace import (
    FastDLLMState,
    FastDLLMStep,
    FastDLLMTrace,
    GTraceResult,
    compute_g_trace,
    summarize_g_trace,
)
from common.math_protocol import PROTOCOL, protocol_metadata
from common.generation.generation import (
    DenoiserState,
    evaluate_denoiser_sequence,
    generate_unified,
    prepare_generation_attention_mask,
    prepare_generation_token_mask,
)
from data.loaders.training_math import TrainingMathDataset

MASK_IDS = {"LLaDA": 126336, "Dream": 151666}
MASK_ID = MASK_IDS["LLaDA"]


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_gzip_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.write(value)
    temporary.replace(path)


def _result_from_dict(value: dict) -> GTraceResult:
    return GTraceResult(
        sample_id=str(value["sample_id"]),
        baseline_nfe=int(value["baseline_nfe"]),
        trace_nfe=int(value["trace_nfe"]),
        g_trace=int(value["g_trace"]),
        relative_g_trace=float(value["relative_g_trace"]),
        compressed_path=tuple(int(item) for item in value["compressed_path"]),
        max_reach=tuple(int(item) for item in value["max_reach"]),
        number_of_nontrivial_jumps=int(value["number_of_nontrivial_jumps"]),
        maximum_jump_length=int(value["maximum_jump_length"]),
        mean_jump_length=float(value["mean_jump_length"]),
    )


class ResumableArtifactWriter:
    """Write huge trajectory artifacts with bounded RAM and atomic completion.

    Each completed sample is an independently atomic gzip member.  A rerun with
    identical metadata resumes from those parts; a mismatched invocation fails
    closed instead of mixing trajectories.  The public artifact appears only
    after every expected sample is present.
    """

    def __init__(self, output: Path, metadata: dict, dataset_indices: list[int]):
        self.output = output
        self.metadata = metadata
        self.dataset_indices = dataset_indices
        if output.exists():
            raise FileExistsError(f"refusing to overwrite completed artifact: {output}")
        self.parts = output.with_name(f".{output.name}.parts")
        self.parts.mkdir(parents=True, exist_ok=True)
        identity = {"metadata": metadata, "dataset_indices": dataset_indices}
        identity_path = self.parts / "identity.json"
        if identity_path.exists():
            observed = json.loads(identity_path.read_text(encoding="utf-8"))
            if observed != identity:
                raise RuntimeError(
                    f"in-progress artifact does not match this invocation: {self.parts}"
                )
        else:
            _write_json_atomic(identity_path, identity)

    def _paths(self, local_index: int, dataset_index: int) -> tuple[Path, Path]:
        stem = f"{local_index:06d}-{dataset_index}"
        return self.parts / f"{stem}.json.gz", self.parts / f"{stem}.result.json"

    def completed_result(
        self, local_index: int, dataset_index: int
    ) -> GTraceResult | None:
        sample_path, result_path = self._paths(local_index, dataset_index)
        if not sample_path.exists() or not result_path.exists():
            return None
        value = json.loads(result_path.read_text(encoding="utf-8"))
        result = _result_from_dict(value)
        expected_id = f"{self.metadata['split']}:{dataset_index}"
        if result.sample_id != expected_id:
            raise RuntimeError(
                f"wrong sample in resumable artifact: {result.sample_id} != {expected_id}"
            )
        return result

    def append(
        self,
        local_index: int,
        dataset_index: int,
        sample: dict,
        result: GTraceResult,
    ) -> None:
        sample_path, result_path = self._paths(local_index, dataset_index)
        prefix = "" if local_index == 0 else ","
        serialized = prefix + json.dumps(
            sample, ensure_ascii=False, separators=(",", ":")
        )
        _write_gzip_text_atomic(sample_path, serialized)
        _write_json_atomic(result_path, result.to_dict())

    def finalize(self, results: list[GTraceResult]) -> None:
        if len(results) != len(self.dataset_indices):
            raise RuntimeError(
                f"cannot finalize {len(results)} of {len(self.dataset_indices)} samples"
            )
        final_metadata = {
            **self.metadata,
            "summary": summarize_g_trace(
                results,
                bootstrap_samples=int(self.metadata["bootstrap_samples"]),
                seed=int(self.metadata["seed"]),
            ),
        }
        # bootstrap_samples and seed are execution metadata used to make resume
        # identity strict; seed remains a public field and bootstrap_samples is
        # already represented in summary.
        final_metadata.pop("bootstrap_samples")
        prefix = json.dumps(
            final_metadata, ensure_ascii=False, separators=(",", ":")
        )[:-1] + ',"samples":['
        suffix = "]}\n"
        temporary = self.output.with_name(f".{self.output.name}.{os.getpid()}.tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.suffix == ".gz":
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
                handle.write(prefix)
            with temporary.open("ab") as target:
                for local_index, dataset_index in enumerate(self.dataset_indices):
                    sample_path, _ = self._paths(local_index, dataset_index)
                    with sample_path.open("rb") as source:
                        shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            with gzip.open(temporary, "at", encoding="utf-8", compresslevel=1) as handle:
                handle.write(suffix)
        else:
            with temporary.open("w", encoding="utf-8") as target:
                target.write(prefix)
                for local_index, dataset_index in enumerate(self.dataset_indices):
                    sample_path, _ = self._paths(local_index, dataset_index)
                    with gzip.open(sample_path, "rt", encoding="utf-8") as source:
                        shutil.copyfileobj(source, target)
                target.write(suffix)
                target.flush()
                os.fsync(target.fileno())
        temporary.replace(self.output)
        shutil.rmtree(self.parts)


def optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return parsed


def shard_indices(
    dataset_size: int, num_shards: int, shard_index: int, limit: int = -1
) -> list[int]:
    if dataset_size < 0:
        raise ValueError("dataset_size must be nonnegative")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard count or index")
    indices = list(range(shard_index, dataset_size, num_shards))
    return indices if limit < 0 else indices[:limit]


def resolve_model_type_and_mask_id(
    model, model_path: str, model_type: str | None, mask_id: int | None
) -> tuple[str, int]:
    """Resolve the denoiser family and check its mask token against the config.

    A wrong mask id is silent for Dream: LLaDA's 126336 is an ordinary token in
    Dream's vocabulary, so generation would run on unmasked garbage.
    """
    configured_type = str(getattr(model.config, "model_type", ""))
    if model_type is None:
        model_type = (
            "Dream" if configured_type.lower() == "dream" or "Dream" in model_path
            else "LLaDA"
        )
    if mask_id is None:
        mask_id = MASK_IDS[model_type]
    configured_mask = getattr(model.config, "mask_token_id", None)
    if configured_mask is not None and int(configured_mask) != mask_id:
        raise ValueError(
            f"mask id {mask_id} differs from model config mask_token_id {configured_mask}"
        )
    return model_type, mask_id


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _state(
    sequence: torch.Tensor,
    active_block: int,
    prompt_length: int,
    block_length: int,
    attention_mask: torch.Tensor,
    threshold: float,
    mask_id: int = MASK_ID,
) -> FastDLLMState:
    if sequence.ndim != 1:
        raise ValueError("trajectory state requires a one-dimensional token sequence")
    if attention_mask.ndim != 1 or attention_mask.numel() != sequence.numel():
        raise ValueError(
            "saved attention mask must be the one-dimensional token-validity mask"
        )
    tokens = tuple(int(value) for value in sequence.tolist())
    return FastDLLMState(
        token_ids=tokens,
        mask_bitmap=tuple(value == mask_id for value in tokens),
        active_block=active_block,
        block_length=block_length,
        prompt_length=prompt_length,
        attention_mask=tuple(int(value) for value in attention_mask.bool().flatten().tolist()),
        controller_state=(("threshold", threshold),),
    )


def record_trace(
    model,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    sample_id: str,
    threshold: float,
    generation_length: int,
    block_length: int,
    model_type: str = "LLaDA",
    mask_id: int = MASK_ID,
) -> tuple[FastDLLMTrace, torch.Tensor, torch.Tensor]:
    if prompt_ids.shape[0] != 1:
        raise ValueError("G_trace recording requires batch size one")
    prompt_length = prompt_ids.shape[1]
    full_token_mask = prepare_generation_token_mask(
        prompt_attention_mask, prompt_length + generation_length
    )
    records: list[tuple[FastDLLMState, tuple[int, ...], tuple[float, ...]]] = []

    def observe(observed: DenoiserState) -> None:
        sequence = observed.sequence[0].detach().cpu().clone()
        greedy_device = observed.greedy_tokens[0]
        confidence = torch.gather(
            observed.generation_probabilities[0], -1, greedy_device.unsqueeze(-1)
        ).squeeze(-1).detach().cpu()
        greedy = greedy_device.detach().cpu()
        top1 = sequence.clone()
        top1[prompt_length:] = greedy
        full_confidence = torch.ones(sequence.shape[0], dtype=torch.float32)
        full_confidence[prompt_length:] = confidence.float()
        records.append(
            (
                _state(
                    sequence, observed.block_index, prompt_length, block_length,
                    full_token_mask[0].detach().cpu(), threshold, mask_id,
                ),
                tuple(int(value) for value in top1.tolist()),
                tuple(float(value) for value in full_confidence.tolist()),
            )
        )

    result = generate_unified(
        model=model,
        prompt=prompt_ids,
        attention_mask=prompt_attention_mask,
        remasking="fastdllm",
        thres=threshold,
        gen_length=generation_length,
        block_length=block_length,
        temperature=0.0,
        mask_id=mask_id,
        model_type=model_type,
        state_observer=observe,
    )
    terminal = _state(
        result.sequences[0].detach().cpu(),
        (generation_length + block_length - 1) // block_length,
        prompt_length, block_length, full_token_mask[0].detach().cpu(), threshold,
        mask_id,
    )
    states = [record[0] for record in records] + [terminal]
    steps = []
    for index, (source, top1, confidence) in enumerate(records):
        target = states[index + 1]
        committed = tuple(
            position for position, (before, after) in enumerate(
                zip(source.mask_bitmap, target.mask_bitmap)
            )
            if before and not after
        )
        steps.append(FastDLLMStep(source, top1, confidence, committed))
    trace = FastDLLMTrace(sample_id, tuple(steps), terminal)
    if int(result.steps_taken.item()) != trace.num_forward_steps:
        raise RuntimeError("recorded forward count differs from generation result")
    return trace, result.sequences.detach().cpu(), full_token_mask.detach().cpu()


def verify_compressed_path(
    model,
    trace: FastDLLMTrace,
    result,
    device: torch.device,
    model_type: str = "LLaDA",
    mask_id: int = MASK_ID,
) -> None:
    """Replay the compressed path from s_0, carrying only simulated states."""
    path = result.compressed_path
    if not path or path[0] != 0 or path[-1] != trace.num_forward_steps:
        raise AssertionError(f"compressed path does not span the trace: {path}")
    current = trace.states[0]
    forwards = 0
    for t, u in zip(path, path[1:]):
        source_step = trace.steps[t]
        source = source_step.state
        if current.state_hash() != source.state_hash():
            raise AssertionError(f"replayed state diverged from baseline step {t}")
        sequence = torch.tensor(current.token_ids, device=device).unsqueeze(0)
        token_attention = torch.tensor(
            source.attention_mask, device=device
        ).unsqueeze(0)
        attention = prepare_generation_attention_mask(
            model, token_attention, source.sequence_length, model_type
        )
        prediction = evaluate_denoiser_sequence(
            model, sequence, current.prompt_length, model_type, attention,
            temperature=0.0, forbidden_token_id=mask_id,
        )
        forwards += 1
        actual_top1 = prediction.greedy_tokens[0].detach().cpu().tolist()
        recorded_top1 = source_step.top1_token_ids[source.prompt_length:]
        if tuple(actual_top1) != recorded_top1:
            raise AssertionError(f"top-1 determinism failed at step {t}")
        target = trace.states[u]
        positions = {
            i for i, (masked, target_masked) in enumerate(
                zip(current.mask_bitmap, target.mask_bitmap)
            ) if masked and not target_masked
        }
        replayed_top1 = (
            current.token_ids[:current.prompt_length] + tuple(actual_top1)
        )
        current = current.commit(positions, replayed_top1)
        if current.state_hash() != target.state_hash():
            raise AssertionError(f"compressed jump {t}->{u} failed state-hash equality")
    if current != trace.terminal_state:
        raise AssertionError("compressed path did not preserve the final state")
    if forwards != result.trace_nfe:
        raise AssertionError(
            f"replayed {forwards} forwards but trace_nfe is {result.trace_nfe}"
        )


def _state_json(state: FastDLLMState) -> dict:
    return {
        "token_ids": list(state.token_ids),
        "mask_bitmap": list(state.mask_bitmap),
        "active_block": state.active_block,
        "attention_mask": list(state.attention_mask),
        "controller_state": dict(state.controller_state),
        "state_hash": state.state_hash(),
    }


def _trace_json(trace: FastDLLMTrace, result) -> dict:
    steps = []
    for index, step in enumerate(trace.steps):
        steps.append({
            "step_index": index,
            "state_before_forward": _state_json(step.state),
            "top1_token_id_per_position": list(step.top1_token_ids),
            "top1_probability_per_position": list(step.top1_probabilities),
            "eligible_positions": sorted(step.state.eligible_positions),
            "committed_positions": list(step.committed_positions),
            "state_after_commit": _state_json(trace.states[index + 1]),
        })
    jump_records = []
    for t, u in zip(result.compressed_path, result.compressed_path[1:]):
        if u == t + 1:
            continue
        source = trace.steps[t]
        positions = sorted(
            i for i, (masked, target_masked) in enumerate(
                zip(source.state.mask_bitmap, trace.states[u].mask_bitmap)
            ) if masked and not target_masked
        )
        jump_records.append({
            "source_step": t,
            "target_step": u,
            "jump_length": u - t,
            "positions": positions,
            "top1_probabilities": [source.top1_probabilities[i] for i in positions],
        })
    return {
        **result.to_dict(),
        "nontrivial_jumps": jump_records,
        "steps": steps,
        "terminal_state": _state_json(trace.terminal_state),
    }


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("trace collection requires a GPU")
    _seed(args.seed)
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        revision=args.model_revision,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, revision=args.model_revision
    )
    model_type, mask_id = resolve_model_type_and_mask_id(
        model, args.model_path, args.model_type, args.mask_id
    )
    dataset = TrainingMathDataset(
        tokenizer,
        args.dataset,
        max_prompt_length=args.max_prompt_length,
        protocol=PROTOCOL,
        invalid_gold=args.math_training_invalid_gold,
    )
    dataset_indices = shard_indices(
        len(dataset), args.num_shards, args.shard_index, args.max_shard_samples
    )
    if args.num_samples >= 0:
        dataset_indices = dataset_indices[:args.num_samples]
    metadata = {
        "schema_version": 2,
        "definition": "G_trace = baseline_nfe - minimum exact-rejoin NFE",
        "edge_representation": "interval_max_reach",
        "model_path": args.model_path,
        "model_type": model_type,
        "mask_id": mask_id,
        "dataset": args.dataset,
        "split": "train",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "dataset_indices": dataset_indices,
        "threshold": args.threshold,
        "temperature": 0.0,
        "generation_length": args.generation_length,
        "block_length": args.block_length,
        "few_shot": 0,
        "actual_few_shot": 0,
        "max_prompt_length": args.max_prompt_length,
        "dataset_id": (
            "openai/gsm8k" if args.dataset == "gsm8k"
            else getattr(dataset, "dataset_id", None)
        ),
        "training_math_protocol": PROTOCOL,
        "math_training_invalid_gold": args.math_training_invalid_gold,
        "math_protocol_metadata": protocol_metadata(args.max_prompt_length),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    writer = ResumableArtifactWriter(Path(args.output), metadata, dataset_indices)
    results = []
    for local_index, dataset_index in enumerate(dataset_indices):
        completed = writer.completed_result(local_index, dataset_index)
        if completed is not None:
            results.append(completed)
            print(f"resumed sample={completed.sample_id}", flush=True)
            continue
        prompt, question, answer = dataset[dataset_index]
        task_id = None
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if args.max_prompt_length is not None:
            encoded.input_ids = encoded.input_ids[:, -args.max_prompt_length:]
            encoded.attention_mask = encoded.attention_mask[:, -args.max_prompt_length:]
        trace, sequences, _ = record_trace(
            model, encoded.input_ids.to(device), encoded.attention_mask.to(device),
            f"train:{dataset_index}", args.threshold,
            args.generation_length, args.block_length, model_type, mask_id,
        )
        result = compute_g_trace(trace)
        results.append(result)
        if local_index < args.verify_samples:
            verify_compressed_path(model, trace, result, device, model_type, mask_id)
        sample = _trace_json(trace, result)
        sample.update({
            "question": question,
            "reference_answer": answer,
            "task_id": task_id,
            "final_text": tokenizer.decode(
                sequences[0, -args.generation_length:], skip_special_tokens=True
            ),
        })
        writer.append(local_index, dataset_index, sample, result)
        print(
            f"sample=train:{dataset_index} baseline_nfe={result.baseline_nfe} "
            f"trace_nfe={result.trace_nfe} g_trace={result.g_trace}",
            flush=True,
        )
    writer.finalize(results)
    print(f"wrote {len(results)} samples to {writer.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-type", choices=("LLaDA", "Dream"))
    parser.add_argument("--mask-id", type=int)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-shard-samples", type=int, default=-1)
    parser.add_argument("--verify-samples", type=int, default=0,
                        help="Replay the shortest exact-rejoin path of the first N samples.")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--generation-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-prompt-length", type=optional_positive_int, default=256)
    parser.add_argument(
        "--math-training-invalid-gold",
        choices=("error", "exclude"),
        default="exclude",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
