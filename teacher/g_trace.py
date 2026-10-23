"""Exact hindsight trajectory compression for deterministic Fast-dLLM.

A recorded Fast-dLLM trajectory s_0, ..., s_T may jump from state s_t to a
later state s_u when committing s_t's recorded top-1 predictions at the
positions unmasked between t and u (all in s_t's active block) reproduces s_u
exactly.  No counterfactual model state is ever evaluated: every jump rejoins
a state the baseline trajectory actually visited, so the final output and the
remaining trajectory are unchanged.  G_trace = T minus the shortest number of
such jumps from s_0 to s_T.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class FastDLLMState:
    token_ids: tuple[int, ...]
    mask_bitmap: tuple[bool, ...]
    active_block: int
    block_length: int
    prompt_length: int
    attention_mask: tuple[int, ...]
    controller_state: tuple[tuple[str, int | float | str | bool], ...] = ()

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        if len(self.mask_bitmap) != length:
            raise ValueError("token_ids and mask_bitmap lengths differ")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if not 0 <= self.prompt_length <= length:
            raise ValueError("prompt_length is outside the sequence")

    @property
    def sequence_length(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        response_length = self.sequence_length - self.prompt_length
        return (response_length + self.block_length - 1) // self.block_length

    @property
    def eligible_positions(self) -> frozenset[int]:
        start = self.prompt_length + self.active_block * self.block_length
        end = min(start + self.block_length, self.sequence_length)
        return frozenset(i for i in range(start, end) if self.mask_bitmap[i])

    def canonical_payload(self) -> dict:
        return {
            "token_ids": self.token_ids,
            "mask_bitmap": self.mask_bitmap,
            "active_block": self.active_block,
            "block_length": self.block_length,
            "prompt_length": self.prompt_length,
            "attention_mask": self.attention_mask,
            "controller_state": self.controller_state,
        }

    def state_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(), separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def commit(self, positions: Iterable[int], top1_token_ids: tuple[int, ...]) -> FastDLLMState:
        positions = frozenset(positions)
        if not positions.issubset(self.eligible_positions):
            raise ValueError("commit contains a position outside the eligible set")
        tokens = list(self.token_ids)
        masks = list(self.mask_bitmap)
        for position in positions:
            tokens[position] = top1_token_ids[position]
            masks[position] = False
        active_block = self.active_block
        while active_block < self.num_blocks:
            start = self.prompt_length + active_block * self.block_length
            end = min(start + self.block_length, self.sequence_length)
            if any(masks[start:end]):
                break
            active_block += 1
        return FastDLLMState(
            token_ids=tuple(tokens),
            mask_bitmap=tuple(masks),
            active_block=active_block,
            block_length=self.block_length,
            prompt_length=self.prompt_length,
            attention_mask=self.attention_mask,
            controller_state=self.controller_state,
        )


@dataclass(frozen=True)
class FastDLLMStep:
    state: FastDLLMState
    top1_token_ids: tuple[int, ...]
    top1_probabilities: tuple[float, ...]
    committed_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        length = self.state.sequence_length
        if len(self.top1_token_ids) != length or len(self.top1_probabilities) != length:
            raise ValueError("prediction lengths must match the decoder state")


@dataclass(frozen=True)
class FastDLLMTrace:
    sample_id: str
    steps: tuple[FastDLLMStep, ...]
    terminal_state: FastDLLMState

    @property
    def states(self) -> tuple[FastDLLMState, ...]:
        return tuple(step.state for step in self.steps) + (self.terminal_state,)

    @property
    def num_forward_steps(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class GTraceResult:
    sample_id: str
    baseline_nfe: int
    trace_nfe: int
    g_trace: int
    relative_g_trace: float
    compressed_path: tuple[int, ...]
    max_reach: tuple[int, ...]
    number_of_nontrivial_jumps: int
    maximum_jump_length: int
    mean_jump_length: float

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "baseline_nfe": self.baseline_nfe,
            "trace_nfe": self.trace_nfe,
            "g_trace": self.g_trace,
            "relative_g_trace": self.relative_g_trace,
            "compressed_path": list(self.compressed_path),
            "max_reach": list(self.max_reach),
            "number_of_nontrivial_jumps": self.number_of_nontrivial_jumps,
            "maximum_jump_length": self.maximum_jump_length,
            "mean_jump_length": self.mean_jump_length,
        }


def newly_committed_positions(source: FastDLLMState, target: FastDLLMState) -> frozenset[int]:
    if source.sequence_length != target.sequence_length:
        raise ValueError("source and target sequence lengths differ")
    return frozenset(
        i for i, (source_masked, target_masked) in enumerate(
            zip(source.mask_bitmap, target.mask_bitmap)
        )
        if source_masked and not target_masked
    )


def admissible_jump(trace: FastDLLMTrace, t: int, u: int) -> bool:
    if not 0 <= t < u <= trace.num_forward_steps:
        return False
    source = trace.steps[t].state
    target = trace.states[u]
    committed = newly_committed_positions(source, target)
    if not committed or not committed.issubset(source.eligible_positions):
        return False
    top1 = trace.steps[t].top1_token_ids
    if any(top1[i] != target.token_ids[i] for i in committed):
        return False
    simulated = source.commit(committed, top1)
    return simulated == target


def _state_metadata(state: FastDLLMState) -> tuple:
    """State fields that committing tokens cannot alter."""
    return (
        state.block_length,
        state.prompt_length,
        state.attention_mask,
        state.controller_state,
    )


def compute_max_reach(trace: FastDLLMTrace) -> tuple[int, ...]:
    """Compute the farthest exact-rejoin state reachable from every step.

    Reachable targets from step ``t`` form the interval
    ``[t + 1, max_reach[t]]``.  The endpoint is the earliest of:

    * the first baseline commit that contradicts the saved top-1 prediction;
    * the end of the source state's active block; and
    * the first change to decoder metadata that token commitment cannot alter.

    Building commit times and scanning the source block costs ``O(TL)`` in the
    worst case, rather than enumerating ``O(T^2)`` candidate jumps and comparing
    ``O(L)`` states for each one.
    """
    total_steps = trace.num_forward_steps
    if total_steps == 0:
        return ()

    states = trace.states
    sequence_length = states[0].sequence_length
    if any(state.sequence_length != sequence_length for state in states):
        raise ValueError("decoder state sequence lengths differ within the trace")

    commit_step = [-1] * sequence_length
    for step_index, step in enumerate(trace.steps):
        if not step.committed_positions:
            raise ValueError(f"step {step_index} commits no positions")
        for position in step.committed_positions:
            if not 0 <= position < sequence_length:
                raise ValueError(f"step {step_index} commits invalid position {position}")
            if commit_step[position] >= 0:
                raise ValueError(f"position {position} is committed more than once")
            if not step.state.mask_bitmap[position]:
                raise ValueError(f"step {step_index} recommits position {position}")
            if states[step_index + 1].mask_bitmap[position]:
                raise ValueError(
                    f"position {position} remains masked after step {step_index}"
                )
            commit_step[position] = step_index

    # metadata_run_end[t] is the final state index in the consecutive run whose
    # non-token metadata equals that of state t.
    metadata = [_state_metadata(state) for state in states]
    metadata_run_end = [total_steps] * (total_steps + 1)
    metadata_run_end[total_steps] = total_steps
    for state_index in range(total_steps - 1, -1, -1):
        metadata_run_end[state_index] = (
            metadata_run_end[state_index + 1]
            if metadata[state_index] == metadata[state_index + 1]
            else state_index
        )

    final_tokens = trace.terminal_state.token_ids
    invalid_adjacent = [
        step_index
        for step_index in range(total_steps)
        if not admissible_jump(trace, step_index, step_index + 1)
    ]
    if invalid_adjacent:
        raise ValueError(
            f"baseline adjacent edges are not admissible at steps {invalid_adjacent}"
        )

    max_reach: list[int] = []
    for step_index, step in enumerate(trace.steps):
        source = step.state
        eligible = source.eligible_positions
        if not eligible:
            raise ValueError(f"step {step_index} has no eligible positions")

        future_commits = [commit_step[position] for position in eligible]
        if any(commit < step_index for commit in future_commits):
            raise ValueError(
                f"step {step_index} has eligible positions with invalid commit times"
            )
        if any(commit < 0 for commit in future_commits):
            raise ValueError(
                f"step {step_index} has eligible positions that are never committed"
            )

        # Filling every position that is currently eligible can advance exactly
        # to the next block, but never commit positions in that next block.
        block_end_state = max(future_commits) + 1
        conflicting_steps = [
            commit_step[position]
            for position in eligible
            if step.top1_token_ids[position] != final_tokens[position]
        ]
        first_conflict = min(conflicting_steps, default=block_end_state)
        reach = min(block_end_state, first_conflict, metadata_run_end[step_index])

        if reach <= step_index:
            raise ValueError(
                f"baseline adjacent edge ({step_index}, {step_index + 1}) is not admissible"
            )
        # One exact comparison per source preserves the full-state rejoin
        # guarantee while retaining O(TL) total complexity.
        if not admissible_jump(trace, step_index, reach):
            raise ValueError(
                f"analytical max reach ({step_index}, {reach}) fails exact rejoin"
            )
        max_reach.append(reach)

    return tuple(max_reach)


def shortest_g_trace_trajectory(
    max_reach: Iterable[int],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Find a minimum-NFE trajectory through interval jump edges in O(T)."""
    reaches = tuple(int(value) for value in max_reach)
    total_steps = len(reaches)
    if total_steps == 0:
        return 0, 0, (0,), (-1,)

    parent = [-1] * (total_steps + 1)
    scan_start = 0
    current_end = 0
    trace_nfe = 0

    while current_end < total_steps:
        next_end = current_end
        for source in range(scan_start, current_end + 1):
            reach = min(reaches[source], total_steps)
            if reach <= source:
                continue
            if reach > next_end:
                for target in range(next_end + 1, reach + 1):
                    parent[target] = source
                next_end = reach
        if next_end == current_end:
            raise RuntimeError(
                f"final state {total_steps} is unreachable from the current trace"
            )
        scan_start = current_end + 1
        current_end = next_end
        trace_nfe += 1

    path = []
    node = total_steps
    while True:
        path.append(node)
        if node == 0:
            break
        node = parent[node]
        if node < 0:
            raise RuntimeError("failed to reconstruct the G-trace trajectory")
    path.reverse()
    return trace_nfe, total_steps - trace_nfe, tuple(path), tuple(parent)


def compute_g_trace(trace: FastDLLMTrace) -> GTraceResult:
    baseline_nfe = trace.num_forward_steps
    if baseline_nfe == 0:
        return GTraceResult(
            sample_id=trace.sample_id,
            baseline_nfe=0,
            trace_nfe=0,
            g_trace=0,
            relative_g_trace=0.0,
            compressed_path=(0,),
            max_reach=(),
            number_of_nontrivial_jumps=0,
            maximum_jump_length=0,
            mean_jump_length=0.0,
        )
    max_reach = compute_max_reach(trace)
    trace_nfe, g_trace, path, _ = shortest_g_trace_trajectory(max_reach)
    jump_lengths = [u - t for t, u in pairwise(path) if u - t > 1]
    return GTraceResult(
        sample_id=trace.sample_id,
        baseline_nfe=baseline_nfe,
        trace_nfe=trace_nfe,
        g_trace=g_trace,
        relative_g_trace=g_trace / baseline_nfe,
        compressed_path=tuple(path),
        max_reach=max_reach,
        number_of_nontrivial_jumps=len(jump_lengths),
        maximum_jump_length=max(jump_lengths, default=1),
        mean_jump_length=(sum(jump_lengths) / len(jump_lengths) if jump_lengths else 1.0),
    )


def summarize_g_trace(
    results: Iterable[GTraceResult], bootstrap_samples: int = 2000, seed: int = 0
) -> dict:
    results = tuple(results)
    if not results:
        raise ValueError("cannot summarize an empty G_trace result set")
    baseline = np.asarray([row.baseline_nfe for row in results], dtype=float)
    trace_nfe = np.asarray([row.trace_nfe for row in results], dtype=float)
    gaps = baseline - trace_nfe
    relative = np.divide(gaps, baseline, out=np.zeros_like(gaps), where=baseline > 0)
    rng = np.random.default_rng(seed)
    if bootstrap_samples > 0:
        indices = rng.integers(0, len(results), size=(bootstrap_samples, len(results)))
        bootstrap_gap = gaps[indices].mean(axis=1)
        ci_low, ci_high = np.percentile(bootstrap_gap, [2.5, 97.5])
    else:
        ci_low = ci_high = gaps.mean()
    return {
        "num_samples": len(results),
        "mean_baseline_nfe": float(baseline.mean()),
        "mean_trace_nfe": float(trace_nfe.mean()),
        "mean_g_trace": float(gaps.mean()),
        "median_g_trace": float(np.median(gaps)),
        "mean_relative_g_trace": float(relative.mean()),
        "probability_g_trace_positive": float(np.mean(gaps > 0)),
        "g_trace_percentiles": {
            str(percentile): float(np.percentile(gaps, percentile))
            for percentile in (0, 25, 50, 75, 90, 95, 100)
        },
        "mean_g_trace_paired_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "bootstrap_samples": bootstrap_samples,
    }
