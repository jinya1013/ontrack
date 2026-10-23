"""Learn2PD's Easy First Generation Policy (EGP) oracle teacher."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from common.generation.generation import (
    DenoiserState,
    GenerationResult,
    generate_unified,
    prepare_generation_token_mask,
)


@dataclass(frozen=True)
class EGPDecision:
    """Oracle action plus the reference-token fallback used to make progress."""

    commit_mask: torch.Tensor
    fallback_mask: torch.Tensor


@dataclass(frozen=True)
class EGPTeacherRollout:
    """A standard-decode reference paired with its on-policy EGP states."""

    reference_tokens: torch.Tensor
    generation: GenerationResult
    transitions: tuple[dict, ...]


def offline_teacher_row(
    rollout: EGPTeacherRollout,
    *,
    dataset: str,
    sample_id: str,
    prompt_length: int,
    block_length: int,
    split: str = "train",
    source_shard: int = 0,
    training_math_protocol: str = "legacy",
    math_training_invalid_gold: str = "error",
    math_protocol_metadata: dict | None = None,
) -> dict:
    """Package an EGP rollout as a row accepted by offline teacher forcing."""
    generation_length = int(rollout.reference_tokens.shape[-1])
    if not rollout.transitions:
        raise ValueError("cannot package an empty EGP rollout")
    if prompt_length + generation_length != len(rollout.transitions[0]["input_ids"]):
        raise ValueError("prompt and generation lengths do not match transition state")
    # The Parquet schema intentionally stores only fields consumed by training;
    # fallback positions remain available on the in-memory rollout for audits.
    transition_fields = {
        "source_step",
        "target_step",
        "active_block",
        "input_ids",
        "attention_mask",
        "mask_bitmap",
        "eligible_positions",
        "target_commit_positions",
        "target_action_sequence",
        "target_token_ids",
        "target_confidences",
    }
    return {
        "schema_version": 2,
        "dataset": dataset,
        "split": split,
        "source_shard": source_shard,
        "sample_id": str(sample_id),
        "prompt_length": prompt_length,
        "generation_length": generation_length,
        "block_length": block_length,
        "training_math_protocol": training_math_protocol,
        "math_training_invalid_gold": math_training_invalid_gold,
        "math_protocol_metadata": json.dumps(
            math_protocol_metadata or {"name": training_math_protocol}, sort_keys=True
        ),
        "transitions": [
            {key: transition[key] for key in transition_fields}
            for transition in rollout.transitions
        ],
        "max_reach": [],
        "artifact_steps": [],
        "terminal_mask_bitmap": [],
    }


def egp_decision(
    greedy_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    eligible_mask: torch.Tensor,
) -> EGPDecision:
    """Commit eligible top-1 predictions iff they equal the reference.

    If an active row has no correct proposal, EGP copies its remaining active
    block from the reference.  Those forced tokens are rollout progress only:
    the supervised commit label remains the all-zero oracle decision.
    """
    if greedy_tokens.shape != reference_tokens.shape:
        raise ValueError("greedy_tokens and reference_tokens must have equal shape")
    if eligible_mask.shape != greedy_tokens.shape:
        raise ValueError("eligible_mask must match token shapes")
    if eligible_mask.dtype != torch.bool:
        raise TypeError("eligible_mask must be boolean")
    commit = eligible_mask & greedy_tokens.eq(reference_tokens)
    stalled = eligible_mask.any(dim=-1) & ~commit.any(dim=-1)
    fallback = eligible_mask & stalled.unsqueeze(-1)
    return EGPDecision(commit_mask=commit, fallback_mask=fallback)


def transition_from_state(
    state: DenoiserState,
    reference_tokens: torch.Tensor,
    *,
    batch_index: int = 0,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """Convert one observed EGP state to the existing offline-TF transition schema."""
    response_length = state.greedy_tokens.shape[1]
    if reference_tokens.ndim == 1:
        reference_tokens = reference_tokens.unsqueeze(0)
    if reference_tokens.shape != state.greedy_tokens.shape:
        raise ValueError("reference_tokens must cover the generated response")
    decision = egp_decision(
        state.greedy_tokens, reference_tokens, state.block_masked_positions
    )
    eligible = state.block_masked_positions[batch_index].nonzero(as_tuple=True)[0]
    committed = decision.commit_mask[batch_index].nonzero(as_tuple=True)[0]
    probabilities = state.generation_probabilities[batch_index]
    confidences = probabilities.max(dim=-1).values
    full_attention = (
        torch.ones_like(state.sequence[batch_index], dtype=torch.bool)
        if attention_mask is None
        else attention_mask[batch_index].bool()
    )
    if full_attention.shape != state.sequence[batch_index].shape:
        raise ValueError("attention_mask must cover the complete sequence")
    return {
        "source_step": int(state.macro_step),
        "target_step": int(state.macro_step + 1),
        "active_block": int(state.block_index),
        "input_ids": state.sequence[batch_index].detach().cpu().tolist(),
        "attention_mask": full_attention.detach().cpu().to(torch.int8).tolist(),
        "mask_bitmap": torch.cat((
            torch.zeros(state.prompt_length, dtype=torch.bool, device=state.sequence.device),
            state.masked_positions[batch_index],
        )).cpu().tolist(),
        "eligible_positions": eligible.detach().cpu().tolist(),
        "target_commit_positions": committed.detach().cpu().tolist(),
        "target_action_sequence": committed.detach().cpu().tolist() + [response_length],
        "target_token_ids": reference_tokens[batch_index, committed].detach().cpu().tolist(),
        "target_confidences": confidences[committed].detach().cpu().tolist(),
        "egp_fallback_positions": decision.fallback_mask[batch_index]
        .nonzero(as_tuple=True)[0]
        .detach().cpu().tolist(),
    }


@torch.no_grad()
def collect_egp_teacher_rollout(
    model,
    prompt: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    reference_tokens: torch.Tensor | None = None,
    reference_steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    model_type: str | None = None,
) -> EGPTeacherRollout:
    """Collect faithful EGP states against a standard-decoding reference.

    Supplying ``reference_tokens`` skips the reference pass.  Otherwise the
    reference is the final output of deterministic low-confidence decoding,
    matching Learn2PD rather than using dataset gold answers.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("EGP teacher collection currently requires batch size one")
    if reference_tokens is None:
        reference = generate_unified(
            model=model,
            prompt=prompt,
            attention_mask=attention_mask,
            remasking="low_confidence",
            steps=reference_steps or gen_length,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            mask_id=mask_id,
            model_type=model_type,
        )
        reference_tokens = reference.sequences[:, -gen_length:].detach()
    elif reference_tokens.shape != (1, gen_length):
        raise ValueError("reference_tokens must have shape [1, gen_length]")
    if reference_tokens.dtype != prompt.dtype:
        raise TypeError("reference_tokens and prompt must use the same token dtype")
    reference_tokens = reference_tokens.to(prompt.device)

    full_attention_mask = prepare_generation_token_mask(
        attention_mask, prompt.shape[1] + gen_length
    )
    transitions: list[dict] = []

    def observe(state: DenoiserState) -> None:
        transitions.append(
            transition_from_state(
                state,
                reference_tokens,
                attention_mask=full_attention_mask,
            )
        )

    generation = generate_unified(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        remasking="learn2pd_teacher",
        learn2pd_reference_tokens=reference_tokens,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        mask_id=mask_id,
        model_type=model_type,
        state_observer=observe,
    )
    if not torch.equal(generation.sequences[:, -gen_length:], reference_tokens):
        raise RuntimeError("EGP rollout failed to reproduce its reference sequence")
    return EGPTeacherRollout(
        reference_tokens=reference_tokens.detach().cpu(),
        generation=generation,
        transitions=tuple(transitions),
    )
