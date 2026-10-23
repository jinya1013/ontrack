"""Blockwise masked-diffusion decoding with interchangeable unmasking rules.

Every rule commits a nonempty subset of the masked positions of the active
block per denoiser evaluation (NFE):

* ``policy``: a learned unmasking policy (``bernoulli-deterministic``:
  commit positions whose policy probability exceeds a threshold);
* ``fastdllm``: Fast-dLLM confidence thresholding;
* ``eb_sampler``: EB-Sampler entropy-bounded commits;
* ``low_confidence``: a fixed number of most-confident tokens per step
  (one token per step is the one-by-one reference);
* ``learn2pd_teacher``: Learn2PD's oracle against reference tokens (teacher
  collection only).

``kv_cache="prefix_cache"`` evaluates the denoiser with Fast-dLLM's prefix
KV cache (LLaDA only).
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import sample_token_ids

KV_CACHE_MODES = ("no_cache", "prefix_cache")
HIDDEN_INPUT_TYPES = {"hidden", "hidden_only"}


@dataclass(frozen=True)
class DenoiserState:
    """Read-only view of one denoising macro state for teacher collection.

    Observers must not retain the large GPU logits, probabilities, or hidden
    states. Compute what is needed immediately or copy small values.
    """

    sequence: torch.Tensor
    generation_logits: torch.Tensor
    generation_probabilities: torch.Tensor
    greedy_tokens: torch.Tensor
    final_hidden_states: torch.Tensor | None
    masked_positions: torch.Tensor
    block_masked_positions: torch.Tensor
    macro_step: int
    block_index: int
    prompt_length: int


@dataclass(frozen=True)
class DenoiserPrediction:
    """One denoiser evaluation with repository-standard output alignment."""

    model_output: Any
    generation_logits: torch.Tensor
    generation_probabilities: torch.Tensor
    greedy_tokens: torch.Tensor


class GenerationResult(NamedTuple):
    sequences: torch.Tensor  # (B, prompt_L + gen_L)
    steps_taken: torch.Tensor  # (B,) denoiser evaluations (NFE)
    selected_set_sizes: torch.Tensor | None = None  # (B, steps) tokens committed per step


def policy_core(policy):
    """Unwrap DDP/Hugging Face wrappers to the policy network."""
    while hasattr(policy, "module"):
        policy = policy.module
    return getattr(policy, "base_policy", policy)


def policy_uses_hidden_states(policy) -> bool:
    return policy is not None and getattr(policy_core(policy), "input_type", "confidence") in HIDDEN_INPUT_TYPES


def policy_timestep(steps_taken: torch.Tensor, generation_length: int) -> torch.Tensor:
    """Normalized controller step used to condition the policy."""
    return steps_taken.unsqueeze(-1) * (1 / generation_length)


def prepare_generation_token_mask(
    prompt_attention_mask: torch.Tensor | None,
    total_length: int,
) -> torch.Tensor | None:
    """Return the canonical one-dimensional validity mask for a full sequence."""
    if prompt_attention_mask is None:
        return None
    batch_size, prompt_length = prompt_attention_mask.shape
    if total_length < prompt_length:
        raise ValueError("total_length cannot be shorter than the prompt mask")
    attention_mask = torch.ones(
        (batch_size, total_length), dtype=torch.float,
        device=prompt_attention_mask.device,
    )
    attention_mask[:, :prompt_length] = prompt_attention_mask.float()
    return attention_mask


def prepare_generation_attention_mask(
    model,
    prompt_attention_mask: torch.Tensor | None,
    total_length: int,
    model_type: str | None,
) -> torch.Tensor | None:
    """Build a model-facing mask without changing the canonical saved mask.

    Dream uses bidirectional attention.  Its remote implementation accepts a
    boolean ``[B, 1, query, key]`` padding mask, but uses unrestricted attention
    when no padding is present.  LLaDA consumes the ordinary ``[B, length]``
    validity mask.
    """
    attention_mask = prepare_generation_token_mask(prompt_attention_mask, total_length)
    if attention_mask is None:
        return None
    if model_type == "Dream":
        if not torch.any(attention_mask == 0):
            return None
        valid = attention_mask.bool()
        return torch.logical_and(
            valid.unsqueeze(1).unsqueeze(-2),
            valid.unsqueeze(1).unsqueeze(-1),
        )
    model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
    return attention_mask.to(model_dtype)


def evaluate_denoiser_sequence(
    model,
    sequence: torch.Tensor,
    prompt_length: int,
    model_type: str | None,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 0.0,
    output_hidden_states: bool = False,
    forbidden_token_id: int | None = None,
) -> DenoiserPrediction:
    """Evaluate a sequence with the LLaDA/Dream response-position alignment.

    ``attention_mask`` is the already-expanded full-sequence mask returned by
    :func:`prepare_generation_attention_mask`.
    """
    model_output = model(
        sequence,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
    )
    if model_type == "Dream":
        logits = model_output.logits[:, prompt_length - 1 : -1]
    else:
        logits = model_output.logits[:, prompt_length:]
    return DenoiserPrediction(
        model_output=model_output,
        generation_logits=logits,
        generation_probabilities=F.softmax(logits, dim=-1),
        greedy_tokens=sample_token_ids(logits, temperature, forbidden_token_id),
    )


class CachedDenoiser:
    """Fast-dLLM v1 prefix-cache evaluation for one active block.

    The first evaluation of a block is a full forward that stores K/V for the
    prefix before the block; later evaluations of the same block recompute the
    block and everything after it from that cache (``generate_with_prefix_cache``).
    Positions before the block keep their block-start logits and final hidden
    states, so full-context consumers (Fast-dLLM thresholds and policies) see
    block-start values there.  ``model_output.hidden_states[-1]`` is a
    full-sequence buffer with the same layout as an uncached forward.  Requires
    :mod:`common.models.modeling_llada_fastdllm` and unpadded inputs.
    """

    def __init__(
        self, model, prompt_length: int, temperature: float,
        forbidden_token_id: int | None, output_hidden_states: bool = False,
    ):
        self.model = model
        self.prompt_length = prompt_length
        self.temperature = temperature
        self.forbidden_token_id = forbidden_token_id
        self.output_hidden_states = output_hidden_states
        self.block_start: int | None = None

    def evaluate(self, sequence: torch.Tensor, block_slice: slice) -> DenoiserPrediction:
        start = self.prompt_length + block_slice.start
        if self.block_start != block_slice.start:
            output = self.model(
                sequence, use_cache=True, output_hidden_states=self.output_hidden_states
            )
            self.past_key_values = [
                tuple(tensor[:, :, :start] for tensor in layer)
                for layer in output.past_key_values
            ]
            self.logits = output.logits[:, self.prompt_length:].clone()
            self.probabilities = F.softmax(self.logits, dim=-1)
            self.greedy = sample_token_ids(self.logits, self.temperature, self.forbidden_token_id)
            if self.output_hidden_states:
                self.hidden = output.hidden_states[-1].clone()
            self.block_start = block_slice.start
        else:
            output = self.model(
                sequence[:, start:], past_key_values=self.past_key_values,
                use_cache=True, output_hidden_states=self.output_hidden_states,
            )
            updated = slice(block_slice.start, None)
            self.logits[:, updated] = output.logits
            self.probabilities[:, updated] = F.softmax(output.logits, dim=-1)
            self.greedy[:, updated] = sample_token_ids(
                output.logits, self.temperature, self.forbidden_token_id
            )
            if self.output_hidden_states:
                self.hidden[:, start:] = output.hidden_states[-1]
        model_output = SimpleNamespace(
            logits=output.logits,
            hidden_states=(self.hidden,) if self.output_hidden_states else None,
        )
        return DenoiserPrediction(
            model_output=model_output,
            generation_logits=self.logits,
            generation_probabilities=self.probabilities,
            greedy_tokens=self.greedy,
        )


def prefix_cache_block_predictions(
    model, states: torch.Tensor, prompt_length: int, block_slice: slice,
    mask_id: int, temperature: float, output_hidden_states: bool = False,
) -> DenoiserPrediction:
    """Batched :class:`CachedDenoiser` features for states of one block.

    ``states`` are unpadded active-block decoding states of one trajectory
    inside the same active block.  Their block-entry state, with the block and
    everything after it masked, is reconstructed exactly; it is evaluated by a
    full forward whose prefix K/V and pre-block features are shared by every
    state, as in evaluation.  States equal to the entry state keep the
    full-forward values, matching the first evaluation of a block; the others
    recompute the block and suffix from the cached prefix.
    """
    start = prompt_length + block_slice.start
    end = prompt_length + block_slice.stop
    if not bool((states[:, end:] == mask_id).all()):
        raise ValueError("prefix-cache features require a fully masked suffix")
    entry = states[:1].clone()
    entry[:, start:] = mask_id
    if not torch.equal(states[:, :start], entry[:, :start].expand(len(states), -1)):
        raise ValueError("prefix-cache states must share one decoded prefix")
    full = model(entry, use_cache=True, output_hidden_states=output_hidden_states)
    logits = full.logits[:, prompt_length:].expand(len(states), -1, -1).clone()
    hidden = (
        full.hidden_states[-1].expand(len(states), -1, -1).clone()
        if output_hidden_states else None
    )
    recompute = (states[:, start:] != mask_id).any(dim=-1)
    if bool(recompute.any()):
        rows = states[recompute]
        past_key_values = [
            tuple(tensor[:, :, :start].expand(len(rows), -1, -1, -1) for tensor in layer)
            for layer in full.past_key_values
        ]
        cached = model(
            rows[:, start:], past_key_values=past_key_values,
            use_cache=True, output_hidden_states=output_hidden_states,
        )
        logits[recompute, block_slice.start:] = cached.logits
        if hidden is not None:
            hidden[recompute, start:] = cached.hidden_states[-1]
    return DenoiserPrediction(
        model_output=SimpleNamespace(
            logits=logits, hidden_states=(hidden,) if hidden is not None else None
        ),
        generation_logits=logits,
        generation_probabilities=F.softmax(logits, dim=-1),
        greedy_tokens=sample_token_ids(logits, temperature, mask_id),
    )


def generation_aligned_hidden_states(
    hidden_states: torch.Tensor,
    prompt_length: int,
    model_type: str | None,
) -> torch.Tensor:
    """Align final-layer hidden states with generation logits.

    Dream predicts each response token from the preceding sequence position,
    whereas LLaDA predicts from the response position itself.
    """
    if model_type == "Dream":
        return hidden_states[:, prompt_length - 1 : -1]
    return hidden_states[:, prompt_length:]


def policy_inputs_from_prediction(
    policy, mask_index, probs, hidden_states, timestep, confidences_top_p, prompt_length, model_type,
):
    """Assemble the policy input tuple ``(mask, features, timestep[, confidence])``."""
    confidence = probs.topk(confidences_top_p, dim=-1).values
    core = policy_core(policy)
    if getattr(core, "input_type", "confidence") in HIDDEN_INPUT_TYPES:
        hidden = generation_aligned_hidden_states(hidden_states, prompt_length, model_type)
        inputs = (mask_index, hidden, timestep)
        if getattr(core, "policy_hidden_residual", False):
            inputs += (confidence,)
        return inputs
    return (mask_index, confidence, timestep)


@torch.no_grad()
def generate_unified(
    model,
    prompt: torch.Tensor,
    remasking: str,
    policy=None,
    thres: float | None = None,
    steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    sampling_mode: str = "bernoulli-deterministic",
    model_type: str | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature_policy: float = 1.0,
    confidences_top_p: int = 1,
    state_observer: Callable[[DenoiserState], None] | None = None,
    policy_accept_threshold: float | None = None,
    learn2pd_reference_tokens: torch.Tensor | None = None,
    eb_gamma: float | None = None,
    eb_error_proxy: str = "confidence",
    kv_cache: str = "no_cache",
) -> GenerationResult:
    if kv_cache not in KV_CACHE_MODES:
        raise ValueError(f"kv_cache must be one of {KV_CACHE_MODES}")
    if kv_cache != "no_cache":
        import inspect
        if remasking not in {"fastdllm", "eb_sampler", "policy"}:
            raise ValueError(f"{kv_cache} supports remasking='fastdllm', 'eb_sampler' or 'policy'")
        if model_type != "LLaDA":
            raise ValueError(f"{kv_cache} currently supports LLaDA only")
        if attention_mask is not None and not bool(attention_mask.bool().all()):
            raise ValueError(f"{kv_cache} does not support padded prompts")
        if "replace_position" not in inspect.signature(model.forward).parameters:
            raise ValueError(
                f"{kv_cache} requires common.models.modeling_llada_fastdllm.LLaDAModelLM"
            )
    if eb_error_proxy not in {"confidence", "entropy"}:
        raise ValueError("eb_error_proxy must be confidence or entropy")
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
        if sampling_mode != "bernoulli-deterministic":
            raise ValueError("policy decoding uses sampling_mode='bernoulli-deterministic'")
        if policy_accept_threshold is not None and not 0.0 <= policy_accept_threshold < 1.0:
            raise ValueError("policy_accept_threshold must be in [0, 1)")
    elif remasking == "fastdllm":
        if thres is None:
            raise ValueError("thres must be provided for remasking='fastdllm'")
    elif remasking == "eb_sampler":
        if eb_gamma is None or not torch.isfinite(torch.tensor(eb_gamma)) or eb_gamma < 0:
            raise ValueError("eb_gamma must be a finite nonnegative value")
    elif remasking == "low_confidence":
        if steps is None:
            raise ValueError("steps must be provided for remasking='low_confidence'")
    elif remasking == "learn2pd_teacher":
        if learn2pd_reference_tokens is None:
            raise ValueError("learn2pd_reference_tokens is required for Learn2PD teacher rollout")
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    B, prompt_L = prompt.shape
    L = gen_length
    if learn2pd_reference_tokens is not None:
        if learn2pd_reference_tokens.shape != (B, L):
            raise ValueError("learn2pd_reference_tokens must have shape [batch, gen_length]")
        if learn2pd_reference_tokens.dtype != prompt.dtype:
            raise TypeError("learn2pd_reference_tokens and prompt must use the same token dtype")
        learn2pd_reference_tokens = learn2pd_reference_tokens.to(prompt.device)
    x = torch.full((B, L + prompt_L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_L] = prompt
    steps_taken = torch.zeros((B,), dtype=torch.int32, device=x.device)
    num_blocks = L // block_length

    _attn_mask = prepare_generation_attention_mask(
        model=model,
        prompt_attention_mask=attention_mask,
        total_length=L + prompt_L,
        model_type=model_type,
    )
    selected_set_size_history = []

    max_steps = L
    if remasking == "low_confidence":
        assert steps is not None and steps <= L
        tokens_per_step = L // steps
        max_steps = steps

    output_hidden_states = remasking == "policy" and policy_uses_hidden_states(policy)
    cached_denoiser = (
        CachedDenoiser(model, prompt_L, temperature, mask_id, output_hidden_states=output_hidden_states)
        if kv_cache != "no_cache" else None
    )
    for num_block in range(num_blocks):
        start_idx = num_block * block_length
        block_slice = slice(start_idx, min(start_idx + block_length, L))

        for _ in range(block_length):
            generation_part = x[:, prompt_L:]
            mask_index = (generation_part == mask_id) & (
                steps_taken < max_steps
            ).unsqueeze(-1)
            block_mask_index = mask_index[:, block_slice]  # (B, active BL)
            if (~block_mask_index).all():
                break

            if cached_denoiser is not None:
                prediction = cached_denoiser.evaluate(x, block_slice)
            else:
                prediction = evaluate_denoiser_sequence(
                    model=model,
                    sequence=x,
                    prompt_length=prompt_L,
                    model_type=model_type,
                    attention_mask=_attn_mask,
                    output_hidden_states=output_hidden_states,
                    temperature=temperature,
                    forbidden_token_id=mask_id,
                )
            model_output = prediction.model_output
            probs = prediction.generation_probabilities
            x0 = prediction.greedy_tokens

            if state_observer is not None:
                block_masked_full = torch.zeros_like(mask_index)
                block_masked_full[:, block_slice] = block_mask_index
                hidden_states = getattr(model_output, "hidden_states", None)
                state_observer(
                    DenoiserState(
                        sequence=x,
                        generation_logits=prediction.generation_logits,
                        generation_probabilities=probs,
                        greedy_tokens=x0,
                        final_hidden_states=(
                            hidden_states[-1] if hidden_states is not None else None
                        ),
                        masked_positions=mask_index,
                        block_masked_positions=block_masked_full,
                        macro_step=len(selected_set_size_history),
                        block_index=num_block,
                        prompt_length=prompt_L,
                    )
                )

            proposed_tokens = x0
            if remasking == "policy":
                unmask = _policy_unmask_decisions(
                    mask_index, block_mask_index, probs, steps_taken, block_slice, L,
                    policy, confidences_top_p, model_output, prompt_L, model_type,
                    temperature_policy, policy_accept_threshold,
                )
            elif remasking == "fastdllm":
                unmask = _confidence_threshold_unmask(
                    block_mask_index, probs, block_slice, thres, mask_id
                )
            elif remasking == "eb_sampler":
                unmask = entropy_bounded_actions(
                    block_mask_index, probs, block_slice,
                    gamma=eb_gamma, error_proxy=eb_error_proxy,
                )
            elif remasking == "low_confidence":
                unmask = _low_confidence_unmask(
                    block_mask_index, probs, x0, block_slice, tokens_per_step
                )
            else:  # learn2pd_teacher
                from teacher.learn2pd_oracle import egp_decision

                egp_eligible = torch.zeros_like(mask_index)
                egp_eligible[:, block_slice] = block_mask_index
                decision = egp_decision(x0, learn2pd_reference_tokens, egp_eligible)
                unmask = decision.commit_mask | decision.fallback_mask
                proposed_tokens = torch.where(
                    decision.fallback_mask, learn2pd_reference_tokens, x0
                )

            x[:, prompt_L:] = torch.where(unmask, proposed_tokens, generation_part)
            selected_set_size_history.append(unmask.sum(dim=-1).to(torch.int32))
            # Count a step only for batch elements that had work to do.
            step_had_work = mask_index if remasking == "eb_sampler" else block_mask_index
            steps_taken += step_had_work.any(dim=-1).int()

    selected_set_sizes = (
        torch.stack(selected_set_size_history, dim=1)
        if selected_set_size_history
        else torch.zeros((B, 0), dtype=torch.int32, device=x.device)
    )
    return GenerationResult(
        sequences=x, steps_taken=steps_taken, selected_set_sizes=selected_set_sizes
    )


def entropy_bounded_actions(
    block_mask_index: torch.Tensor,
    probabilities: torch.Tensor,
    block_slice: slice,
    *,
    gamma: float,
    error_proxy: str = "confidence",
) -> torch.Tensor:
    """Select the largest entropy-bounded prefix from each active block.

    This is Equation (2) of Ben-Hamu et al.'s EB-Sampler. Candidates are
    ordered by the requested local-error proxy, then the largest prefix whose
    summed categorical entropy minus its maximum entropy is at most ``gamma``
    is committed. The first candidate is therefore always selected, including
    at ``gamma=0``.
    """
    if gamma < 0 or not torch.isfinite(torch.tensor(gamma)):
        raise ValueError("gamma must be a finite nonnegative value")
    if error_proxy not in {"confidence", "entropy"}:
        raise ValueError("error_proxy must be confidence or entropy")
    if probabilities.ndim != 3 or block_mask_index.ndim != 2:
        raise ValueError("expected probabilities [B,L,V] and block mask [B,BL]")

    # Accumulate entropy in float32: the bound can be much smaller than the
    # bfloat16 spacing used for the backbone forward pass.
    entropy = torch.special.entr(probabilities.float()).sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    selected = torch.zeros(
        probabilities.shape[:2], dtype=torch.bool, device=probabilities.device
    )
    start = 0 if block_slice.start is None else block_slice.start
    for row in range(probabilities.shape[0]):
        local_indices = torch.nonzero(block_mask_index[row], as_tuple=False).flatten()
        if local_indices.numel() == 0:
            continue
        global_indices = local_indices + start
        if error_proxy == "confidence":
            order = torch.argsort(
                confidence[row, global_indices], descending=True, stable=True
            )
        else:
            order = torch.argsort(
                entropy[row, global_indices], descending=False, stable=True
            )
        ordered_indices = global_indices[order]
        ordered_entropy = entropy[row, ordered_indices]
        bound = ordered_entropy.cumsum(dim=0) - ordered_entropy.cummax(dim=0).values
        prefix_size = int(bound.le(gamma).sum().item())
        selected[row, ordered_indices[:prefix_size]] = True
    return selected


def _policy_unmask_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    model_type: str | None,
    temperature_policy: float = 1.0,
    policy_accept_threshold: float | None = None,
) -> torch.Tensor:
    """Deterministic policy decision within the active block.

    The policy scores every response position (full context).  With
    ``policy_accept_threshold`` tau it commits positions with
    ``sigmoid(logit) > tau`` (Learn2PD's rule); otherwise the Bernoulli MAP
    ``logit >= 0``.  An active row always commits at least its argmax.
    """
    sampling_mask = torch.zeros_like(mask_index)
    sampling_mask[:, block_slice] = block_mask_index
    hidden_states = model_output.hidden_states[-1] if policy_uses_hidden_states(policy) else None
    policy_inputs = policy_inputs_from_prediction(
        policy, mask_index, probs, hidden_states, policy_timestep(steps_taken, L),
        confidences_top_p, prompt_L, model_type,
    )
    policy_logits = policy(*policy_inputs)
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    if policy_accept_threshold is not None:
        b = policy_logits.float().sigmoid().gt(policy_accept_threshold) & sampling_mask
    else:
        b = (policy_logits >= 0) & sampling_mask
    needs_fallback = ~b.any(dim=-1) & sampling_mask.any(dim=-1)
    if needs_fallback.any():
        masked_logits = policy_logits.masked_fill(~sampling_mask, -torch.inf)
        force_idx = masked_logits.argmax(dim=-1)
        batch_indices = needs_fallback.nonzero(as_tuple=True)[0]
        b[batch_indices, force_idx[batch_indices]] = True

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = b[:, block_slice]
    return unmask


def _confidence_threshold_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float | torch.Tensor,
    mask_id: int | None = None,
) -> torch.Tensor:
    # Confidence is that of the committed greedy token, which never is mask_id.
    block_probs = probs[:, block_slice]
    if mask_id is not None and 0 <= mask_id < block_probs.shape[-1]:
        top2 = block_probs.topk(2, dim=-1)
        confidence_masked = torch.where(
            top2.indices[..., 0].eq(mask_id), top2.values[..., 1], top2.values[..., 0]
        )
    else:
        confidence_masked = block_probs.max(dim=-1).values.clone()

    # Only consider masked positions in current block
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    empty_rows = ~unmask_local.any(dim=-1)
    if empty_rows.any():
        force_idx = torch.argmax(confidence_masked[empty_rows], dim=-1)
        unmask_local[empty_rows] = unmask_local[empty_rows].scatter(
            1, force_idx.unsqueeze(-1), True
        )

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


def _low_confidence_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    block_slice: slice,
    tokens_per_step: int,
) -> torch.Tensor:
    B = block_mask_index.shape[0]
    confidence_block = torch.gather(
        probs[:, block_slice], dim=-1, index=x0[:, block_slice].unsqueeze(-1)
    ).squeeze(-1)
    confidence_masked = torch.where(block_mask_index, confidence_block, -torch.inf)

    num_masked_block = block_mask_index.sum(dim=-1)
    k = torch.clamp(num_masked_block, max=tokens_per_step)
    max_k = k.max().item()

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    if max_k == 0:
        return unmask

    _, topk_indices = torch.topk(confidence_masked, k=max_k, dim=-1)
    positions = torch.arange(max_k, device=x0.device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < k.unsqueeze(-1)
    unmask_local = torch.zeros_like(block_mask_index, dtype=torch.bool)
    unmask_local.scatter_(1, topk_indices, valid_mask)
    unmask[:, block_slice] = unmask_local
    return unmask
