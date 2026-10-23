"""Offline teacher forcing of an unmasking policy on a frozen diffusion LM.

Each training row is one teacher trajectory: a sequence of decoding states
(token ids, masked positions, active block) and the positions the teacher
committed at each state.  For every state the frozen denoiser is re-evaluated
to recompute the policy's input features, and the policy is trained by
maximum likelihood of the teacher's commit set under its independent
Bernoulli decisions over the active block's masked positions.
"""

import warnings
from typing import Any

import torch
from torch import nn
from torch.utils.data import RandomSampler
from trl.trainer.grpo_trainer import GRPOTrainer

from common.generation.generation import evaluate_denoiser_sequence
from common.generation.generation import generation_aligned_hidden_states
from common.generation.generation import policy_timestep
from common.generation.generation import prefix_cache_block_predictions
from common.generation.sampling import bernoulli_batch_loglik


def _unused_reward(**kwargs):
    raise RuntimeError("offline teacher forcing does not compute rewards")


class Trainer(GRPOTrainer):
    """TRL's GRPOTrainer supplies batching, optimization and checkpointing;
    the loss is the offline teacher-forcing objective only."""

    _REPORT_METRICS = frozenset(
        {"loss", "train_loss", "grad_norm", "learning_rate", "epoch", "teacher_forcing_loss"}
    )

    def __init__(self, model, dllm: nn.Module, args, train_dataset, processing_class, callbacks=None):
        super().__init__(
            model=model,
            reward_funcs=[_unused_reward],
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            peft_config=None,
        )
        torch._dynamo.config.capture_scalar_outputs = True
        self.dllm = torch.compile(dllm)
        self.dllm.eval()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        buffered = self._metrics[mode]
        for name in tuple(buffered):
            if name not in self._REPORT_METRICS:
                del buffered[name]
        logs = {name: value for name, value in logs.items()
                if name.removeprefix("eval_") in self._REPORT_METRICS}
        super().log(logs, start_time)

    def _get_train_sampler(self, dataset=None):
        """Use each teacher trajectory exactly once per epoch."""
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return RandomSampler(dataset if dataset is not None else self.train_dataset, generator=generator)

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, list):
            return self._prepare_g_trace_teacher_inputs(inputs)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("returning outputs is not supported")
        model.train()
        return self._compute_teacher_forcing_loss(model, inputs)

    def _compute_teacher_forcing_loss(self, model, inputs):
        """Per-trajectory mean over states of the per-candidate Bernoulli NLL."""
        trajectory_losses = []
        for batch_output in inputs["policy_outputs"]:
            sampling_masks = batch_output["sampling_masks"]
            samples = batch_output["samples"]
            _, timesteps, _ = sampling_masks.shape
            timestep_losses = []
            timestep_active = []
            timestep_bs = self.args.timestep_batch_size or timesteps
            for start in range(0, timesteps, timestep_bs):
                stop = start + timestep_bs
                policy_inputs = [value[:, start:stop] for value in batch_output["policy_inputs"]]
                masks = sampling_masks[:, start:stop]
                with torch.amp.autocast("cuda", enabled=self.args.fp16):
                    logits = model(*policy_inputs)
                log_probs = bernoulli_batch_loglik(
                    samples[:, start:stop], logits, masks, dtype=self.args.loglikelihood_dtype
                )
                del logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # The Bernoulli log likelihood is joint over all active candidates;
                # normalize it so the loss scale does not depend on block size.
                timestep_losses.append(-log_probs / masks.sum(dim=-1).clamp_min(1))
                timestep_active.append(masks.any(dim=-1))
            losses = torch.cat(timestep_losses, dim=1)
            active = torch.cat(timestep_active, dim=1)
            trajectory_losses.append(
                (losses * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
            )

        loss = torch.cat(trajectory_losses).mean() * self.args.teacher_forcing_weight
        self._metrics["train"]["teacher_forcing_loss"].append(
            self.accelerator.gather_for_metrics(loss.detach()).mean().item()
        )
        return loss

    def _backbone_attention_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        """Adapt a [B, L] token-validity mask to the backbone's attention input.

        LLaDA consumes the [B, L] mask directly.  Dream's SDPA attention needs a
        key-padding mask [B, 1, 1, L], or None when nothing is padded.
        """
        if mask is None or self.args.model_type != "Dream":
            return mask
        valid = mask.bool()
        return None if bool(valid.all()) else valid[:, None, None, :]

    @torch.no_grad()
    def _prepare_g_trace_teacher_inputs(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Recompute frozen-denoiser policy features at every teacher state."""
        device = self.accelerator.device
        generation_lengths = {int(row["generation_length"]) for row in rows}
        if len(generation_lengths) != 1:
            raise ValueError("a teacher batch must have one generation length")
        generation_length = generation_lengths.pop()
        uses_hidden = self.args.policy_input_type in {"hidden", "hidden_only"}
        replay_mismatches = 0
        replay_targets = 0
        holders = [{"policy_inputs": [], "samples": [], "sampling_masks": []} for _ in rows]
        flat = [
            (sample_index, transition_index, row, transition)
            for sample_index, row in enumerate(rows)
            for transition_index, transition in enumerate(row["transitions"])
        ]
        denoiser_batch_size = self.args.generation_batch_size
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        prefix_cache = self.args.g_trace_feature_kv_cache == "prefix_cache"
        if prefix_cache:
            # Each chunk holds states of one trajectory and active block, which
            # share the block-entry prefix cache (see CachedDenoiser).
            chunks = []
            for item in flat:
                key = (item[0], int(item[3]["active_block"]))
                if (
                    chunks and chunks[-1][0] == key
                    and len(chunks[-1][1]) < denoiser_batch_size
                ):
                    chunks[-1][1].append(item)
                elif chunks and chunks[-1][0][0] == key[0] and chunks[-1][0][1] > key[1]:
                    raise ValueError("teacher transitions must visit blocks in order")
                else:
                    chunks.append((key, [item]))
            chunks = [chunk for _, chunk in chunks]
        else:
            chunks = [
                flat[start : start + denoiser_batch_size]
                for start in range(0, len(flat), denoiser_batch_size)
            ]
        for real_chunk in chunks:
            real_chunk_size = len(real_chunk)
            chunk = real_chunk
            if real_chunk_size < denoiser_batch_size and not prefix_cache:
                # Pad to a fixed batch shape for the compiled denoiser.
                chunk = real_chunk + [real_chunk[-1]] * (denoiser_batch_size - real_chunk_size)
            max_prompt_length = self.args.max_prompt_length
            if any(int(row["prompt_length"]) > max_prompt_length for _, _, row, _ in chunk):
                raise ValueError("teacher prompt exceeds max_prompt_length")
            total_length = max_prompt_length + generation_length
            sequences = torch.full(
                (len(chunk), total_length), int(pad_token_id), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(chunk), total_length), dtype=torch.bool, device=device
            )
            for batch_index, (_, _, row, transition) in enumerate(chunk):
                values = torch.tensor(transition["input_ids"], dtype=torch.long, device=device)
                mask = torch.tensor(transition["attention_mask"], dtype=torch.bool, device=device)
                expected_length = int(row["prompt_length"]) + generation_length
                if values.numel() != expected_length or mask.numel() != expected_length:
                    raise ValueError("teacher sequence length does not match metadata")
                sequences[batch_index, -expected_length:] = values
                attention_mask[batch_index, -expected_length:] = mask

            feature_prompt_length = max_prompt_length
            if prefix_cache:
                feature_prompt_length = int(chunk[0][2]["prompt_length"])
                pad = max_prompt_length - feature_prompt_length
                if not bool(attention_mask[:, pad:].all()):
                    raise ValueError("prefix-cache teacher features require unpadded states")
                block_start = int(chunk[0][3]["active_block"]) * self.args.block_length
                prediction = prefix_cache_block_predictions(
                    # Uncompiled: cached forwards vary in batch and query length.
                    model=getattr(self.dllm, "_orig_mod", self.dllm),
                    states=sequences[:, pad:],
                    prompt_length=feature_prompt_length,
                    block_slice=slice(block_start, block_start + self.args.block_length),
                    mask_id=self.args.mask_id,
                    temperature=0.0,
                    output_hidden_states=uses_hidden,
                )
            else:
                prediction = evaluate_denoiser_sequence(
                    model=self.dllm,
                    sequence=sequences,
                    prompt_length=max_prompt_length,
                    model_type=self.args.model_type,
                    attention_mask=self._backbone_attention_mask(attention_mask),
                    temperature=0.0,
                    output_hidden_states=uses_hidden,
                )
            response_hidden_states = (
                generation_aligned_hidden_states(
                    prediction.model_output.hidden_states[-1],
                    feature_prompt_length,
                    self.args.model_type,
                )
                if uses_hidden
                else None
            )
            for batch_index, (sample_index, transition_index, row, transition) in enumerate(
                chunk[:real_chunk_size]
            ):
                sampling_mask = torch.zeros((1, generation_length), dtype=torch.bool, device=device)
                eligible = torch.tensor(transition["eligible_positions"], dtype=torch.long, device=device)
                sampling_mask[0, eligible] = True
                teacher_sample = torch.zeros((1, generation_length), dtype=torch.bool, device=device)
                teacher_sample[
                    0,
                    torch.tensor(transition["target_commit_positions"], dtype=torch.long, device=device),
                ] = True
                response_mask = torch.tensor(
                    transition["mask_bitmap"][-generation_length:], dtype=torch.bool, device=device
                ).unsqueeze(0)
                confidence_topk = prediction.generation_probabilities[
                    batch_index : batch_index + 1
                ].topk(self.args.confidences_top_p, dim=-1).values
                block_start = int(transition["active_block"]) * self.args.block_length
                block_slice = slice(block_start, block_start + self.args.block_length)
                # Eligibility is exactly the masked positions in the active
                # block, even though the policy observes the full response.
                expected_eligible = torch.zeros_like(response_mask)
                expected_eligible[:, block_slice] = response_mask[:, block_slice]
                if not torch.equal(sampling_mask, expected_eligible):
                    raise ValueError("teacher eligible positions must equal active-block masked positions")
                if (teacher_sample & ~sampling_mask).any():
                    raise ValueError("teacher commits must be within the active eligible block")
                timestep = policy_timestep(
                    torch.tensor([transition_index], device=device), generation_length
                )
                features = (
                    response_hidden_states[batch_index : batch_index + 1].detach().clone()
                    if uses_hidden
                    else confidence_topk.detach()
                )
                policy_inputs = (response_mask, features, timestep)
                if self.args.policy_hidden_residual:
                    policy_inputs += (confidence_topk.detach(),)
                holders[sample_index]["samples"].append(teacher_sample)
                holders[sample_index]["sampling_masks"].append(sampling_mask)
                holders[sample_index]["policy_inputs"].append(policy_inputs)

                if transition.get("target_token_ids"):
                    positions = torch.tensor(
                        transition["target_commit_positions"], dtype=torch.long, device=device
                    )
                    expected = torch.tensor(transition["target_token_ids"], device=device)
                    replay_mismatches += int(
                        (prediction.greedy_tokens[batch_index, positions] != expected).sum().item()
                    )
                    replay_targets += int(expected.numel())

        if replay_mismatches and not getattr(self, "_warned_replay_mismatch", False):
            warnings.warn(
                "Batched teacher feature replay changed "
                f"{replay_mismatches}/{replay_targets} stored top-1 targets; "
                "using the stored teacher labels."
            )
            self._warned_replay_mismatch = True

        policy_outputs = []
        for holder in holders:
            policy_outputs.append(
                {
                    "sampling_masks": torch.stack(holder["sampling_masks"], dim=1),
                    "samples": torch.stack(holder["samples"], dim=1),
                    "policy_inputs": tuple(
                        torch.stack(column, dim=1) for column in zip(*holder["policy_inputs"])
                    ),
                }
            )
        return {"policy_outputs": policy_outputs}
