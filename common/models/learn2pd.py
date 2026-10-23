"""The confidence filter used by Learning to Parallel Decode (Learn2PD)."""

from __future__ import annotations

import torch
from torch import nn


class Learn2PDPolicy(nn.Module):
    """Shared block-local ``d -> d -> d`` confidence MLP.

    Learn2PD treats the confidence vector of a decoding block as one feature
    vector.  The same filter is shared across blocks.  Accepting a complete
    response here keeps the policy compatible with this repository's
    full-context teacher format while preserving strictly block-local mixing.
    """

    input_type = "confidence"

    def __init__(self, block_length: int = 32):
        super().__init__()
        if isinstance(block_length, bool) or not isinstance(block_length, int):
            raise TypeError("block_length must be an integer")
        if block_length <= 0:
            raise ValueError("block_length must be positive")
        self.block_length = block_length
        self.linear1 = nn.Linear(block_length, block_length)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(block_length, block_length)

    def forward(
        self,
        mask: torch.Tensor,
        confidence: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep  # Canonical Learn2PD has no explicit step conditioning.
        if mask.ndim < 2:
            raise ValueError("mask must have shape [..., response_length]")
        if confidence.ndim == mask.ndim + 1:
            if confidence.shape[-1] != 1:
                raise ValueError("Learn2PD requires top-1 confidence only")
            confidence = confidence[..., 0]
        if confidence.shape != mask.shape:
            raise ValueError("confidence and mask shapes must match")

        leading_shape = confidence.shape[:-1]
        response_length = confidence.shape[-1]
        padding = (-response_length) % self.block_length
        if padding:
            confidence = torch.nn.functional.pad(confidence, (0, padding))
        blocks = confidence.reshape(-1, self.block_length)
        logits = self.linear2(self.activation(self.linear1(blocks)))
        return logits.reshape(*leading_shape, -1)[..., :response_length]
