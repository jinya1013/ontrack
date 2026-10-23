from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def sinusoidal_time_embedding(
    timesteps: torch.Tensor, dim: int, max_period: float = 10000, dtype=torch.bfloat16
):
    half = dim // 2
    freqs = torch.exp(
        -torch.log(
            torch.tensor(max_period, dtype=torch.float32, device=timesteps.device)
        )
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.unsqueeze(-1).float() * freqs.view(*([1] * timesteps.ndim), -1)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding.to(dtype)


class RoPEDiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        batch_first: bool,
        activation: str | Callable,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.batch_first = batch_first

        # Attention projections
        # TODO: make qkv just a single d_model -> 3xd_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE setup
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        if isinstance(activation, str):
            assert hasattr(F, activation)
            self.activation_fn = getattr(F, activation)
        else:
            assert isinstance(activation, Callable)
            self.activation_fn = activation

        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # AdaLN scales and shifts -- 6 in total for
        # 2 * (scale and shift) + 2 * (just scale)
        # Exactly as in the original DiT paper, Figure 3
        self.ada_conditioning = nn.Linear(d_model, 6 * d_model)

        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _get_rope_embed(self, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        cos_emb = torch.cos(angles)
        sin_emb = torch.sin(angles)
        return cos_emb, sin_emb

    def forward(
        self, src, cond, src_mask=None, src_key_padding_mask=None, is_causal=False,
    ):
        if self.batch_first:
            batch_size, seq_len = src.shape[:2]
        else:
            seq_len, batch_size = src.shape[:2]

        ada_mod = self.ada_conditioning(cond)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = ada_mod.chunk(6, dim=-1)

        # first adaLN
        residual = src
        src = self.norm1(src)
        src = src * (1 + gamma_1) + beta_1

        if self.batch_first:
            # src shape: (batch, seq_len, d_model)
            q = self.q_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
            k = self.k_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
            v = self.v_proj(src).view(batch_size, seq_len, self.nhead, self.head_dim)
        else:
            # src shape: (seq_len, batch, d_model)
            q = self.q_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)
            k = self.k_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)
            v = self.v_proj(src).view(seq_len, batch_size, self.nhead, self.head_dim)

        # Apply RoPE
        cos_emb, sin_emb = self._get_rope_embed(seq_len, src.device)

        if self.batch_first:
            cos_emb = cos_emb.unsqueeze(0).unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(0).unsqueeze(2)
        else:
            cos_emb = cos_emb.unsqueeze(1).unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(1).unsqueeze(2)

        q = apply_rotary_pos_emb(q, cos_emb, sin_emb)
        k = apply_rotary_pos_emb(k, cos_emb, sin_emb)

        if self.batch_first:
            # (batch, seq_len, nhead, head_dim) -> (batch, nhead, seq_len, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            # (seq_len, batch, nhead, head_dim) -> (batch, nhead, seq_len, head_dim)
            q = q.permute(1, 2, 0, 3)
            k = k.permute(1, 2, 0, 3)
            v = v.permute(1, 2, 0, 3)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=src_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        if self.batch_first:
            # (batch, nhead, seq_len, head_dim) -> (batch, seq_len, d_model)
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.d_model)
            )
        else:
            # (batch, nhead, seq_len, head_dim) -> (seq_len, batch, d_model)
            attn_output = (
                attn_output.permute(2, 0, 1, 3)
                .contiguous()
                .view(seq_len, batch_size, self.d_model)
            )

        attn_output = self.out_proj(attn_output)
        attn_output = attn_output * (1 + alpha_1)
        src = residual + self.dropout1(attn_output)

        # Feed forward
        residual = src
        src = self.norm2(src)
        src = src * (1 + gamma_2) + beta_2  # second adaLN
        src = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        src = src * (1 + alpha_2)
        src = residual + self.dropout2(src)

        return src
