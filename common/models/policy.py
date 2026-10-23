import torch
from torch import nn
from transformers import PretrainedConfig
from transformers import PreTrainedModel

from common.models.policy_layers import RoPEDiTBlock
from common.models.policy_layers import sinusoidal_time_embedding

INPUT_TYPES = {"confidence", "hidden", "hidden_only"}


class PolicyConfig(PretrainedConfig):
    model_type = "policy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PolicyHFWrapper(PreTrainedModel):
    """Hugging Face wrapper so the Trainer can save and load unmasking policies."""

    config_class = PolicyConfig

    def __init__(
        self,
        base_policy: nn.Module,
        policy_type: str,
        config: PolicyConfig | None = None,
    ):
        super().__init__(config or PolicyConfig())
        self.base_policy = base_policy
        self.policy_type = policy_type
        if policy_type == "learn2pd":
            self.config.learn2pd_block_length = base_policy.block_length
        if getattr(base_policy, "policy_hidden_residual", False):
            self.config.policy_hidden_residual = True
            self.config.policy_hidden_residual_joint = True
            self.config.policy_hidden_residual_mode = "hidden_stream_residual"
            self.config.policy_hidden_stream_num_blocks = 1
        if getattr(base_policy, "input_type", "confidence") != "confidence":
            self.config.policy_input_type = base_policy.input_type
            if not getattr(base_policy, "hidden_only_norm", True):
                self.config.policy_hidden_only_norm = False
            self.config.policy_backbone_hidden_dim = base_policy.backbone_hidden_dim
        if getattr(base_policy, "use_top1_margin", False):
            self.config.use_top1_margin = True

    def forward(self, *args, **kwargs):
        args = tuple(arg.to(self.dtype) if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {k: v.to(self.dtype) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.base_policy(*args, **kwargs)

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, *args, **kwargs):
        pass

    def tie_weights(self):
        pass


class DiTConfidencePolicy(nn.Module):
    """AdaLN DiT over per-position features, conditioned on the mask and step.

    ``input_type`` selects the per-position feature: top-k denoiser
    confidences (``confidence``), or the denoiser's final hidden state with
    (``hidden``) or without (``hidden_only``) the confidence-side DiT stream.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        feedforward_dim: int = 512,
        num_heads: int = 1,
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        confidences_top_p: int = 1,
        smart_init: float | None = None,
        num_blocks: int = 1,
        time_period: float = 1,
        use_top1_margin: bool = False,
        input_type: str = "confidence",
        backbone_hidden_dim: int | None = None,
        hidden_only_norm: bool = True,
    ):
        super().__init__()
        if input_type not in INPUT_TYPES:
            raise ValueError(f"policy input_type must be one of {sorted(INPUT_TYPES)}")
        if not hidden_only_norm and input_type != "hidden_only":
            raise ValueError("hidden_only_norm=False applies to input_type=hidden_only only")
        if input_type != "confidence" and (not backbone_hidden_dim or backbone_hidden_dim < 1):
            raise ValueError("hidden-state inputs require the backbone hidden dimension")
        if use_top1_margin and (confidences_top_p != 2 or input_type != "confidence"):
            raise ValueError("top1-margin input requires confidence input with confidences_top_p=2")
        self.input_type = input_type
        self.backbone_hidden_dim = backbone_hidden_dim
        self.hidden_only_norm = hidden_only_norm
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.time_period = time_period
        self.num_blocks = num_blocks
        self.confidences_top_p = confidences_top_p
        self.use_top1_margin = use_top1_margin

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if input_type == "hidden":
            self.hidden_proj = nn.Linear(backbone_hidden_dim, self.hidden_dim)
        elif input_type == "hidden_only":
            # The LLaDA hidden input is already post-final-RMSNorm; the extra
            # LayerNorm can be ablated.  (Parameter names kept for checkpoints.)
            self.output_row_norm = (
                nn.LayerNorm(backbone_hidden_dim) if hidden_only_norm else nn.Identity()
            )
            self.output_row_proj = nn.Linear(backbone_hidden_dim, self.hidden_dim)
        else:
            self.confidence_proj = nn.Linear(confidences_top_p, self.hidden_dim)
        self.mask_embedding = nn.Embedding(2, self.hidden_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                RoPEDiTBlock(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

        if smart_init is not None:
            self.apply_smart_init(smart_init)

    def apply_smart_init(self, target_logit: float):
        """Identity AdaLN modulation and output bias ``target_logit``."""
        with torch.no_grad():
            for block in self.transformer_blocks:
                block.ada_conditioning.weight.data.zero_()
                block.ada_conditioning.bias.data.zero_()
            self.output_proj.bias.data.fill_(target_logit)

    def forward(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        *,
        stream_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """:param m: (*B,L) mask with 1=masked, 0=unmasked
        :param c: (*B,L,F) per-position features (confidences or hidden states)
        :param timestep: (*B,1) normalized controller step
        :param stream_delta: optional (*B,L,hidden_dim) addition to the input stream
        :return: (*B,L) unmasking logits
        """
        features = self.encode_features(m, c, timestep, stream_delta=stream_delta)
        return self.output_proj(features).squeeze(-1)

    def encode_features(
        self,
        m: torch.Tensor,
        c: torch.Tensor,
        timestep: torch.Tensor,
        *,
        stream_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the contextual token features used before the output head."""
        *batch_dims, length, _ = c.shape
        if m.shape != (*batch_dims, length):
            raise ValueError("mask shape must match feature positions")
        if timestep.shape != (*batch_dims, 1):
            raise ValueError("timestep must contain one value per batch element")
        conditioning = self.mask_embedding(m.int())
        time_embedding = sinusoidal_time_embedding(
            timestep, self.time_embed_dim, max_period=self.time_period
        ).to(conditioning.dtype)
        time_embedding = time_embedding.expand((*([-1] * len(batch_dims)), length, -1))
        conditioning = conditioning + self.time_mlp(time_embedding)
        if self.input_type == "hidden":
            features = self.hidden_proj(c.detach())
        elif self.input_type == "hidden_only":
            features = self.output_row_proj(self.output_row_norm(c.detach()))
        else:
            if self.use_top1_margin:
                c = torch.stack((c[..., 0], c[..., 0] - c[..., 1]), dim=-1)
            features = self.confidence_proj(c)
        if stream_delta is not None:
            if stream_delta.shape != features.shape:
                raise ValueError("stream_delta must match confidence-stream features")
            features = features + stream_delta
        original_shape = features.shape
        features = features.view(-1, length, self.hidden_dim)
        conditioning = conditioning.view(-1, length, self.hidden_dim)
        for block in self.transformer_blocks:
            features = block(features, conditioning)
        return self.final_norm(features.view(original_shape))
