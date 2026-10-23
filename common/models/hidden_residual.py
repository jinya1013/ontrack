"""Joint confidence + hidden-state policy.

The confidence DiT stream receives a learned projection of the denoiser's final
hidden state as an additive input-stream residual.  The projection is
zero-initialized, so training starts from the confidence policy's function,
and both branches are trained jointly.
"""

from torch import nn

from common.models.policy import DiTConfidencePolicy


class DiTHiddenResidualPolicy(nn.Module):
    input_type = "hidden"
    policy_hidden_residual = True

    def __init__(self, *, backbone_hidden_dim, **kwargs):
        super().__init__()
        if not backbone_hidden_dim:
            raise ValueError("the hidden stream requires the backbone hidden dimension")
        if kwargs.get("confidences_top_p", 1) != 1:
            raise ValueError("the joint policy uses top-1 confidence")
        self.backbone_hidden_dim = backbone_hidden_dim
        self.confidences_top_p = 1
        self.confidence_policy = DiTConfidencePolicy(input_type="confidence", **kwargs)
        hidden_proj = nn.Linear(backbone_hidden_dim, self.confidence_policy.hidden_dim)
        nn.init.zeros_(hidden_proj.weight)
        nn.init.zeros_(hidden_proj.bias)
        self.residual_policy = nn.ModuleDict({"hidden_proj": hidden_proj})
        for name in ("time_period", "time_embed_dim"):
            setattr(self, name, getattr(self.confidence_policy, name))

    def forward(self, mask, hidden_states, timestep, confidence):
        if hidden_states.shape[-1] != self.backbone_hidden_dim:
            raise ValueError("hidden residual received the wrong backbone dimension")
        hidden = self.residual_policy["hidden_proj"](hidden_states.detach())
        return self.confidence_policy(mask, confidence.detach(), timestep, stream_delta=hidden)
