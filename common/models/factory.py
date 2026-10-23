"""Build an unmasking policy from a training/evaluation config."""

from common.models.hidden_residual import DiTHiddenResidualPolicy
from common.models.learn2pd import Learn2PDPolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper


def build_policy(config, backbone_hidden_size: int) -> PolicyHFWrapper:
    """Return the wrapped policy described by ``config``.

    ``policy_type: dit_confidence`` with ``policy_input_type`` ``confidence``,
    ``hidden_only`` or ``hidden`` (+ ``policy_hidden_residual``: the joint
    confidence+hidden policy); ``policy_type: learn2pd`` is the Learn2PD filter.
    """
    if config.policy_type == "learn2pd":
        return PolicyHFWrapper(Learn2PDPolicy(block_length=config.block_length), "learn2pd")
    if config.policy_type != "dit_confidence":
        raise ValueError(f"unsupported policy_type {config.policy_type!r}")
    hidden_dim = config.policy_hidden_dim or 128
    kwargs = dict(
        hidden_dim=hidden_dim,
        feedforward_dim=config.policy_feedforward_dim or 4 * hidden_dim,
        num_heads=config.policy_num_heads,
        dropout=config.policy_dropout,
        time_embed_dim=config.policy_time_embed_dim,
        smart_init=config.policy_smart_init,
        confidences_top_p=config.confidences_top_p,
        num_blocks=config.policy_num_blocks,
        time_period=config.policy_time_period,
    )
    if config.policy_hidden_residual:
        if config.policy_input_type != "hidden":
            raise ValueError("the joint hidden-residual policy requires policy_input_type=hidden")
        core = DiTHiddenResidualPolicy(backbone_hidden_dim=backbone_hidden_size, **kwargs)
    else:
        if config.policy_input_type == "hidden":
            raise ValueError("policy_input_type=hidden requires policy_hidden_residual=true")
        core = DiTConfidencePolicy(
            input_type=config.policy_input_type,
            backbone_hidden_dim=(
                backbone_hidden_size if config.policy_input_type != "confidence" else None
            ),
            hidden_only_norm=config.policy_hidden_only_norm,
            use_top1_margin=config.use_top1_margin,
            **kwargs,
        )
    return PolicyHFWrapper(core, config.policy_type)
