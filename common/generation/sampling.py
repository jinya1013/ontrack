import torch
import torch.nn.functional as F


def bernoulli_batch_loglik(
    samples: torch.Tensor,
    utilities: torch.Tensor,
    mask_index: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-timestep log-likelihood of independent Bernoulli outcomes.

    :param samples: (*B, L) Bernoulli samples (0/1 or bool)
    :param utilities: (*B, L) Bernoulli logits
    :param mask_index: (*B, L) valid positions to include in likelihood
    :param dtype: if not None, computation will be in this dtype
    :return: (*B,) log-likelihood per batch item
    """
    if samples.shape != utilities.shape:
        raise ValueError(
            f"Shape mismatch: samples {samples.shape} vs utilities {utilities.shape}"
        )
    if mask_index.shape != utilities.shape:
        raise ValueError(
            f"Shape mismatch: mask_index {mask_index.shape} vs utilities {utilities.shape}"
        )
    # Work in logit space and promote before the nonlinearity (BF16 sigmoid can
    # round to 1).
    logits = utilities.to(dtype=dtype).masked_fill(~mask_index, 0.0)
    ll = torch.where(samples.bool(), F.logsigmoid(logits), F.logsigmoid(-logits))
    ll = ll.masked_fill(~mask_index, 0.0)
    return ll.sum(dim=-1)


def sample_token_ids(logits, temperature, mask_id=None):
    """Greedy (temperature 0) or Gumbel-max token proposals; never the mask token."""
    if temperature == 0:
        if mask_id is not None and 0 <= mask_id < logits.shape[-1]:
            logits = logits.clone()
            logits[..., mask_id] = -torch.inf
        return logits.argmax(-1)
    if temperature < 0:
        raise ValueError("token temperature must be nonnegative")
    scaled = logits.float() / temperature
    if mask_id is not None and 0 <= mask_id < scaled.shape[-1]:
        scaled = scaled.clone()
        scaled[..., mask_id] = -torch.inf
    uniform = torch.rand(scaled.shape, device=scaled.device, dtype=scaled.dtype).clamp_(
        min=torch.finfo(scaled.dtype).tiny, max=1 - torch.finfo(scaled.dtype).eps
    )
    return (scaled - (-uniform.log()).log()).argmax(-1)
