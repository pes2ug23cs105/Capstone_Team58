from __future__ import annotations

import torch


def generate_reasoning_mask(
    entropy: torch.Tensor,
    threshold: float = 1.5,
) -> torch.Tensor:
    """
    Convert per-token entropy scores into a boolean reasoning mask.

    Tokens with entropy >= threshold are flagged as "reasoning tokens."
    These are the positions where the teacher was actively deliberating,
    and they will receive a higher distillation loss weight during
    student training.

    Args:
        entropy   : (seq_len,)  per-token entropy values (nats).
        threshold : Minimum entropy to classify a token as a reasoning token.
                    Default 1.5 nats — tunable via config/model_config.yaml.

    Returns:
        mask : (seq_len,) bool tensor.  True = reasoning token.
    """
    return entropy >= threshold
