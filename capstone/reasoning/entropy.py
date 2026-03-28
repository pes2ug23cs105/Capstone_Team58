from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy for each token position from raw logit scores.

    H(t) = -sum_v  p(v|t) * log p(v|t)

    A high entropy value means the teacher was uncertain at that position,
    which corresponds to active deliberation / a reasoning decision point.

    Args:
        logits : (seq_len, vocab_size)  raw logits (float32).

    Returns:
        entropy : (seq_len,)  non-negative entropy values (nats).
    """
    probs = F.softmax(logits.float(), dim=-1)          # (seq_len, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (seq_len, V)
    entropy = -(probs * log_probs).sum(dim=-1)         # (seq_len,)
    return entropy
