from __future__ import annotations

import torch


class LogitExtractor:
    """
    Converts raw teacher logit tensors into sparse Top-K representations.

    Storing full vocabulary distributions (152K tokens × seq_len) would
    require petabyte-scale disk space.  Top-K sparse caching (K=64) reduces
    this to gigabytes while retaining >99% of the probability mass for
    common reasoning sequences.

    Usage:
        extractor = LogitExtractor(top_k=64)
        indices, log_probs = extractor.extract(logits)   # logits: (seq_len, V)
    """

    def __init__(self, top_k: int = 64):
        self.top_k = top_k

    def extract(
        self, logits: torch.Tensor
    ) -> tuple[list[list[int]], list[list[float]]]:
        """
        Args:
            logits : (seq_len, vocab_size) raw logit tensor (float32, CPU).

        Returns:
            topk_indices : list[list[int]]   shape (seq_len, K)
            topk_logprobs: list[list[float]] shape (seq_len, K), log-softmax values
        """
        log_probs = torch.log_softmax(logits.float(), dim=-1)   # (seq_len, V)
        topk_vals, topk_idx = torch.topk(log_probs, k=self.top_k, dim=-1)  # (seq_len, K)

        return (
            topk_idx.tolist(),    # int indices
            topk_vals.tolist(),   # float log-probs
        )
