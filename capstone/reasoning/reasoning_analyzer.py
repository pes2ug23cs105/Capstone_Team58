from __future__ import annotations

import torch

from reasoning.entropy import compute_token_entropy
from reasoning.mask_generator import generate_reasoning_mask


class ReasoningAnalyzer:
    """
    Combines entropy computation and decision-point detection into a single
    component, matching the ReasoningAnalyzer class in the master diagram.

    Methods mirror the class diagram exactly:
        computeEntropy()      -> per-token Shannon entropy
        detectDecisionPoints() -> indices of high-entropy positions
        generateMask()         -> boolean reasoning mask
    """

    def __init__(self, entropy_threshold: float = 1.5):
        self.entropy_threshold = entropy_threshold

    def computeEntropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (seq_len, vocab_size) raw logit tensor.
        Returns:
            entropy : (seq_len,) float tensor in nats.
        """
        return compute_token_entropy(logits)

    def detectDecisionPoints(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Return token positions (indices) where entropy exceeds threshold.

        Args:
            entropy : (seq_len,) float tensor.
        Returns:
            indices : 1-D LongTensor of high-entropy token positions.
        """
        return torch.where(entropy >= self.entropy_threshold)[0]

    def generateMask(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entropy : (seq_len,) float tensor.
        Returns:
            mask : (seq_len,) bool tensor.  True = reasoning token.
        """
        return generate_reasoning_mask(entropy, threshold=self.entropy_threshold)
