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

    def estimateVisualAttentionMass(
        self,
        logits: torch.Tensor,
        has_image: bool,
    ) -> torch.Tensor:
        """
        Fast proxy for per-step visual attention mass.

        For smoke/preflight mode we avoid storing full attention tensors and
        instead derive a compact scalar trace from token uncertainty.
        """
        seq_len = logits.size(0)
        if seq_len == 0:
            return torch.zeros(0, dtype=torch.float32)
        if not has_image:
            return torch.zeros(seq_len, dtype=torch.float32)

        probs = torch.softmax(logits.float(), dim=-1)
        peak_confidence = probs.max(dim=-1).values.clamp(0.0, 1.0)
        # Higher uncertainty implies broader context use; we treat this as
        # a lightweight visual-attention proxy in approx mode.
        return (1.0 - peak_confidence).to(torch.float32)

    def estimateDisengagementPoint(
        self,
        entropy: torch.Tensor,
        visual_attention_mass: torch.Tensor,
        stability_eps: float = 0.02,
        attention_threshold: float = 0.15,
        window: int = 3,
    ) -> int:
        """
        Estimate t* as first step where entropy stabilizes and visual proxy
        attention remains below threshold for a short window.
        """
        seq_len = int(entropy.numel())
        if seq_len == 0:
            return 0
        if seq_len == 1:
            return 0

        if visual_attention_mass.numel() != seq_len:
            raise ValueError(
                "visual_attention_mass length must match entropy length "
                f"({visual_attention_mass.numel()} != {seq_len})"
            )

        ent = entropy.float()
        vis = visual_attention_mass.float()
        max_start = max(1, seq_len - window + 1)

        for i in range(1, max_start):
            ent_window = ent[i : i + window]
            vis_window = vis[i : i + window]
            if ent_window.numel() < window or vis_window.numel() < window:
                continue

            ent_deltas = torch.abs(ent_window[1:] - ent_window[:-1])
            entropy_stable = bool((ent_deltas <= stability_eps).all().item())
            visual_low = bool((vis_window <= attention_threshold).all().item())

            if entropy_stable and visual_low:
                return int(i)

        return seq_len - 1
