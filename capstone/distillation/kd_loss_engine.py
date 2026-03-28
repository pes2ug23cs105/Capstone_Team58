from __future__ import annotations

import torch
import torch.nn.functional as F
import yaml


class KDLossEngine:
    """
    Computes the combined training loss for reasoning-aware knowledge distillation.

    Loss formula:
        L = alpha * L_CE  +  (1 - alpha) * L_KD_weighted

    where L_KD_weighted is a KL-divergence loss that:
        - Uses temperature scaling to soften both teacher and student distributions.
        - Applies a per-token reasoning weight (w_t = reasoning_weight if mask[t]
          else 1.0), so high-entropy "decision" tokens receive higher penalties.
        - Operates only on Top-K sparse teacher logits to match the cached format.

    Attributes (from class diagram):
        temperature     : Softmax temperature T for KD (default 4.0).
        alpha           : Blending weight between CE and KD loss (default 0.5).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reasoning_weight: float = 3.0,
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.reasoning_weight = reasoning_weight

    @classmethod
    def from_config(cls, config_path: str = "config/model_config.yaml") -> "KDLossEngine":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["distillation"]
        return cls(
            temperature=cfg["temperature"],
            alpha=cfg["alpha"],
            reasoning_weight=cfg["reasoning_weight"],
        )

    # ------------------------------------------------------------------ #

    def computeKLDivergence(
        self,
        student_logits: torch.Tensor,   # (B, seq_len, vocab_size)
        teacher_indices: torch.Tensor,  # (B, seq_len, K)  int64
        teacher_logprobs: torch.Tensor, # (B, seq_len, K)  float32, log-probs
    ) -> torch.Tensor:
        """
        KL(teacher || student) over the Top-K teacher vocabulary subset.

        We expand the sparse teacher distribution back into a (B, seq_len, K)
        slice of the student distribution, then compute KL divergence.

        Returns:
            kl : (B, seq_len) per-token KL divergence values.
        """
        T = self.temperature

        # Gather student log-probs at teacher Top-K positions
        student_logprobs_full = F.log_softmax(student_logits / T, dim=-1)   # (B, L, V)
        student_topk = student_logprobs_full.gather(
            dim=-1, index=teacher_indices
        )   # (B, L, K)

        # Normalise teacher Top-K log-probs (re-softmax to a valid distribution)
        teacher_topk_probs = F.softmax(teacher_logprobs / T, dim=-1)   # (B, L, K)

        # KL(p_teacher || p_student) = sum_k  p_T * (log p_T - log p_S)
        teacher_log_topk = F.log_softmax(teacher_logprobs / T, dim=-1)
        kl = (teacher_topk_probs * (teacher_log_topk - student_topk)).sum(dim=-1)  # (B, L)

        return kl * (T ** 2)   # scale correction from temperature

    def applyReasoningWeight(
        self,
        kl: torch.Tensor,              # (B, seq_len)
        reasoning_mask: torch.Tensor,  # (B, seq_len) bool
    ) -> torch.Tensor:
        """
        Up-weight KL loss at reasoning token positions.

        Returns:
            weighted_kl_loss : scalar mean loss.
        """
        weights = torch.where(
            reasoning_mask,
            torch.full_like(kl, self.reasoning_weight),
            torch.ones_like(kl),
        )
        return (kl * weights).mean()

    def compute(
        self,
        student_logits: torch.Tensor,   # (B, seq_len, vocab_size)
        labels: torch.Tensor,           # (B, seq_len)  int64, -100 for ignored
        teacher_indices: torch.Tensor,  # (B, seq_len, K)
        teacher_logprobs: torch.Tensor, # (B, seq_len, K)
        reasoning_mask: torch.Tensor,   # (B, seq_len) bool
    ) -> dict[str, torch.Tensor]:
        """
        Full combined loss.

        Returns dict with keys: "loss", "ce_loss", "kd_loss".
        """
        # Cross-entropy loss (standard LM objective)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Reasoning-weighted KD loss
        kl = self.computeKLDivergence(student_logits, teacher_indices, teacher_logprobs)
        kd_loss = self.applyReasoningWeight(kl, reasoning_mask)

        total = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

        return {"loss": total, "ce_loss": ce_loss.detach(), "kd_loss": kd_loss.detach()}
