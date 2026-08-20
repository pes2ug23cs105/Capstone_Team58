from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm

from data.schemas import DistillationRecord
from reasoning.reasoning_analyzer import ReasoningAnalyzer
from teacher.logit_extractor import LogitExtractor
from teacher.teacher_model import TeacherModel

logger = logging.getLogger(__name__)


class TeacherInferencePipeline:
    """
    Orchestrates offline teacher inference over a dataset iterator.

    For each sample:
        1. Run teacher to get logits + reasoning trace.
        2. Extract Top-K sparse logits.
        3. Compute token entropy + reasoning mask.
        4. Yield a DistillationRecord ready for caching.

    This pipeline is intentionally stateless — call run() with any iterator.
    """

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        disengagement_mode: str = "approx",
        max_new_tokens_override: int | None = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.teacher = TeacherModel(config_path)
        self.extractor = LogitExtractor(top_k=cfg["teacher"]["top_k_logits"])
        self.analyzer = ReasoningAnalyzer(
            entropy_threshold=cfg["distillation"]["entropy_threshold"]
        )
        self.disengagement_mode = disengagement_mode
        self.max_new_tokens_override = max_new_tokens_override

    def run(
        self, samples: Iterator[dict], total: int | None = None
    ) -> Iterator[DistillationRecord]:
        """
        Args:
            samples : Iterator of normalized samples (from DatasetManager).
            total   : Optional length hint for the tqdm progress bar.

        Yields:
            DistillationRecord for each sample processed successfully.
        """
        seen_sample_ids: set[str] = set()
        for sample in tqdm(samples, total=total, desc="Teacher inference"):
            sample_id = str(sample.get("id", "")).strip()
            if not sample_id:
                raise ValueError("Missing sample id in phase1 sample stream.")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id detected during Phase 1: {sample_id}")
            seen_sample_ids.add(sample_id)

            for key in ("question", "answer", "source"):
                if key not in sample:
                    raise KeyError(f"Missing required sample field '{key}' for sample {sample_id}")

            try:
                logits, trace = self.teacher.extractLogits(
                    question=sample["question"],
                    image=sample.get("image"),
                    max_new_tokens=self.max_new_tokens_override,
                )
            except Exception as exc:
                import torch
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if is_oom:
                    torch.cuda.empty_cache()
                    logger.warning("OOM skipping sample %s (source=%s) — cache cleared.", sample_id, sample.get("source"))
                else:
                    logger.warning("Error skipping sample %s (source=%s): %s", sample_id, sample.get("source"), exc)
                continue

            topk_indices, topk_logprobs = self.extractor.extract(logits)
            topk_indices = self._normalize_topk_2d(topk_indices, "topk_indices")
            topk_logprobs = self._normalize_topk_2d(topk_logprobs, "topk_logits")
            entropy_scores = self.analyzer.computeEntropy(logits)
            reasoning_mask = self.analyzer.generateMask(entropy_scores)

            t_star: int | None = None
            visual_attention_mass: list[float] | None = None
            if self.disengagement_mode == "approx":
                visual_mass_tensor = self.analyzer.estimateVisualAttentionMass(
                    logits,
                    has_image=sample.get("image") is not None,
                )
                t_star = self.analyzer.estimateDisengagementPoint(
                    entropy=entropy_scores,
                    visual_attention_mass=visual_mass_tensor,
                )
                visual_attention_mass = visual_mass_tensor.tolist()

            seq_len = int(entropy_scores.numel())
            if seq_len == 0:
                raise ValueError(f"Empty sequence produced for sample {sample_id}")
            if len(topk_indices) != seq_len or len(topk_logprobs) != seq_len:
                raise ValueError(
                    "Top-K shape mismatch for sample "
                    f"{sample_id}: indices={len(topk_indices)} logits={len(topk_logprobs)} seq_len={seq_len}"
                )

            avg_entropy = float(entropy_scores.mean().item())
            if t_star is not None:
                before_pct = (100.0 * t_star / seq_len)
                after_pct = 100.0 - before_pct
                logger.info(
                    "source=%s sample=%s seq_len=%d avg_entropy=%.4f t_star=%d before=%.1f%% after=%.1f%%",
                    sample["source"],
                    sample_id,
                    seq_len,
                    avg_entropy,
                    t_star,
                    before_pct,
                    after_pct,
                )
            else:
                logger.info(
                    "source=%s sample=%s seq_len=%d avg_entropy=%.4f",
                    sample["source"],
                    sample_id,
                    seq_len,
                    avg_entropy,
                )

            yield DistillationRecord(
                sample_id=sample_id,
                source=sample["source"],
                question=sample["question"],
                answer=sample["answer"],
                rationale=trace,
                topk_indices=topk_indices,
                topk_logits=topk_logprobs,
                entropy_scores=entropy_scores.tolist(),
                reasoning_mask=reasoning_mask.tolist(),
                t_star=t_star,
                visual_attention_mass=visual_attention_mass,
            )

    @staticmethod
    def _normalize_topk_2d(data: list[list], field_name: str) -> list[list]:
        if not data:
            return data
        first = data[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            # Accept legacy shape (seq_len, 1, K) and collapse singleton axis.
            collapsed: list[list] = []
            for row in data:
                if len(row) != 1:
                    raise ValueError(
                        f"{field_name} expected singleton middle dim in legacy format, got row len {len(row)}"
                    )
                collapsed.append(row[0])
            data = collapsed

        for row in data:
            if not isinstance(row, list):
                raise ValueError(f"{field_name} must be 2D list, got row type {type(row)}")
            if row and isinstance(row[0], list):
                raise ValueError(f"{field_name} must be 2D after normalization")
        return data
