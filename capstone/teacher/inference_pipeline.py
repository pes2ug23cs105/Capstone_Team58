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

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.teacher = TeacherModel(config_path)
        self.extractor = LogitExtractor(top_k=cfg["teacher"]["top_k_logits"])
        self.analyzer = ReasoningAnalyzer(
            entropy_threshold=cfg["distillation"]["entropy_threshold"]
        )

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
        for sample in tqdm(samples, total=total, desc="Teacher inference"):
            try:
                logits, trace = self.teacher.extractLogits(
                    question=sample["question"],
                    image=sample.get("image"),
                )
            except Exception as e:
                logger.warning(
                    "Teacher inference failed for sample %s: %s",
                    sample["id"], e,
                )
                continue

            topk_indices, topk_logprobs = self.extractor.extract(logits)
            entropy_scores = self.analyzer.computeEntropy(logits)
            reasoning_mask = self.analyzer.generateMask(entropy_scores)

            yield DistillationRecord(
                sample_id=sample["id"],
                source=sample["source"],
                question=sample["question"],
                answer=sample["answer"],
                rationale=trace,
                topk_indices=topk_indices,
                topk_logits=topk_logprobs,
                entropy_scores=entropy_scores.tolist(),
                reasoning_mask=reasoning_mask.tolist(),
            )
