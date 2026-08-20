from __future__ import annotations

import logging

from datasets import load_dataset

from evaluation.evaluation_engine import EvaluationEngine


logger = logging.getLogger(__name__)


class VSREval:
    """Loads VSR test split and runs evaluation via EvaluationEngine."""

    BENCHMARK_NAME = "vsr"
    DATASET_IDS = [
        "cambridgeltl/vsr_random",
        "cambridgeltl/vsr_zeroshot",
        "cambridgeltl/visual-spatial-reasoning",
    ]

    def __init__(self, engine: EvaluationEngine, split: str = "test"):
        self.engine = engine
        self.split = split

    def run(self, max_samples: int | None = None) -> dict:
        dataset = None
        last_error: Exception | None = None
        for dataset_id in self.DATASET_IDS:
            try:
                dataset = load_dataset(dataset_id, split=self.split)
                logger.info("VSR benchmark using dataset id: %s", dataset_id)
                break
            except Exception as e:
                last_error = e

        if dataset is None:
            raise RuntimeError(
                "Unable to load any configured VSR dataset ids: "
                f"{self.DATASET_IDS}. Last error: {last_error}"
            )

        samples = [
            {
                "question": (
                    f"Is the following statement true or false about the image?\n"
                    f"\"{s.get('caption', '')}\""
                ),
                "answer": "true" if s.get("label", 0) == 1 else "false",
                "image": s.get("image"),
            }
            for s in dataset
        ]
        return self.engine.runBenchmark(
            self.BENCHMARK_NAME,
            samples,
            extract_pred=self._extract,
            max_samples=max_samples,
        )

    @staticmethod
    def _extract(raw: str) -> str:
        raw_lower = raw.strip().lower()
        if raw_lower.startswith("true"):
            return "true"
        if raw_lower.startswith("false"):
            return "false"
        return raw_lower
