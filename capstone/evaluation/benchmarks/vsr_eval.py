from __future__ import annotations

from datasets import load_dataset

from evaluation.evaluation_engine import EvaluationEngine


class VSREval:
    """Loads VSR test split and runs evaluation via EvaluationEngine."""

    BENCHMARK_NAME = "vsr"

    def __init__(self, engine: EvaluationEngine, split: str = "test"):
        self.engine = engine
        self.split = split

    def run(self) -> dict:
        dataset = load_dataset("cambridgeltl/visual-spatial-reasoning", split=self.split)
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
            self.BENCHMARK_NAME, samples, extract_pred=self._extract
        )

    @staticmethod
    def _extract(raw: str) -> str:
        raw_lower = raw.strip().lower()
        if raw_lower.startswith("true"):
            return "true"
        if raw_lower.startswith("false"):
            return "false"
        return raw_lower
