from __future__ import annotations

from datasets import load_dataset

from evaluation.evaluation_engine import EvaluationEngine


class MMEEval:
    """Loads MME test split and runs evaluation via EvaluationEngine."""

    BENCHMARK_NAME = "mme"

    def __init__(self, engine: EvaluationEngine, split: str = "test"):
        self.engine = engine
        self.split = split

    def run(self) -> dict:
        dataset = load_dataset("lmms-lab/MME", split=self.split)
        samples = [
            {
                "question": s.get("question", ""),
                "answer": str(s.get("answer", "")),
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
        if "yes" in raw_lower:
            return "yes"
        if "no" in raw_lower:
            return "no"
        return raw_lower
