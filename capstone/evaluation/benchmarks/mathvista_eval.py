from __future__ import annotations

import re

from datasets import load_dataset

from evaluation.evaluation_engine import EvaluationEngine


class MathVistaEval:
    """
    Loads the MathVista testmini split and runs evaluation via EvaluationEngine.
    """

    BENCHMARK_NAME = "mathvista"

    def __init__(self, engine: EvaluationEngine, split: str = "testmini"):
        self.engine = engine
        self.split = split

    def run(self) -> dict:
        dataset = load_dataset("AI4Math/MathVista", split=self.split)
        samples = [
            {
                "question": s["question"],
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
        """Extract the final numeric or short answer from model output."""
        # Look for patterns like "The answer is X" or "= X" or just the last number
        match = re.search(r"(?:answer is|=)\s*([\w\.\-]+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        numbers = re.findall(r"[-+]?\d*\.?\d+", raw)
        return numbers[-1] if numbers else raw.strip()
