from __future__ import annotations

from datasets import load_dataset

from evaluation.evaluation_engine import EvaluationEngine


class EgoSchemaEval:
    """Loads EgoSchema test split and runs evaluation via EvaluationEngine."""

    BENCHMARK_NAME = "egoschema"
    CHOICE_KEYS = ["option 0", "option 1", "option 2", "option 3", "option 4"]

    def __init__(self, engine: EvaluationEngine, split: str = "test"):
        self.engine = engine
        self.split = split

    def run(self) -> dict:
        dataset = load_dataset("lmms-lab/EgoSchema", split=self.split)
        samples = []
        for s in dataset:
            question = s.get("question", "")
            choices = [s.get(k, "") for k in self.CHOICE_KEYS]
            formatted = question + "\n" + "\n".join(
                f"({i}) {c}" for i, c in enumerate(choices) if c
            )
            answer_idx = s.get("answer", 0)
            answer = choices[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
            samples.append({"question": formatted, "answer": answer, "image": None})

        return self.engine.runBenchmark(self.BENCHMARK_NAME, samples)
