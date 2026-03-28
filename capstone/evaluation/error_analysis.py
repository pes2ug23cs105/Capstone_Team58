from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyses benchmark failures and feeds actionable signals back into
    the dataset refinement loop (step 10/11 in the architecture diagram).

    Produces:
        - Per-source failure rates (mathvista / egoschema / vsr).
        - Most common failure patterns (empty predictions, wrong format, etc.).
        - Threshold recommendation for entropy_threshold tuning.
    """

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        samples: list[dict],
        predictions: list[str],
        benchmark_name: str,
    ) -> dict:
        """
        Compare predictions against ground truth and categorise failures.

        Args:
            samples         : list of eval samples (with "answer", "source" keys).
            predictions     : corresponding model predictions.
            benchmark_name  : tag for output file naming.

        Returns:
            analysis dict with failure counts and recommendations.
        """
        assert len(samples) == len(predictions), "samples/predictions length mismatch"

        failures: list[dict] = []
        source_counter: Counter = Counter()
        failure_types: Counter = Counter()

        for sample, pred in zip(samples, predictions):
            gt = str(sample["answer"]).strip().lower()
            p = pred.strip().lower()

            if p != gt:
                ftype = self._classify_failure(pred, gt)
                failures.append({
                    "id": sample.get("id", ""),
                    "source": sample.get("source", benchmark_name),
                    "question": sample["question"][:80],
                    "expected": gt,
                    "got": p,
                    "failure_type": ftype,
                })
                source_counter[sample.get("source", benchmark_name)] += 1
                failure_types[ftype] += 1

        total = len(samples)
        analysis = {
            "benchmark": benchmark_name,
            "total": total,
            "num_failures": len(failures),
            "failure_rate": len(failures) / total if total > 0 else 0.0,
            "by_source": dict(source_counter),
            "by_type": dict(failure_types),
            "recommendations": self._make_recommendations(failure_types, source_counter),
        }

        # Write detailed failures list
        fail_path = self.output_dir / f"{benchmark_name}_failures.json"
        with open(fail_path, "w") as f:
            json.dump(failures[:200], f, indent=2)  # cap to 200 for readability

        summary_path = self.output_dir / f"{benchmark_name}_error_analysis.json"
        with open(summary_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(
            "Error analysis [%s]: %.1f%% failure rate. Top type: %s",
            benchmark_name,
            analysis["failure_rate"] * 100,
            failure_types.most_common(1)[0][0] if failure_types else "none",
        )
        return analysis

    @staticmethod
    def _classify_failure(pred: str, gt: str) -> str:
        if not pred.strip():
            return "empty_prediction"
        if len(pred) > 10 * len(gt):
            return "over_generation"
        if pred.strip().isdigit() and not gt.strip().isdigit():
            return "format_mismatch"
        return "wrong_answer"

    @staticmethod
    def _make_recommendations(
        failure_types: Counter, source_counter: Counter
    ) -> list[str]:
        recs = []
        if failure_types.get("empty_prediction", 0) > 5:
            recs.append("Increase max_new_tokens; many predictions are empty.")
        if failure_types.get("over_generation", 0) > 5:
            recs.append("Add stop tokens or lower max_new_tokens to reduce over-generation.")
        if failure_types.get("format_mismatch", 0) > 5:
            recs.append("Improve answer-extraction post-processing (extract_pred function).")
        dominant = source_counter.most_common(1)
        if dominant and dominant[0][1] > sum(source_counter.values()) * 0.6:
            recs.append(
                f"Dataset '{dominant[0][0]}' drives >60% of failures — "
                f"consider upsampling its reasoning-dense subset."
            )
        return recs
