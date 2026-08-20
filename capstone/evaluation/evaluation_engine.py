from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml

from student.student_model import StudentModel

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Runs benchmarks and measures latency / throughput / VRAM for the student model.

    Methods (from class diagram):
        runBenchmark()       : evaluate accuracy on a named benchmark.
        measureLatency()     : measure per-sample inference latency (ms).
        measureThroughput()  : measure samples/second throughput.
    """

    def __init__(
        self,
        student: StudentModel,
        config_path: str = "config/eval_config.yaml",
        demo_logger: Optional[object] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.student = student
        self.output_dir = Path(self.cfg["evaluation"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latency_target_ms: float = self.cfg["evaluation"]["latency_target_ms"]
        self.demo_logger = demo_logger

    # ------------------------------------------------------------------ #

    def runBenchmark(
        self,
        benchmark_name: str,
        samples: list[dict],
        extract_pred: Callable[[str], str] | None = None,
        max_samples: int | None = None,
    ) -> dict:
        """
        Evaluate the student on a list of samples from a named benchmark.

        Args:
            benchmark_name : e.g. "mathvista".
            samples        : list of dicts with keys "question", "answer", "image".
            extract_pred   : Optional post-processing function to normalise
                             raw prediction strings before comparison.

        Returns:
            results dict with "accuracy", "num_correct", "num_total".
        """
        self.student.model.eval()
        if max_samples is not None:
            samples = samples[:max_samples]
        correct = 0

        for idx, sample in enumerate(samples):
            image = self._sanitize_image(sample.get("image"))
            pred = self.student.predict(
                question=sample["question"],
                image=image,
            )
            if extract_pred:
                pred = extract_pred(pred)

            gt = str(sample["answer"]).strip().lower()
            pred_clean = self.extract_pred(pred)
            gt_clean = gt.strip().lower()

            is_correct = pred_clean == gt_clean or gt_clean in pred_clean
            if is_correct:
                correct += 1
            
            # Log first few examples if demo_logger is available
            if self.demo_logger and idx < 5:
                self.demo_logger.log_example(
                    question=sample.get("question", ""),
                    student_output=pred,
                    ground_truth=str(sample["answer"]),
                    teacher_trace=sample.get("teacher_rationale"),
                    is_correct=is_correct,
                    benchmark_name=benchmark_name,
                )

        total = len(samples)
        accuracy = correct / total if total > 0 else 0.0
        results = {
            "benchmark": benchmark_name,
            "accuracy": accuracy,
            "num_correct": correct,
            "num_total": total,
        }
        logger.info(
            "Benchmark %s: %.2f%% (%d/%d)",
            benchmark_name, accuracy * 100, correct, total,
        )
        self._save_results(benchmark_name, results)
        return results

    @staticmethod
    def _sanitize_image(image):
        """
        Normalize benchmark image inputs for model processor compatibility.

        Some HF datasets return relative paths like "images/1.jpg" that are
        not resolvable from the current working directory. In smoke/eval mode,
        treat unresolved strings as missing image input to avoid hard crashes.
        """
        if image is None:
            return None
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return image
            return image if Path(image).exists() else None
        if isinstance(image, dict):
            path = image.get("path")
            if isinstance(path, str) and Path(path).exists():
                return path
            return None
        return image

    def measureLatency(
        self,
        sample: dict,
        num_warmup: int | None = None,
        num_runs: int | None = None,
    ) -> dict:
        """
        Measure mean per-sample inference latency in milliseconds.

        Returns:
            dict with "mean_ms", "min_ms", "max_ms", "meets_target".
        """
        cfg = self.cfg["evaluation"]
        num_warmup = num_warmup or cfg["num_warmup_runs"]
        num_runs = num_runs or cfg["num_timed_runs"]

        self.student.model.eval()

        # Warmup
        for _ in range(num_warmup):
            self.student.predict(sample["question"], sample.get("image"))

        times: list[float] = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.student.predict(sample["question"], sample.get("image"))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = sum(times) / len(times)
        results = {
            "mean_ms": round(mean_ms, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "meets_target": mean_ms <= self.latency_target_ms,
        }
        logger.info(
            "Latency: mean=%.1f ms  min=%.1f ms  max=%.1f ms  target=%s",
            mean_ms, min(times), max(times),
            "MET" if results["meets_target"] else "MISSED",
        )
        return results

    def measureThroughput(
        self, samples: list[dict], batch_size: int = 1
    ) -> dict:
        """
        Measure samples-per-second throughput over a list of samples.

        Returns:
            dict with "samples_per_sec" and "total_time_sec".
        """
        self.student.model.eval()
        start = time.perf_counter()

        for sample in samples:
            self.student.predict(sample["question"], sample.get("image"))

        elapsed = time.perf_counter() - start
        sps = len(samples) / elapsed if elapsed > 0 else 0.0
        results = {
            "samples_per_sec": round(sps, 2),
            "total_time_sec": round(elapsed, 2),
            "num_samples": len(samples),
        }
        logger.info("Throughput: %.2f samples/s over %d samples", sps, len(samples))
        return results

    def measureVRAM(self) -> dict:
        """Return current GPU VRAM usage in MB."""
        if not torch.cuda.is_available():
            return {"vram_allocated_mb": 0, "vram_reserved_mb": 0}
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        results = {
            "vram_allocated_mb": round(allocated, 1),
            "vram_reserved_mb": round(reserved, 1),
        }
        logger.info("VRAM: allocated=%.1f MB  reserved=%.1f MB", allocated, reserved)
        return results

    # ------------------------------------------------------------------ #

    def _save_results(self, name: str, results: dict) -> None:
        path = self.output_dir / f"{name}_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def extract_pred(self,text: str) -> str:
        text = text.lower().strip()

        # remove common prefixes
        for prefix in ["answer:", "the answer is", "it is", "it's"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # remove punctuation
        text = text.strip(" .,:;!?\n")

        # keep last token (helps for numbers)
        tokens = text.split()
        if len(tokens) > 1:
            text = tokens[-1]

        return text