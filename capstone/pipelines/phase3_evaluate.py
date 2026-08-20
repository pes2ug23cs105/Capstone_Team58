"""
Phase 3: Evaluation and Deployment Analysis
=============================================
Load the trained student model (with optional LoRA adapter), run all configured
benchmarks, measure latency/VRAM, and write a consolidated report.

Usage:
    python -m pipelines.phase3_evaluate
    python -m pipelines.phase3_evaluate --adapter-path outputs/checkpoints/final_adapter
                                         --eval-config  config/eval_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from evaluation.benchmarks import EgoSchemaEval, MathVistaEval, MMEEval, VSREval
from evaluation.demo_logger import DemoExampleLogger
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.evaluation_engine import EvaluationEngine
from student.lora_adapter import LoRAAdapter
from student.student_model import StudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_BENCHMARK_MAP = {
    "mathvista": MathVistaEval,
    "egoschema": EgoSchemaEval,
    "vsr": VSREval,
    "mme": MMEEval,
}


def run_phase3(
    model_config: str = "config/model_config.yaml",
    eval_config: str = "config/eval_config.yaml",
    adapter_path: str | None = None,
    benchmark_max_samples: int | None = None,
    run_system_metrics: bool = True,
) -> None:
    logger.info("=== Phase 3: Evaluation and Deployment Analysis ===")

    with open(eval_config) as f:
        ecfg = yaml.safe_load(f)

    # 1. Load student (+ optional LoRA adapter)
    logger.info("Loading student model...")
    student = StudentModel(config_path=model_config)

    if adapter_path and Path(adapter_path).exists():
        logger.info("Loading LoRA adapter from %s", adapter_path)
        lora = LoRAAdapter(config_path=model_config)
        student.model = lora.load_adapter(student.model, adapter_path)
    else:
        logger.info("No adapter path provided — evaluating base student model.")

    # Initialize demo example logger for qualitative analysis
    output_dir = Path(ecfg["evaluation"]["output_dir"])
    demo_logger = DemoExampleLogger(output_dir=output_dir)
    logger.info("Demo examples will be logged to %s", demo_logger.examples_file)

    # 2. Build evaluation engine (with demo logger)
    engine = EvaluationEngine(student, config_path=eval_config, demo_logger=demo_logger)
    all_results: dict = {}

    # 3. Run benchmarks
    benchmarks_to_run = ecfg["evaluation"]["benchmarks"]
    for bname in benchmarks_to_run:
        cls = _BENCHMARK_MAP.get(bname)
        if cls is None:
            logger.warning("Unknown benchmark '%s' — skipping.", bname)
            continue
        logger.info("Running benchmark: %s", bname)
        result = cls(engine).run(max_samples=benchmark_max_samples)
        all_results[bname] = result

    # 4. Latency + VRAM measurement (using first sample from MathVista as a proxy)
    if run_system_metrics:
        logger.info("Measuring latency and VRAM...")
        proxy_sample = {"question": "What is 2 + 2?", "image": None, "answer": "4"}
        latency = engine.measureLatency(proxy_sample)
        vram = engine.measureVRAM()
        all_results["latency"] = latency
        all_results["vram"] = vram

    # 5. Save consolidated report
    output_dir = Path(ecfg["evaluation"]["output_dir"])
    report_path = output_dir / "consolidated_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # 6. Finalize demo logger with summary and WOW cases
    demo_logger.finalize()

    logger.info("Phase 3 complete. Report saved to %s", report_path)
    _print_summary(all_results)


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for bname, res in results.items():
        if "accuracy" in res:
            print(f"  {bname:15s}  accuracy = {res['accuracy']*100:.1f}%")
    if "latency" in results:
        lat = results["latency"]
        status = "✓ MEETS" if lat["meets_target"] else "✗ MISSES"
        print(f"  {'latency':15s}  mean = {lat['mean_ms']:.1f} ms  [{status} target]")
    if "vram" in results:
        v = results["vram"]
        print(f"  {'VRAM':15s}  allocated = {v['vram_allocated_mb']:.0f} MB")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--benchmark-max-samples", type=int, default=None)
    parser.add_argument("--skip-system-metrics", action="store_true")
    args = parser.parse_args()
    run_phase3(
        args.model_config,
        args.eval_config,
        args.adapter_path,
        benchmark_max_samples=args.benchmark_max_samples,
        run_system_metrics=not args.skip_system_metrics,
    )
