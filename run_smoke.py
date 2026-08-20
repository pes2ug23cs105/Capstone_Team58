from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _clear_cache_dir(train_config_path: Path) -> None:
    cfg = _load_yaml(train_config_path)
    cache_cfg = cfg["logit_cache"]
    output_dir = Path(cache_cfg["output_dir"])
    file_prefix = cache_cfg["file_prefix"]

    if output_dir.exists():
        for path in output_dir.glob(f"{file_prefix}_*.jsonl"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", default="capstone/config/smoke.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    capstone_root = repo_root / "capstone"
    smoke_cfg_path = (repo_root / args.smoke_config).resolve()

    if not smoke_cfg_path.exists():
        raise FileNotFoundError(f"Smoke config not found: {smoke_cfg_path}")

    smoke_cfg = _load_yaml(smoke_cfg_path)
    phase1_cfg = smoke_cfg["phase1"]
    phase2_cfg = smoke_cfg["phase2"]
    phase3_cfg = smoke_cfg["phase3"]
    paths_cfg = smoke_cfg["paths"]

    os.chdir(capstone_root)
    sys.path.insert(0, str(capstone_root))

    from distillation.distillation_dataset import DistillationDataset
    from distillation.logit_cache import LogitCache
    from pipelines.phase1_teacher_distill import run_phase1
    from pipelines.phase2_student_train import run_phase2
    from pipelines.phase3_evaluate import run_phase3
    from student.student_model import StudentModel

    train_config = Path(paths_cfg["train_config"])
    model_config = Path(paths_cfg["model_config"])
    eval_config = Path(paths_cfg["eval_config"])

    if not train_config.exists() or not model_config.exists() or not eval_config.exists():
        raise FileNotFoundError("One or more smoke paths are invalid (train/model/eval config).")

    # Phase 1 smoke
    _clear_cache_dir(train_config)
    run_phase1(
        model_config=str(model_config),
        train_config=str(train_config),
        max_samples=int(phase1_cfg.get("dataset_size", 30)),
        max_new_tokens=int(phase1_cfg.get("max_new_tokens", 64)),
        disengagement_mode=str(phase1_cfg.get("disengagement_mode", "approx")),
    )

    # Cache -> Dataset smoke validation
    cache = LogitCache.from_config(str(train_config))
    num_records = cache.count()
    if num_records <= 0:
        raise RuntimeError("Smoke Phase 1 wrote zero records.")

    student = StudentModel(config_path=str(model_config))
    ds = DistillationDataset(cache=cache, processor=student.processor, max_seq_length=512)
    if len(ds) <= 0:
        raise RuntimeError("Smoke dataset build failed: zero-length DistillationDataset.")

    # Phase 2 smoke (very short)
    run_phase2(
        model_config=str(model_config),
        train_config=str(train_config),
        max_steps=int(phase2_cfg.get("max_steps", 3)),
    )

    # Phase 3 smoke (1 sample per benchmark, cheap metrics)
    run_phase3(
        model_config=str(model_config),
        eval_config=str(eval_config),
        adapter_path=str(Path(phase2_cfg.get("adapter_path", "outputs/checkpoints_smoke/final_adapter"))),
        benchmark_max_samples=int(phase3_cfg.get("max_samples_per_benchmark", 1)),
        run_system_metrics=not bool(phase3_cfg.get("disable_expensive_metrics", True)),
    )

    print("Smoke run OK: Phase 1/2/3 completed with short-run constraints.")


if __name__ == "__main__":
    main()
