from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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


def _validate_records(records: list[dict], disengagement_enabled: bool) -> None:
    seen_ids: set[str] = set()

    for idx, rec in enumerate(records):
        sample_id = str(rec.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Record {idx}: missing sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"Record {idx}: duplicate sample_id '{sample_id}'")
        seen_ids.add(sample_id)

        topk_logits = np.asarray(rec.get("topk_logits"), dtype=np.float32)
        topk_indices = np.asarray(rec.get("topk_indices"), dtype=np.int64)

        if topk_logits.ndim != 2:
            raise ValueError(
                f"Record {sample_id}: topk_logits must be rank-2 (seq_len, K), got {topk_logits.shape}"
            )
        if topk_indices.ndim != 2:
            raise ValueError(
                f"Record {sample_id}: topk_indices must be rank-2 (seq_len, K), got {topk_indices.shape}"
            )
        if topk_logits.shape != topk_indices.shape:
            raise ValueError(
                f"Record {sample_id}: topk shape mismatch logits={topk_logits.shape}, indices={topk_indices.shape}"
            )

        seq_len = int(topk_logits.shape[0])
        entropy_scores = rec.get("entropy_scores")
        reasoning_mask = rec.get("reasoning_mask")

        if not isinstance(entropy_scores, list) or len(entropy_scores) != seq_len:
            raise ValueError(
                f"Record {sample_id}: entropy_scores length {len(entropy_scores) if isinstance(entropy_scores, list) else 'N/A'} != seq_len {seq_len}"
            )

        if not isinstance(reasoning_mask, list) or len(reasoning_mask) != seq_len:
            raise ValueError(
                f"Record {sample_id}: reasoning_mask length {len(reasoning_mask) if isinstance(reasoning_mask, list) else 'N/A'} != seq_len {seq_len}"
            )

        if disengagement_enabled:
            if "t_star" not in rec:
                raise ValueError(f"Record {sample_id}: missing t_star while disengagement is enabled")
            t_star = rec["t_star"]
            if not isinstance(t_star, int):
                raise ValueError(f"Record {sample_id}: t_star must be int, got {type(t_star)}")
            if not (0 <= t_star < seq_len):
                raise ValueError(f"Record {sample_id}: t_star out of range ({t_star}) for seq_len={seq_len}")


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
    preflight_cfg = smoke_cfg["preflight"]
    phase1_cfg = smoke_cfg["phase1"]
    paths_cfg = smoke_cfg["paths"]

    os.chdir(capstone_root)
    sys.path.insert(0, str(capstone_root))

    from distillation.logit_cache import LogitCache
    from pipelines.phase1_teacher_distill import run_phase1

    train_config = Path(paths_cfg["train_config"])
    model_config = Path(paths_cfg["model_config"])

    if not train_config.exists():
        raise FileNotFoundError(f"train_config not found: {train_config}")
    if not model_config.exists():
        raise FileNotFoundError(f"model_config not found: {model_config}")

    sample_count = int(preflight_cfg.get("sample_count", 5))
    if sample_count < 3 or sample_count > 5:
        raise ValueError("preflight.sample_count must be between 3 and 5")

    _clear_cache_dir(train_config)

    run_phase1(
        model_config=str(model_config),
        train_config=str(train_config),
        max_samples=sample_count,
        max_new_tokens=int(phase1_cfg.get("max_new_tokens", 64)),
        disengagement_mode=str(phase1_cfg.get("disengagement_mode", "approx")),
    )

    cache = LogitCache.from_config(str(train_config))
    records = [json.loads(r.to_jsonl_line()) for r in cache.read()]

    if len(records) != sample_count:
        raise ValueError(f"Expected exactly {sample_count} records, got {len(records)}")

    _validate_records(records, disengagement_enabled=bool(phase1_cfg.get("enable_disengagement", True)))

    print(f"Preflight OK: validated {len(records)} records with strict shape/id checks.")


if __name__ == "__main__":
    main()
