#!/usr/bin/env python3
"""
Data preprocessing status check and Phase 1 runner.

Usage:
    python preprocess.py            # Check status only
    python preprocess.py --run      # Check status, then preprocess missing data (requires GPU)
    python preprocess.py --run --model-config capstone/config/model_config.smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
CAPSTONE = ROOT / "capstone"
sys.path.insert(0, str(CAPSTONE))


# ─────────────────────────── helpers ──────────────────────────────────────── #

def _check_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return False, []
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "vram_gb": props.total_memory / 1024 ** 3,
            })
        return True, devices
    except Exception as e:
        return False, [{"error": str(e)}]


def _count_cached_by_source(cache_dir: Path, prefix: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(cache_dir.glob(f"{prefix}_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    src = record.get("source", "unknown")
                    counts[src] = counts.get(src, 0) + 1
                except Exception:
                    pass
    return counts


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────── status report ────────────────────────────────── #

def report_status(train_cfg: dict, cached_counts: dict[str, int]) -> list[str]:
    """Print a status table and return list of dataset names that still need preprocessing."""
    datasets_cfg = train_cfg.get("datasets", {})
    cache_cfg = train_cfg.get("logit_cache", {})
    cache_dir = CAPSTONE / cache_cfg.get("output_dir", "outputs/logits")
    total_cached = sum(cached_counts.values())

    print("\n" + "=" * 62)
    print("  PREPROCESSING STATUS")
    print("=" * 62)
    print(f"\n  Cache location : {cache_dir}")
    print(f"  Total cached   : {total_cached} records\n")

    missing: list[str] = []
    for name, cfg in datasets_cfg.items():
        cached = cached_counts.get(name, 0)
        max_s = cfg.get("max_samples")
        target = f"{max_s:,}" if max_s else "all"
        pct = f"{cached/max_s*100:.0f}%" if max_s and max_s > 0 else ("done" if cached > 0 else "?")
        if cached == 0:
            status = "MISSING"
            missing.append(name)
        elif max_s and cached < max_s:
            status = "PARTIAL"
            missing.append(name)
        else:
            status = "  OK  "
        print(f"  [{status}] {name:<12} {cached:>6,} / {target:>6} cached  {pct}")

    extra = {k: v for k, v in cached_counts.items() if k not in datasets_cfg}
    if extra:
        print("\n  Extra cached (not in current config):")
        for name, cnt in extra.items():
            print(f"         {name:<14} {cnt:>6,} records")

    print()
    return missing


# ─────────────────────────── main ─────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="Check data preprocessing status and optionally run Phase 1.")
    parser.add_argument("--run", action="store_true",
                        help="Run Phase 1 preprocessing for any missing/partial datasets (requires CUDA GPU).")
    parser.add_argument("--model-config", default="capstone/config/model_config.yaml",
                        help="Path to model config YAML (default: capstone/config/model_config.yaml).")
    parser.add_argument("--train-config", default="capstone/config/training_config.yaml",
                        help="Path to training config YAML (default: capstone/config/training_config.yaml).")
    parser.add_argument("--disengagement-mode", default="approx",
                        choices=["approx", "full"],
                        help="Disengagement detection mode for Phase 1 (default: approx).")
    args = parser.parse_args()

    model_cfg_path = ROOT / args.model_config
    train_cfg_path = ROOT / args.train_config

    for p in (model_cfg_path, train_cfg_path):
        if not p.exists():
            print(f"ERROR: Config not found: {p}", file=sys.stderr)
            sys.exit(1)

    train_cfg = _load_yaml(train_cfg_path)
    cache_cfg = train_cfg.get("logit_cache", {})
    cache_dir = CAPSTONE / cache_cfg.get("output_dir", "outputs/logits")
    prefix = cache_cfg.get("file_prefix", "topk_logits")

    cached_counts = _count_cached_by_source(cache_dir, prefix)
    missing = report_status(train_cfg, cached_counts)

    # ── GPU report ─────────────────────────────────────────────────────────── #
    has_gpu, devices = _check_gpu()
    print("GPU status:")
    if has_gpu:
        for d in devices:
            print(f"  GPU {d['id']}: {d['name']}  ({d['vram_gb']:.1f} GB VRAM)")
    else:
        print("  No CUDA GPU detected")
        if devices and "error" in devices[0]:
            print(f"  (torch error: {devices[0]['error']})")
    print()

    # ── Summary line ───────────────────────────────────────────────────────── #
    if not missing:
        print("All configured datasets are fully preprocessed.")
        if not args.run:
            return
        print("Nothing to preprocess — exiting.")
        return

    print(f"Preprocessing needed for: {', '.join(missing)}")

    if not args.run:
        print("\nTo start preprocessing:  python preprocess.py --run")
        print("(requires a CUDA GPU with ≥20 GB VRAM for the 32B teacher model)")
        return

    # ── Run Phase 1 ────────────────────────────────────────────────────────── #
    if not has_gpu:
        print(
            "\nERROR: No CUDA GPU available.\n"
            "       Phase 1 runs the Qwen2.5-VL-32B teacher model which requires\n"
            "       a CUDA GPU with ≥20 GB VRAM (4-bit quantised).\n"
            "       Run this script on a GPU machine and the cache will be reused here.",
            file=sys.stderr,
        )
        sys.exit(1)

    min_vram = min(d["vram_gb"] for d in devices)
    if min_vram < 18:
        print(
            f"\nWARNING: Detected GPU has only {min_vram:.1f} GB VRAM.\n"
            "         The 32B teacher (4-bit) typically needs ~20 GB.\n"
            "         Proceeding anyway — it may OOM.",
            file=sys.stderr,
        )

    print("\nStarting Phase 1: Offline Teacher Distillation …")
    print(f"  model config  : {model_cfg_path}")
    print(f"  train config  : {train_cfg_path}")
    print(f"  disengagement : {args.disengagement_mode}")
    print()

    # Phase 1 uses relative paths internally — run from within capstone/
    prev_dir = os.getcwd()
    os.chdir(CAPSTONE)
    try:
        from pipelines.phase1_teacher_distill import run_phase1
        run_phase1(
            model_config=str(model_cfg_path),
            train_config=str(train_cfg_path),
            disengagement_mode=args.disengagement_mode,
        )
    finally:
        os.chdir(prev_dir)

    # ── Final status ───────────────────────────────────────────────────────── #
    cached_after = _count_cached_by_source(cache_dir, prefix)
    total_after = sum(cached_after.values())
    print(f"\nPhase 1 complete.  Total cached records: {total_after:,}")
    report_status(train_cfg, cached_after)


if __name__ == "__main__":
    main()
