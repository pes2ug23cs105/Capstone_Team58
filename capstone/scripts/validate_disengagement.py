from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _load_records(jsonl_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Skipping malformed JSON at line {line_number}: {_ascii(error)}")
    return records


def _format_step_line(step_index: int, score: float, step_text: str, t_star: int) -> str:
    marker = "->" if step_index == t_star else "  "
    return f"{marker} step {step_index:02d} score={score:.4f} | {_ascii(step_text)}"


def validate_disengagement(jsonl_path: str) -> None:
    records = _load_records(jsonl_path)
    if not records:
        print("No records found.")
        return

    sample_count = min(20, len(records))
    sampled_records = random.sample(records, sample_count)

    for record in sampled_records:
        sample_id = _ascii(record.get("sample_id", "unknown"))
        question = _ascii(record.get("question", ""))
        reasoning_steps = record.get("reasoning_steps", [])
        scores = record.get("causal_necessity_scores", [])
        t_star = int(record.get("disengagement_point", len(reasoning_steps)))

        print("=" * 80)
        print(f"sample_id: {sample_id}")
        print(f"question: {question}")
        print(f"T*: {t_star}")

        for step_index, step in enumerate(reasoning_steps):
            step_text = " ".join(_ascii(token) for token in step)
            score = float(scores[step_index]) if step_index < len(scores) else 0.0
            print(_format_step_line(step_index, score, step_text, t_star))

        if t_star >= len(reasoning_steps):
            print(f"-> T* points past the final step ({t_star}); never disengaged")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to the disengagement output JSONL")
    args = parser.parse_args()

    validate_disengagement(args.jsonl_path)


if __name__ == "__main__":
    main()
