from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from teacher.teacher_model import TeacherModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_TRANSITION_WORDS = {
    "therefore",
    "since",
    "thus",
    "so",
    "because",
    "next",
    "first",
    "second",
    "finally",
    "additionally",
    "however",
}


def _token_text(token: Any) -> str:
    if isinstance(token, str):
        return token
    return str(token)


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _join_tokens(tokens: list[Any]) -> str:
    return " ".join(_token_text(token).strip() for token in tokens if _token_text(token).strip()).strip()


def _build_text_from_steps(question: str, steps: list[list[Any]], upto_step: int | None = None) -> str:
    chunks = [question.strip()]
    limit = len(steps) if upto_step is None else max(0, min(upto_step, len(steps)))
    for step in steps[:limit]:
        chunk = _join_tokens(step)
        if chunk:
            chunks.append(chunk)
    return "\n".join(chunk for chunk in chunks if chunk)


def _build_inputs(
    processor,
    image: Image.Image,
    text: str,
    device: str,
) -> dict[str, torch.Tensor]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    target_device = torch.device(device)
    return {key: value.to(target_device) for key, value in inputs.items()}


def _load_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _make_corrupted_image(real_image: Image.Image) -> Image.Image:
    width, height = real_image.size
    noise = np.random.normal(loc=128.0, scale=30.0, size=(height, width, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    corrupted = Image.fromarray(noise, mode="RGB")
    if real_image.mode != "RGB":
        corrupted = corrupted.convert(real_image.mode)
    return corrupted


def segment_trace_into_steps(token_list: list) -> list[list]:
    steps: list[list] = []
    current: list = []

    for token in token_list:
        current.append(token)
        token_text = _token_text(token).lower()
        is_boundary = (
            "\n" in token_text
            or "." in token_text
            or "?" in token_text
            or any(re.search(rf"\b{word}\b", token_text) for word in _TRANSITION_WORDS)
        )
        if is_boundary and len(current) >= 3:
            steps.append(current)
            current = []

    if current:
        if steps and len(current) < 3:
            steps[-1].extend(current)
        else:
            steps.append(current)

    merged: list[list] = []
    for step in steps:
        if merged and len(step) < 3:
            merged[-1].extend(step)
        else:
            merged.append(step)

    if not merged and token_list:
        merged.append(list(token_list))

    return merged


@torch.no_grad()
def compute_causal_necessity_scores(
    model,
    processor,
    image_path: str,
    question: str,
    steps: list[list],
    device: str,
) -> list[float]:
    model.eval()

    real_image = _load_image(image_path)
    corrupted_image = _make_corrupted_image(real_image)

    scores: list[float] = []
    for step_index, step in enumerate(steps):
        prefix_text = _build_text_from_steps(question, steps, upto_step=step_index)
        full_text = _build_text_from_steps(question, steps, upto_step=step_index + 1)

        clean_prefix_inputs = _build_inputs(processor, real_image, prefix_text, device)
        clean_full_inputs = _build_inputs(processor, real_image, full_text, device)
        corrupted_prefix_inputs = _build_inputs(processor, corrupted_image, prefix_text, device)
        corrupted_full_inputs = _build_inputs(processor, corrupted_image, full_text, device)

        clean_outputs = model(**clean_full_inputs)
        corrupted_outputs = model(**corrupted_full_inputs)

        clean_log_prob = _mean_log_prob_for_step(
            clean_outputs.logits,
            clean_full_inputs["input_ids"],
            clean_prefix_inputs["input_ids"].shape[1],
        )
        corrupted_log_prob = _mean_log_prob_for_step(
            corrupted_outputs.logits,
            corrupted_full_inputs["input_ids"],
            corrupted_prefix_inputs["input_ids"].shape[1],
        )
        scores.append(clean_log_prob - corrupted_log_prob)

    return scores


def _mean_log_prob_for_step(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prefix_len: int,
) -> float:
    if input_ids.size(1) <= prefix_len:
        return 0.0

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    step_token_ids = input_ids[:, prefix_len:]
    step_log_probs = log_probs[:, prefix_len - 1 : prefix_len - 1 + step_token_ids.size(1), :]
    gathered = step_log_probs.gather(dim=-1, index=step_token_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.mean().item()) if gathered.numel() > 0 else 0.0


def find_disengagement_point(scores: list[float], threshold: float = 0.1, window: int = 2) -> int:
    if not scores:
        return 0

    smoothed: list[float] = []
    for index, score in enumerate(scores):
        start = max(0, index - 1)
        smoothed.append(sum(scores[start : index + 1]) / (index - start + 1))

    for index, score in enumerate(smoothed):
        if score >= threshold:
            continue
        tail = smoothed[index + 1 : index + 1 + window]
        if len(tail) < window:
            continue
        if all(value < threshold for value in tail):
            return index

    return len(scores)


def _extract_reasoning_tokens(record: dict[str, Any]) -> list[Any]:
    candidate_keys = [
        "reasoning_trace_tokens",
        "trace_tokens",
        "reasoning_tokens",
        "rationale_tokens",
        "tokens",
    ]
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, list) and value:
            return value

    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        return rationale.split()
    return []


def _load_jsonl_records(jsonl_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed JSON at line %d in %s: %s", line_number, jsonl_path, error)
    return records


def run_phase2(
    input_jsonl_path: str,
    output_jsonl_path: str,
    model,
    processor,
    device: str,
    threshold: float = 0.1,
) -> None:
    logger.info("Starting disengagement analysis from %s", input_jsonl_path)

    input_path = Path(input_jsonl_path)
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as input_handle, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning("Skipping malformed JSON at line %d in %s: %s", line_number, input_jsonl_path, error)
                continue

            sample_id = str(record.get("sample_id", "")).strip()
            try:
                if "disengagement_point" in record:
                    output_handle.write(json.dumps(record) + "\n")
                    output_handle.flush()
                    continue

                token_list = _extract_reasoning_tokens(record)
                reasoning_steps = segment_trace_into_steps(token_list)
                question = str(record.get("question", ""))
                image_path = str(record.get("image_path", "")).strip()
                if not image_path:
                    image_path = str(record.get("image", "")).strip()

                if not image_path:
                    raise ValueError("Missing image path")
                if not reasoning_steps:
                    raise ValueError("No reasoning steps found")

                causal_necessity_scores = compute_causal_necessity_scores(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    question=question,
                    steps=reasoning_steps,
                    device=device,
                )
                disengagement_point = find_disengagement_point(
                    causal_necessity_scores,
                    threshold=threshold,
                )

                record["reasoning_steps"] = reasoning_steps
                record["causal_necessity_scores"] = causal_necessity_scores
                record["disengagement_point"] = disengagement_point

                output_handle.write(json.dumps(record) + "\n")
                output_handle.flush()
            except Exception as error:
                logger.exception("Failed to process sample %s: %s", sample_id or f"line-{line_number}", error)

    logger.info("Disengagement analysis complete. Output written to %s", output_path)


def _safe_pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _extract_model_answer(record: dict[str, Any]) -> str | None:
    for key in (
        "model_answer",
        "predicted_answer",
        "prediction",
        "response",
        "generated_answer",
        "answer_pred",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _normalise_text(value: Any) -> str:
    return str(value).strip().lower()


def print_disengagement_stats(jsonl_path: str) -> None:
    records = _load_jsonl_records(jsonl_path)
    if not records:
        print("No records found.")
        return

    all_t_star: list[float] = []
    per_source: dict[str, list[float]] = {}
    premature = 0
    never_disengaged = 0
    histogram = {bucket: 0 for bucket in range(11)}
    corr_t_star: list[float] = []
    corr_match: list[float] = []

    for record in records:
        t_star = float(record.get("disengagement_point", 0))
        steps = record.get("reasoning_steps", [])
        step_count = len(steps) if isinstance(steps, list) else 0
        source = str(record.get("source_dataset", record.get("source", "unknown")))

        all_t_star.append(t_star)
        per_source.setdefault(source, []).append(t_star)

        if t_star <= 1:
            premature += 1
        if step_count > 0 and int(t_star) == step_count:
            never_disengaged += 1

        bucket = min(int(t_star), 10)
        histogram[bucket] += 1

        model_answer = _extract_model_answer(record)
        ground_truth = record.get("answer")
        if model_answer is not None and ground_truth is not None:
            corr_t_star.append(t_star)
            corr_match.append(1.0 if _normalise_text(model_answer) == _normalise_text(ground_truth) else 0.0)

    avg_t_star = float(np.mean(all_t_star)) if all_t_star else 0.0
    print(f"Average T* across all samples: {avg_t_star:.4f}")

    print("Average T* per source_dataset:")
    for source, values in sorted(per_source.items()):
        print(f"  {source}: {float(np.mean(values)):.4f}")

    print(f"Premature disengagements (T* <= 1): {premature}")
    print(f"Never-disengaged samples (T* == number of steps): {never_disengaged}")

    print("ASCII histogram of T* values:")
    max_count = max(histogram.values()) if histogram else 0
    for bucket in range(10):
        count = histogram[bucket]
        bar = "#" * (40 * count // max_count) if max_count else ""
        print(f"  {bucket:>2}: {bar} ({count})")
    count = histogram[10]
    bar = "#" * (40 * count // max_count) if max_count else ""
    print(f"  10+: {bar} ({count})")

    correlation = _safe_pearson_correlation(corr_t_star, corr_match)
    if correlation is None:
        print("Pearson correlation between T* and answer match: N/A")
    else:
        print(f"Pearson correlation between T* and answer match: {correlation:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="outputs/logits/topk_logits_0000.jsonl")
    parser.add_argument("--output-jsonl", default="outputs/disengagement/topk_disengagement_0000.jsonl")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    teacher = TeacherModel(config_path=args.model_config)
    model_device = args.device or str(teacher.model.device)
    run_phase2(
        input_jsonl_path=args.input_jsonl,
        output_jsonl_path=args.output_jsonl,
        model=teacher.model,
        processor=teacher.processor,
        device=model_device,
        threshold=args.threshold,
    )
