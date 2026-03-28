from __future__ import annotations

import torch
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from typing import Any


class TeacherModel:
    """
    Wraps Qwen2.5-VL-32B-Instruct as the offline reasoning-expert teacher.

    Responsibilities:
        - generateReasoningTrace : produce chain-of-thought text for a sample
        - extractLogits          : return full vocabulary logits for each
                                   generated token (used by LogitExtractor)

    The teacher is always run in inference-only mode with 4-bit quantization
    to fit within ≤24 GB VRAM on a single consumer GPU.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["teacher"]

        self.model_id: str = cfg["model_id"]
        self.max_new_tokens: int = cfg["max_new_tokens"]
        self.top_k: int = cfg["top_k_logits"]

        quant_cfg = None
        if cfg.get("load_in_4bit", True):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant_cfg,
            device_map=cfg.get("device_map", "auto"),
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generateReasoningTrace(
        self, question: str, image: Image.Image | None = None
    ) -> str:
        """
        Run teacher inference and return the full generated text
        (reasoning trace + answer).
        """
        inputs = self._build_inputs(question, image)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # Decode only the newly generated tokens
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def extractLogits(
        self, question: str, image: Image.Image | None = None
    ) -> tuple[torch.Tensor, str]:
        """
        Run teacher inference and return:
            logits  : (seq_len, vocab_size) float tensor for generated tokens
            trace   : decoded reasoning trace string

        The logits tensor is on CPU to avoid holding GPU memory after return.
        """
        inputs = self._build_inputs(question, image)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,      # scores = per-step logits over vocab
        )
        # scores is a tuple of (vocab_size,) tensors, one per generated step
        logits = torch.stack(outputs.scores, dim=0).cpu()  # (seq_len, vocab_size)

        new_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        trace = self.processor.decode(new_ids, skip_special_tokens=True)

        return logits, trace

    # ------------------------------------------------------------------ #

    def _build_inputs(
        self, question: str, image: Image.Image | None
    ) -> dict[str, torch.Tensor]:
        """Build processor inputs from question text and optional image."""
        if image is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt],
            images=[image] if image else None,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}
