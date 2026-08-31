from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from lumina.models.base import ModelResponse, extract_json_object
from lumina.schemas.states import ImageRef


def _resolve_snapshot(model_path: str) -> str:
    source = Path(model_path)
    snapshots = source / "snapshots"
    if not source.is_dir() or not snapshots.is_dir():
        return model_path
    main_ref = source / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.is_dir():
            return str(candidate)
    candidates = sorted(
        (item for item in snapshots.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else model_path


class TransformersMLLMClient:
    """Generic Transformers image-text client, including local Qwen-VL paths."""

    def __init__(
        self,
        *,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1400,
        trust_remote_code: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        enable_thinking: bool | None = None,
    ):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the local extra with `pip install -e .[local]`.") from exc
        torch_dtype = dtype
        if dtype != "auto":
            torch_dtype = getattr(torch, dtype)
        model_source = _resolve_snapshot(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            model_class = transformers.AutoModelForImageTextToText
        self.model = model_class.from_pretrained(
            model_source,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.call_count = 0

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[ImageRef],
        schema: Mapping[str, object],
    ) -> ModelResponse:
        pil_images = []
        content = []
        for image_ref in images:
            with Image.open(Path(image_ref.path)) as source:
                image = source.convert("RGB").copy()
            pil_images.append(image)
            content.append({"type": "text", "text": f"Image: {image_ref.label}"})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        fallback_kwargs = dict(template_kwargs)
        fallback_kwargs.pop("enable_thinking", None)
        user_only = [
            {
                "role": "user",
                "content": [
                    *content[:-1],
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ]
        attempts = [(messages, template_kwargs), (messages, fallback_kwargs), (user_only, fallback_kwargs)]
        inputs = None
        last_error = None
        for candidate_messages, candidate_kwargs in attempts:
            try:
                inputs = self.processor.apply_chat_template(candidate_messages, **candidate_kwargs)
                break
            except Exception as exc:
                last_error = exc
        if inputs is None:
            legacy_kwargs = {"tokenize": False, "add_generation_prompt": True}
            prompt = self.processor.apply_chat_template(user_only, **legacy_kwargs)
            inputs = self.processor(
                text=[prompt],
                images=pil_images or None,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        if not hasattr(inputs, "items"):
            raise RuntimeError(f"Processor returned unsupported inputs after chat templating: {last_error}")
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
            if key != "token_type_ids"
        }
        self.call_count += 1
        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        prompt_length = inputs["input_ids"].shape[1]
        raw = self.processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
        return ModelResponse(payload=extract_json_object(raw), raw_text=raw, stage=stage)
