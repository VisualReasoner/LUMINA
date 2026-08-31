from __future__ import annotations

import base64
from pathlib import Path
from typing import Mapping, Sequence

from lumina.models.base import ModelResponse, extract_json_object
from lumina.schemas.states import ImageRef


def _data_url(path: str) -> str:
    source = Path(path)
    suffix = source.suffix.lower()
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "application/octet-stream"
    )
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1400,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the API extra with `pip install -e .[api]`.") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
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
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for image in images:
            content.append({"type": "text", "text": f"Image: {image.label}"})
            content.append({"type": "image_url", "image_url": {"url": _data_url(image.path)}})
        self.call_count += 1
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return ModelResponse(payload=extract_json_object(raw), raw_text=raw, stage=stage)
