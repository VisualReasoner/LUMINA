from __future__ import annotations

import os
from pathlib import Path

from lumina.models.base import ModelClient
from lumina.models.replay import ReplayModelClient


def build_model_client(
    *,
    backend: str,
    model: str | None = None,
    replay_json: str | Path | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    device_map: str = "auto",
    dtype: str = "auto",
    max_tokens: int = 1400,
    hf_cache_dir: str | None = None,
    local_files_only: bool = False,
    enable_thinking: bool | None = None,
) -> ModelClient:
    normalized = backend.strip().lower()
    if normalized == "replay":
        if replay_json is None:
            raise ValueError("replay backend requires replay_json.")
        return ReplayModelClient.from_json(replay_json)
    if normalized in {"openai", "openai_compatible", "api"}:
        if not model:
            raise ValueError("API backend requires a model name.")
        from lumina.models.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(
            model=model,
            api_key=os.environ.get(api_key_env),
            base_url=base_url,
            max_tokens=max_tokens,
        )
    if normalized in {"transformers", "hf", "local"}:
        if not model:
            raise ValueError("Transformers backend requires a model path or repository name.")
        from lumina.models.transformers_mllm import TransformersMLLMClient

        return TransformersMLLMClient(
            model_path=model,
            device_map=device_map,
            dtype=dtype,
            max_new_tokens=max_tokens,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
            enable_thinking=enable_thinking,
        )
    raise ValueError(f"Unsupported model backend: {backend!r}")
