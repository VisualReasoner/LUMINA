from __future__ import annotations

from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Mapping, Sequence

from lumina.models.base import ModelResponse
from lumina.schemas.states import ImageRef


class ReplayModelClient:
    """Deterministic model client for tests and saved-response replay."""

    def __init__(self, responses: Mapping[str, Sequence[dict]]):
        self.responses = {stage: deque(items) for stage, items in responses.items()}
        self.call_count = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayModelClient":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Replay file must map stage names to response lists.")
        return cls(payload)

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[ImageRef],
        schema: Mapping[str, object],
    ) -> ModelResponse:
        self.call_count += 1
        queue = self.responses.get(stage)
        if queue is None or not queue:
            raise RuntimeError(f"No replay response remains for stage {stage!r}.")
        payload = dict(queue.popleft())
        return ModelResponse(payload=payload, raw_text=json.dumps(payload), stage=stage)
