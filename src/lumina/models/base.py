from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Protocol, Sequence

from lumina.schemas.states import ImageRef


@dataclass
class ModelResponse:
    payload: dict
    raw_text: str
    stage: str


class ModelClient(Protocol):
    call_count: int

    def generate_json(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[ImageRef],
        schema: Mapping[str, object],
    ) -> ModelResponse:
        ...


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    candidates: list[dict] = []
    cursor = 0
    while cursor < len(stripped):
        start = stripped.find("{", cursor)
        if start < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            cursor = start + 1
            continue
        try:
            value = json.loads(stripped[start:end])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            candidates.append(value)
        cursor = end
    if candidates:
        return candidates[-1]
    raise ValueError("Could not parse a complete JSON object from model response.")
