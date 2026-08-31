from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


ProgressionLabel = Literal["stable", "mild_progression", "clear_progression", "UNLABELED"]
Diagnosis3Way = Literal["CN", "MCI", "AD", "UNLABELED"]


@dataclass(frozen=True)
class ProgressionDefinition:
    task_name: str = "progression_forecasting"
    label_column: str = "progression_forecasting_3way"
    class_order: tuple[str, str, str] = ("stable", "mild_progression", "clear_progression")
    horizon_months: int = 36
    summary: str = (
        "Held-out future-outcome convention over a fixed 36-month post-target horizon, using the worst reached diagnosis bucket."
    )


DEFINITION = ProgressionDefinition()


def normalize_dx_3way(value: object) -> str:
    if not isinstance(value, str):
        return "UNLABELED"
    dx = value.strip().upper()
    if "MILD COGNITIVE IMPAIRMENT" in dx or "LMCI" in dx or "EMCI" in dx or "MCI" in dx:
        return "MCI"
    if "ALZHEIMER" in dx or "DEMENTIA" in dx or dx == "AD":
        return "AD"
    if dx in {"CN", "NL", "NORMAL", "CONTROL", "COGNITIVELY NORMAL"}:
        return "CN"
    return "UNLABELED"


def assign_progression_label(current_dx: object, future_dx_values: Iterable[object]) -> ProgressionLabel:
    current = normalize_dx_3way(current_dx)
    if current not in {"CN", "MCI"}:
        return "UNLABELED"

    normalized = [normalize_dx_3way(v) for v in future_dx_values]
    normalized = [dx for dx in normalized if dx in {"CN", "MCI", "AD"}]
    if not normalized:
        return "UNLABELED"
    if "AD" in normalized:
        return "clear_progression"
    if "MCI" in normalized:
        return "mild_progression"
    return "stable"
