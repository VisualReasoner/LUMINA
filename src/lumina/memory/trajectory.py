from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log

from lumina.schemas.states import TrajectoryEvent, TrajectoryState


@dataclass(frozen=True)
class TrajectoryConfig:
    max_active_events: int = 4
    recency_half_life_days: float = 540.0
    change_grade_weight: float = 0.45
    confidence_weight: float = 0.35
    recency_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.max_active_events < 1:
            raise ValueError("max_active_events must be at least 1.")
        if self.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive.")
        weights = (self.change_grade_weight, self.confidence_weight, self.recency_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Trajectory utility weights must be nonnegative with a positive sum.")


def _date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _recency(event_date: str, latest_date: str, half_life_days: float) -> float:
    event = _date(event_date)
    latest = _date(latest_date)
    if event is None or latest is None:
        return 0.5
    if half_life_days <= 0:
        return 1.0
    delta_days = max(0, (latest - event).days)
    return exp(-log(2.0) * delta_days / half_life_days)


def event_utility(event: TrajectoryEvent, latest_date: str, config: TrajectoryConfig) -> float:
    grade = max(0.0, min(1.0, float(event.change_grade) / 3.0))
    confidence = max(0.0, min(1.0, float(event.confidence)))
    recency = _recency(event.visit_date, latest_date, config.recency_half_life_days)
    utility = (
        config.change_grade_weight * grade
        + config.confidence_weight * confidence
        + config.recency_weight * recency
    )
    return max(1e-9, utility)


def _trajectory_summary(active: list[TrajectoryEvent], total_count: int) -> str:
    if not active:
        return "No trajectory history available."
    substantial = sum(event.change_grade >= 2 and not event.observation_only for event in active)
    mild = sum(event.change_grade == 1 and not event.observation_only for event in active)
    conflicts = sum(bool(event.conflicts) for event in active)
    modalities = []
    for event in active:
        for modality in event.dominant_modalities:
            if modality not in modalities:
                modalities.append(modality)
    if substantial >= 2:
        pattern = "repeated substantial longitudinal change"
    elif substantial == 1 or mild >= 2:
        pattern = "limited or mixed longitudinal change"
    elif all(event.change_grade == 0 for event in active):
        pattern = "no supported longitudinal change"
    else:
        pattern = "subtle longitudinal change"
    modality_text = ", ".join(modalities[:3]) or "available modalities"
    conflict_text = f"; {conflicts} active conflict pattern(s)" if conflicts else ""
    return (
        f"Trajectory contains {total_count} event(s), with {pattern} centered on "
        f"{modality_text}{conflict_text}."
    )


def update_trajectory(
    previous: TrajectoryState,
    event: TrajectoryEvent,
    config: TrajectoryConfig | None = None,
    *,
    use_smc: bool = True,
) -> TrajectoryState:
    cfg = config or TrajectoryConfig()
    if any(item.event_id == event.event_id for item in previous.events):
        raise ValueError(f"Trajectory event_id already exists: {event.event_id}")
    events = [*previous.events, event]
    latest_date = max((item.visit_date for item in events), default=event.visit_date)
    utilities = [event_utility(item, latest_date, cfg) for item in events]
    total = sum(utilities)
    normalized = [value / total for value in utilities]
    if use_smc:
        ranked = sorted(range(len(events)), key=lambda index: (-normalized[index], events[index].visit_date, index))
        selected_indices = ranked[: max(1, min(cfg.max_active_events, len(events)))]
    else:
        selected_indices = list(range(max(0, len(events) - cfg.max_active_events), len(events)))
    active = [events[index] for index in selected_indices]
    active.sort(key=lambda item: (item.visit_date, item.visit_id))
    weights = {events[index].event_id: float(normalized[index]) for index in range(len(events))}
    active_ids = [item.event_id for item in active]
    return TrajectoryState(
        events=events,
        active_event_ids=active_ids,
        weights=weights,
        selected_mass=float(sum(weights[event_id] for event_id in active_ids)),
        summary=_trajectory_summary(active, len(events)),
        adapter_name=previous.adapter_name,
    )
