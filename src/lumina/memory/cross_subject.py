from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import sqrt
from pathlib import Path
import re
from typing import Iterable

from lumina.schemas.states import (
    ActionTraceEntry,
    CrossSubjectNeighbor,
    CurrentVisitEvidence,
    LocalComparison,
    ReconciledEvidence,
    TrajectoryState,
)


_ENTRY_FIELDS = {
    "entry_id",
    "subject_id",
    "signature",
    "summary",
    "modalities",
    "audit_pattern",
    "adapter_name",
}
_BASE_SIGNATURE_FIELDS = {
    "modality_count",
    "trajectory_event_count",
    "trajectory_active_count",
    "trajectory_selected_mass",
    "comparison_count",
    "action_count",
    "conflict_action_count",
    "repair_action_count",
}
_MODALITY_SIGNATURE_PREFIXES = ("has_", "severity_", "has_delta_", "change_grade_")


def _modality_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


@dataclass
class CrossSubjectEntry:
    entry_id: str
    subject_id: str
    signature: dict[str, float]
    summary: str
    modalities: list[str] = field(default_factory=list)
    audit_pattern: str = ""
    adapter_name: str = ""

    def validate_modalities(self, allowed_modalities: Iterable[str]) -> None:
        allowed = set(allowed_modalities)
        unsupported_modalities = sorted(set(self.modalities) - allowed)
        if unsupported_modalities:
            raise ValueError(
                f"Cross-subject entry contains modalities outside the active adapter: {unsupported_modalities}"
            )
        modality_features = {
            f"{prefix}{_modality_token(modality)}"
            for modality in self.modalities
            for prefix in _MODALITY_SIGNATURE_PREFIXES
        }
        unsupported_features = sorted(set(self.signature) - _BASE_SIGNATURE_FIELDS - modality_features)
        if unsupported_features:
            raise ValueError(
                f"Cross-subject signature is inconsistent with its modality list: {unsupported_features}"
            )

    @classmethod
    def from_dict(cls, payload: dict) -> "CrossSubjectEntry":
        unsupported_fields = sorted(set(payload) - _ENTRY_FIELDS)
        if unsupported_fields:
            raise ValueError(f"Cross-subject entry contains unsupported fields: {unsupported_fields}")
        signature = payload.get("signature") or {}
        if not isinstance(signature, dict):
            raise ValueError("Cross-subject entry signature must be an object.")
        unsupported_features = sorted(
            str(key)
            for key in signature
            if str(key) not in _BASE_SIGNATURE_FIELDS
            and not str(key).startswith(_MODALITY_SIGNATURE_PREFIXES)
        )
        if unsupported_features:
            raise ValueError(f"Cross-subject signature contains unsupported features: {unsupported_features}")
        return cls(
            entry_id=str(payload["entry_id"]),
            subject_id=str(payload["subject_id"]),
            signature={str(k): float(v) for k, v in signature.items()},
            summary=str(payload.get("summary") or ""),
            modalities=[str(item) for item in payload.get("modalities") or []],
            audit_pattern=str(payload.get("audit_pattern") or ""),
            adapter_name=str(payload.get("adapter_name") or ""),
        )


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    numerator = sum(float(left.get(key, 0.0)) * float(right.get(key, 0.0)) for key in keys)
    norm_left = sqrt(sum(float(left.get(key, 0.0)) ** 2 for key in keys))
    norm_right = sqrt(sum(float(right.get(key, 0.0)) ** 2 for key in keys))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return numerator / (norm_left * norm_right)


def build_signature(
    current: CurrentVisitEvidence,
    visit_memory: dict[str, LocalComparison],
    trajectory: TrajectoryState,
    action_trace: Iterable[ActionTraceEntry],
) -> dict[str, float]:
    signature: dict[str, float] = {
        "modality_count": min(1.0, len(current.modality_observations) / 6.0),
        "trajectory_event_count": min(1.0, len(trajectory.events) / 10.0),
        "trajectory_active_count": min(1.0, len(trajectory.active_event_ids) / 5.0),
        "trajectory_selected_mass": max(0.0, min(1.0, trajectory.selected_mass)),
        "comparison_count": min(1.0, sum(item.anchor_available for item in visit_memory.values()) / 6.0),
        "conflict_action_count": 0.0,
        "repair_action_count": 0.0,
    }
    for modality, observation in current.modality_observations.items():
        key = _modality_token(modality)
        signature[f"has_{key}"] = 1.0 if observation.present else 0.0
        signature[f"severity_{key}"] = 0.0 if observation.severity is None else observation.severity / 3.0
        comparison = visit_memory.get(modality)
        signature[f"has_delta_{key}"] = 1.0 if comparison and comparison.anchor_available else 0.0
        signature[f"change_grade_{key}"] = 0.0 if comparison is None else comparison.change_grade / 3.0
    trace = list(action_trace)
    if trace:
        signature["action_count"] = min(1.0, len(trace) / 12.0)
        signature["conflict_action_count"] = min(
            1.0, sum(item.reason_code == "unresolved_modality_conflict" for item in trace) / 3.0
        )
        signature["repair_action_count"] = min(1.0, sum(item.action == "repair_belief" for item in trace))
    return signature


def build_entry(
    *,
    entry_id: str,
    subject_id: str,
    current: CurrentVisitEvidence,
    visit_memory: dict[str, LocalComparison],
    trajectory: TrajectoryState,
    reconciled: ReconciledEvidence,
    action_trace: Iterable[ActionTraceEntry],
    adapter_name: str,
    forbidden_terms: Iterable[str] = (),
) -> CrossSubjectEntry:
    modalities = sorted(current.modality_observations)
    change_parts = [
        f"{modality}: {item.summary or item.direction}"
        for modality, item in visit_memory.items()
        if item.anchor_available
    ]
    summary_parts = [current.summary, *change_parts[:3], trajectory.summary, reconciled.summary]
    summary = " | ".join(part.strip() for part in summary_parts if part and part.strip())
    audit_pattern = "; ".join(reconciled.conflicts[:2] + reconciled.unresolved_uncertainties[:2])
    for term in sorted({str(item).strip() for item in forbidden_terms if str(item).strip()}, key=len, reverse=True):
        parts = [part for part in re.split(r"[\s_-]+", term) if part]
        pattern = re.compile(
            r"(?<!\w)" + r"[\s_-]+".join(re.escape(part) for part in parts) + r"(?!\w)",
            flags=re.IGNORECASE,
        )
        summary = pattern.sub("[redacted-label]", summary)
        audit_pattern = pattern.sub("[redacted-label]", audit_pattern)
    return CrossSubjectEntry(
        entry_id=entry_id,
        subject_id=subject_id,
        signature=build_signature(current, visit_memory, trajectory, action_trace),
        summary=summary,
        modalities=modalities,
        audit_pattern=audit_pattern,
        adapter_name=adapter_name,
    )


@dataclass
class CrossSubjectBank:
    entries: list[CrossSubjectEntry] = field(default_factory=list)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "CrossSubjectBank":
        entries = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(CrossSubjectEntry.from_dict(json.loads(line)))
        return cls(entries=entries)

    def save_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(entry), ensure_ascii=True, sort_keys=True) for entry in self.entries]
        destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def retrieve(
        self,
        *,
        subject_id: str,
        signature: dict[str, float],
        modalities: Iterable[str],
        top_k: int = 3,
        allowed_subject_ids: set[str] | None = None,
    ) -> list[CrossSubjectNeighbor]:
        query_modalities = set(modalities)
        scored: list[tuple[float, CrossSubjectEntry]] = []
        for entry in self.entries:
            if entry.subject_id == subject_id:
                continue
            if allowed_subject_ids is not None and entry.subject_id not in allowed_subject_ids:
                continue
            similarity = _cosine(signature, entry.signature)
            overlap = query_modalities & set(entry.modalities)
            if query_modalities and entry.modalities and not overlap:
                similarity *= 0.25
            elif overlap:
                similarity *= 0.8 + 0.2 * len(overlap) / max(1, len(query_modalities))
            if similarity > 0.0:
                scored.append((similarity, entry))
        scored.sort(key=lambda item: (-item[0], item[1].entry_id))
        return [
            CrossSubjectNeighbor(
                entry_id=entry.entry_id,
                similarity=float(similarity),
                summary=entry.summary,
                modalities=list(entry.modalities),
                audit_pattern=entry.audit_pattern,
            )
            for similarity, entry in scored[: max(0, top_k)]
        ]
