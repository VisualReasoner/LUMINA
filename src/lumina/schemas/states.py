from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping


def to_dict(value: object) -> object:
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value


def _strings(value: object, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _confidence(value: object, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _boolean(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return default


def _choice(value: object, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


@dataclass
class ImageRef:
    path: str
    label: str
    modality: str
    visit_id: str | None = None
    kind: str = "study"


@dataclass
class VisitInput:
    subject_id: str
    visit_id: str
    visit_date: str
    modalities: dict[str, list[ImageRef]] = field(default_factory=dict)
    context: dict[str, object] = field(default_factory=dict)
    references: dict[str, list[dict[str, object]]] = field(default_factory=dict)


@dataclass
class InspectionPlan:
    selected_modalities: list[str] = field(default_factory=list)
    anchor_visit_ids: dict[str, str] = field(default_factory=dict)
    visit_questions: dict[str, list[str]] = field(default_factory=dict)
    uncertainties_to_resolve: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InspectionPlan":
        questions = payload.get("visit_questions") if isinstance(payload.get("visit_questions"), dict) else {}
        anchors = payload.get("anchor_visit_ids") if isinstance(payload.get("anchor_visit_ids"), dict) else {}
        return cls(
            selected_modalities=_strings(payload.get("selected_modalities"), 16),
            anchor_visit_ids={str(k): str(v) for k, v in anchors.items() if str(v).strip()},
            visit_questions={str(k): _strings(v, 8) for k, v in questions.items()},
            uncertainties_to_resolve=_strings(payload.get("uncertainties_to_resolve"), 8),
            summary=str(payload.get("summary") or "").strip(),
        )


@dataclass
class ModalityObservation:
    modality: str
    present: bool
    findings: list[str] = field(default_factory=list)
    severity: int | None = None
    summary: str = ""
    confidence: float = 0.5
    uncertainty: str = ""

    @classmethod
    def from_dict(cls, modality: str, payload: Mapping[str, Any]) -> "ModalityObservation":
        severity = payload.get("severity")
        try:
            severity = max(0, min(3, int(severity))) if severity is not None else None
        except (TypeError, ValueError):
            severity = None
        return cls(
            modality=modality,
            present=_boolean(payload.get("present", True)),
            findings=_strings(payload.get("findings"), 8),
            severity=severity,
            summary=str(payload.get("summary") or "").strip(),
            confidence=_confidence(payload.get("confidence")),
            uncertainty=str(payload.get("uncertainty") or "").strip(),
        )


@dataclass
class CurrentVisitEvidence:
    visit_id: str
    modality_observations: dict[str, ModalityObservation] = field(default_factory=dict)
    summary: str = ""
    multimodal_consistency: str = "unknown"
    uncertainties: list[str] = field(default_factory=list)


@dataclass
class ReferenceCalibration:
    modality: str
    reference_id: str
    direction: str = "uncertain"
    magnitude: str = "uncertain"
    summary: str = ""
    confidence: float = 0.5
    uncertainty: str = ""

    @classmethod
    def from_dict(cls, modality: str, reference_id: str, payload: Mapping[str, Any]) -> "ReferenceCalibration":
        return cls(
            modality=modality,
            reference_id=reference_id,
            direction=_choice(
                payload.get("direction"),
                {"below", "similar", "subtle_above", "moderate_above", "marked_above", "uncertain"},
                "uncertain",
            ),
            magnitude=_choice(
                payload.get("magnitude"),
                {"none", "subtle", "moderate", "marked", "uncertain"},
                "uncertain",
            ),
            summary=str(payload.get("summary") or "").strip(),
            confidence=_confidence(payload.get("confidence")),
            uncertainty=str(payload.get("uncertainty") or "").strip(),
        )


@dataclass
class LocalComparison:
    modality: str
    current_evidence: dict[str, object] = field(default_factory=dict)
    anchor_available: bool = False
    anchor_visit_id: str | None = None
    anchor_date: str | None = None
    gap_days: int | None = None
    direction: str = "unavailable"
    magnitude: str = "unavailable"
    change_grade: int = 0
    localized_evidence: list[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    uncertainty: str = ""
    comparison_quality: str = "unavailable"

    @classmethod
    def unavailable(
        cls,
        modality: str,
        reason: str,
        current_evidence: Mapping[str, object] | None = None,
    ) -> "LocalComparison":
        return cls(modality=modality, current_evidence=dict(current_evidence or {}), uncertainty=reason)

    @classmethod
    def from_dict(
        cls,
        modality: str,
        anchor_visit_id: str,
        anchor_date: str,
        gap_days: int,
        payload: Mapping[str, Any],
        current_evidence: Mapping[str, object] | None = None,
    ) -> "LocalComparison":
        try:
            grade = max(0, min(3, int(payload.get("change_grade", 0))))
        except (TypeError, ValueError):
            grade = 0
        return cls(
            modality=modality,
            current_evidence=dict(current_evidence or {}),
            anchor_available=True,
            anchor_visit_id=anchor_visit_id,
            anchor_date=anchor_date,
            gap_days=max(0, int(gap_days)),
            direction=_choice(
                payload.get("direction"),
                {"decreased", "stable", "subtle_increase", "moderate_increase", "marked_increase", "mixed", "uncertain"},
                "uncertain",
            ),
            magnitude=_choice(
                payload.get("magnitude"),
                {"none", "subtle", "moderate", "marked", "uncertain"},
                "uncertain",
            ),
            change_grade=grade,
            localized_evidence=_strings(payload.get("localized_evidence"), 8),
            summary=str(payload.get("summary") or "").strip(),
            confidence=_confidence(payload.get("confidence")),
            uncertainty=str(payload.get("uncertainty") or "").strip(),
            comparison_quality=_choice(
                payload.get("comparison_quality"),
                {"strong", "adequate", "weak", "uninterpretable", "unknown"},
                "unknown",
            ),
        )


@dataclass
class ReconciledEvidence:
    agreements: list[str] = field(default_factory=list)
    complementary_evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    unresolved_uncertainties: list[str] = field(default_factory=list)
    dominant_evidence: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReconciledEvidence":
        return cls(
            agreements=_strings(payload.get("agreements"), 8),
            complementary_evidence=_strings(payload.get("complementary_evidence"), 8),
            conflicts=_strings(payload.get("conflicts"), 8),
            unresolved_uncertainties=_strings(payload.get("unresolved_uncertainties"), 8),
            dominant_evidence=_strings(payload.get("dominant_evidence"), 8),
            summary=str(payload.get("summary") or "").strip(),
        )


@dataclass
class DecisionContext:
    task_name: str
    label_space: list[str]
    rubric: list[str]
    belief_dimensions: dict[str, str] = field(default_factory=dict)
    allowed_context: dict[str, object] = field(default_factory=dict)
    evidence_requirements: list[str] = field(default_factory=list)
    evidence_snapshot: dict[str, object] = field(default_factory=dict)


@dataclass
class BeliefState:
    leading_label: str
    second_best_label: str
    decision_margin: str = "narrow"
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    longitudinal_evidence: list[str] = field(default_factory=list)
    open_uncertainties: list[str] = field(default_factory=list)
    task_support: dict[str, object] = field(default_factory=dict)
    summary: str = ""
    repaired: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], labels: tuple[str, ...]) -> "BeliefState":
        leading = str(payload.get("leading_label") or payload.get("diagnosis_code") or "").strip()
        second = str(payload.get("second_best_label") or payload.get("second_best_diagnosis_code") or "").strip()
        if leading not in labels:
            leading = labels[0]
        if second not in labels or second == leading:
            second = next(label for label in labels if label != leading)
        margin = _choice(payload.get("decision_margin"), {"clear", "moderate", "narrow"}, "narrow")
        return cls(
            leading_label=leading,
            second_best_label=second,
            decision_margin=margin,
            confidence=_confidence(payload.get("confidence")),
            supporting_evidence=_strings(payload.get("supporting_evidence") or payload.get("evidence_for_top_class"), 4),
            conflicting_evidence=_strings(payload.get("conflicting_evidence") or payload.get("evidence_against_top_class"), 4),
            longitudinal_evidence=_strings(payload.get("longitudinal_evidence"), 6),
            open_uncertainties=_strings(payload.get("open_uncertainties"), 6),
            task_support=dict(payload.get("task_support") or {})
            if isinstance(payload.get("task_support"), Mapping)
            else {},
            summary=str(payload.get("summary") or payload.get("belief_summary") or "").strip(),
        )


@dataclass
class AuditRecord:
    decisive_evidence: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    missing_critical_evidence: list[str] = field(default_factory=list)
    next_best_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    repair_performed: bool = False
    summary: str = ""


@dataclass
class ActionTraceEntry:
    index: int
    action: str
    reason_code: str
    status: str = "completed"
    modality: str | None = None
    visit_id: str | None = None
    detail: str = ""


@dataclass
class TrajectoryEvent:
    event_id: str
    subject_id: str
    visit_id: str
    visit_date: str
    summary: str
    change_grade: int
    confidence: float
    dominant_modalities: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    modality_changes: dict[str, dict[str, object]] = field(default_factory=dict)
    belief_shift: str = ""
    observation_only: bool = False


@dataclass
class TrajectoryState:
    events: list[TrajectoryEvent] = field(default_factory=list)
    active_event_ids: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    selected_mass: float = 0.0
    summary: str = "No trajectory history available."
    adapter_name: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryState":
        events = []
        event_fields = {
            "event_id",
            "subject_id",
            "visit_id",
            "visit_date",
            "summary",
            "change_grade",
            "confidence",
            "dominant_modalities",
            "conflicts",
            "uncertainties",
            "modality_changes",
            "belief_shift",
            "observation_only",
        }
        for raw in payload.get("events") or []:
            if isinstance(raw, Mapping):
                events.append(TrajectoryEvent(**{key: value for key, value in raw.items() if key in event_fields}))
        return cls(
            events=events,
            active_event_ids=[str(item) for item in payload.get("active_event_ids") or []],
            weights={str(key): float(value) for key, value in dict(payload.get("weights") or {}).items()},
            selected_mass=float(payload.get("selected_mass") or 0.0),
            summary=str(payload.get("summary") or "No trajectory history available."),
            adapter_name=str(payload.get("adapter_name") or ""),
        )


@dataclass
class CrossSubjectNeighbor:
    entry_id: str
    similarity: float
    summary: str
    modalities: list[str] = field(default_factory=list)
    audit_pattern: str = ""


@dataclass
class VisitResult:
    visit: VisitInput
    inspection: InspectionPlan
    current_evidence: CurrentVisitEvidence
    reference_calibrations: dict[str, ReferenceCalibration]
    visit_memory: dict[str, LocalComparison]
    trajectory_before: TrajectoryState
    cross_subject_context: list[CrossSubjectNeighbor]
    reconciled_evidence: ReconciledEvidence
    decision_context: DecisionContext
    belief: BeliefState
    audit: AuditRecord
    action_trace: list[ActionTraceEntry]
    model_call_count: int = 0


@dataclass
class SubjectResult:
    subject_id: str
    task_name: str
    prediction: str
    visits: list[VisitResult]
    trajectory: TrajectoryState
    call_count: int
    target_call_count: int = 0
    history_call_count: int = 0
