from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean
from typing import Mapping, Sequence

from lumina.adapters.specs import AdapterSpec
from lumina.memory.cross_subject import CrossSubjectBank, build_signature
from lumina.models.base import ModelClient
from lumina.prompts.builders import PromptBuilder, PromptRequest
from lumina.schemas.states import (
    ActionTraceEntry,
    AuditRecord,
    BeliefState,
    CurrentVisitEvidence,
    DecisionContext,
    ImageRef,
    InspectionPlan,
    LocalComparison,
    ModalityObservation,
    ReconciledEvidence,
    ReferenceCalibration,
    TrajectoryEvent,
    TrajectoryState,
    VisitInput,
    VisitResult,
    to_dict,
)


@dataclass(frozen=True)
class ControllerConfig:
    use_references: bool = True
    use_anchor_comparisons: bool = True
    use_cross_subject_memory: bool = True
    cross_subject_top_k: int = 3
    retrieve_when_ambiguous: bool = True
    repair_limit: int = 1
    use_audit: bool = True
    schema_retry_limit: int = 1


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _list(value: object, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


class EvidenceController:
    def __init__(
        self,
        *,
        adapter: AdapterSpec,
        model: ModelClient,
        config: ControllerConfig | None = None,
        cross_subject_bank: CrossSubjectBank | None = None,
    ):
        self.adapter = adapter
        self.model = model
        self.config = config or ControllerConfig()
        self.cross_subject_bank = cross_subject_bank
        if cross_subject_bank is not None:
            missing_adapter = [entry.entry_id for entry in cross_subject_bank.entries if not entry.adapter_name]
            if missing_adapter:
                raise ValueError(
                    "Cross-subject bank entries must identify their adapter; "
                    f"affected entries include {missing_adapter[:5]}."
                )
            for entry in cross_subject_bank.entries:
                entry.validate_modalities(adapter.modality_map)
            mismatched = sorted(
                {
                    entry.adapter_name
                    for entry in cross_subject_bank.entries
                    if entry.adapter_name and entry.adapter_name != adapter.adapter_name
                }
            )
            if mismatched:
                raise ValueError(
                    f"Cross-subject bank adapter mismatch: expected {adapter.adapter_name!r}, found {mismatched}."
                )
            label_terms = set(adapter.task.labels) | set(adapter.task.label_aliases)
            patterns = [
                re.compile(
                    r"(?<!\w)" + r"[\s_-]+".join(re.escape(part) for part in term.lower().split("_")) + r"(?!\w)",
                    flags=re.IGNORECASE,
                )
                for term in label_terms
                if term
            ]
            leaking_entries = [
                entry.entry_id
                for entry in cross_subject_bank.entries
                if any(pattern.search(f"{entry.summary} {entry.audit_pattern}") for pattern in patterns)
            ]
            if leaking_entries:
                raise ValueError(
                    "Cross-subject bank text contains active-task label terms; "
                    f"affected entries include {leaking_entries[:5]}."
                )
        self.prompts = PromptBuilder(adapter)

    def _validate_stage_payload(self, stage: str, payload: object, schema: Mapping[str, object]) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("output must be one JSON object")
        required_types: dict[str, dict[str, type | tuple[type, ...]]] = {
            "inspect": {"selected_modalities": list},
            "observe": {"modality_observations": dict},
            "calibrate_reference": {"direction": str, "magnitude": str},
            "compare_anchor": {"direction": str, "magnitude": str, "change_grade": (int, float)},
            "reconcile": {"summary": str},
            "final_belief": {
                "leading_label": str,
                "second_best_label": str,
                "supporting_evidence": list,
            },
            "repair_belief": {
                "leading_label": str,
                "second_best_label": str,
                "supporting_evidence": list,
            },
            "audit": {"summary": str},
        }
        for key, expected in required_types.get(stage, {}).items():
            if key not in payload:
                raise ValueError(f"missing required field {key!r}")
            if not isinstance(payload[key], expected):
                raise ValueError(f"field {key!r} has the wrong type")
        if stage == "observe":
            expected_raw = schema.get("modality_observations")
            actual_raw = payload.get("modality_observations")
            expected_modalities = set(expected_raw) if isinstance(expected_raw, Mapping) else set()
            actual_modalities = set(actual_raw) if isinstance(actual_raw, Mapping) else set()
            normalize = lambda value: str(value).lower().replace("_", " ").strip()
            missing = sorted(
                modality
                for modality in expected_modalities
                if normalize(modality) not in {normalize(item) for item in actual_modalities}
            )
            if missing:
                raise ValueError(f"modality_observations is missing selected modalities: {missing}")
        if stage in {"final_belief", "repair_belief"}:
            leading = self.adapter.task.normalize_label(payload["leading_label"])
            second = self.adapter.task.normalize_label(payload["second_best_label"])
            if leading == second:
                raise ValueError("leading and second-best labels must differ")
            support = payload.get("task_support")
            expected_dimensions = set(self.adapter.task.belief_dimensions)
            if expected_dimensions:
                if not isinstance(support, dict):
                    raise ValueError("missing task_support object")
                missing = sorted(expected_dimensions - set(support))
                if missing:
                    raise ValueError(f"task_support is missing fields: {missing}")
            payload = dict(payload)
            payload["leading_label"] = leading
            payload["second_best_label"] = second
        return payload

    def _generate(self, request: PromptRequest) -> dict:
        retry_limit = max(0, int(self.config.schema_retry_limit))
        user_prompt = request.user_prompt
        last_error: Exception | None = None
        for attempt in range(retry_limit + 1):
            try:
                response = self.model.generate_json(
                    stage=request.stage,
                    system_prompt=request.system_prompt,
                    user_prompt=user_prompt,
                    images=request.images,
                    schema=request.schema,
                )
                return self._validate_stage_payload(request.stage, response.payload, request.schema)
            except (ValueError, TypeError, KeyError) as exc:
                last_error = exc
                if attempt >= retry_limit:
                    break
                user_prompt = (
                    f"{request.user_prompt}\n"
                    f"The previous output was invalid: {exc}. Return exactly one corrected JSON object "
                    f"matching this schema: {json.dumps(request.schema, ensure_ascii=True, sort_keys=True)}"
                )
        raise ValueError(
            f"Stage {request.stage!r} failed schema validation after {retry_limit + 1} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _valid_anchors(history: Sequence[VisitInput], current: VisitInput) -> dict[str, VisitInput]:
        anchors: dict[str, VisitInput] = {}
        for modality in current.modalities:
            for visit in reversed(history):
                if modality in visit.modalities:
                    anchors[modality] = visit
                    break
        return anchors

    @staticmethod
    def _history_inventory(history: Sequence[VisitInput]) -> list[dict]:
        return [
            {
                "visit_id": f"V{index}",
                "visit_date": visit.visit_date,
                "modalities": sorted(visit.modalities),
            }
            for index, visit in enumerate(history, start=1)
        ]

    @staticmethod
    def _visit_handle(history: Sequence[VisitInput], visit: VisitInput) -> str:
        for index, item in enumerate(history, start=1):
            if item.visit_id == visit.visit_id:
                return f"V{index}"
        return f"V{len(history) + 1}"

    @staticmethod
    def _model_visit_memory(memory: Mapping[str, LocalComparison]) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for modality, comparison in memory.items():
            values = dict(to_dict(comparison))
            values.pop("anchor_visit_id", None)
            payload[modality] = values
        return payload

    @staticmethod
    def _model_calibrations(
        calibrations: Mapping[str, ReferenceCalibration],
    ) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for modality, calibration in calibrations.items():
            values = dict(to_dict(calibration))
            values.pop("reference_id", None)
            payload[modality] = values
        return payload

    @staticmethod
    def _model_trajectory(trajectory: TrajectoryState) -> dict[str, object]:
        active_ids = set(trajectory.active_event_ids)
        active_events = []
        for event in trajectory.events:
            if event.event_id not in active_ids:
                continue
            values = dict(to_dict(event))
            for key in ("event_id", "subject_id", "visit_id"):
                values.pop(key, None)
            active_events.append(values)
        return {"summary": trajectory.summary, "active_events": active_events}

    def _inspection(
        self,
        history: Sequence[VisitInput],
        current: VisitInput,
        valid_anchors: Mapping[str, VisitInput],
    ) -> InspectionPlan:
        anchor_payload = {
            modality: {"visit_id": self._visit_handle(history, visit), "visit_date": visit.visit_date}
            for modality, visit in valid_anchors.items()
        }
        request = self.prompts.inspect(
            visit_id=self._visit_handle(history, current),
            available_modalities=sorted(current.modalities),
            history_inventory=self._history_inventory(history),
            valid_anchors=anchor_payload,
            context=current.context,
        )
        plan = InspectionPlan.from_dict(self._generate(request))
        available = set(current.modalities)
        plan.selected_modalities = [item for item in plan.selected_modalities if item in available]
        if not plan.selected_modalities:
            plan.selected_modalities = sorted(available)
        plan.anchor_visit_ids = {
            modality: self._visit_handle(history, valid_anchors[modality])
            for modality in plan.selected_modalities
            if modality in valid_anchors
        }
        for modality in plan.selected_modalities:
            if not plan.visit_questions.get(modality):
                plan.visit_questions[modality] = list(self.adapter.modality_map[modality].observation_questions)
        return plan

    def _observe(self, current: VisitInput, plan: InspectionPlan, visit_handle: str) -> CurrentVisitEvidence:
        images = {
            modality: self._model_images(
                current.modalities[modality],
                modality=modality,
                role="current",
                visit_handle=visit_handle,
            )
            for modality in plan.selected_modalities
        }
        request = self.prompts.observe(
            visit_id=visit_handle,
            modality_images=images,
            visit_questions=plan.visit_questions,
        )
        payload = self._generate(request)
        raw_observations = payload.get("modality_observations")
        if not isinstance(raw_observations, dict):
            raw_observations = {}
        observations: dict[str, ModalityObservation] = {}
        for modality in plan.selected_modalities:
            raw = raw_observations.get(modality)
            if not isinstance(raw, dict):
                match = next(
                    (
                        value
                        for key, value in raw_observations.items()
                        if str(key).lower().replace("_", " ") == modality.lower().replace("_", " ")
                        and isinstance(value, dict)
                    ),
                    {},
                )
                raw = match
            observations[modality] = ModalityObservation.from_dict(modality, raw)
        consistency = str(payload.get("multimodal_consistency") or "unknown").strip().lower()
        if consistency not in {"convergent", "complementary", "conflicting", "unknown"}:
            consistency = "unknown"
        return CurrentVisitEvidence(
            visit_id=visit_handle,
            modality_observations=observations,
            summary=str(payload.get("summary") or "").strip(),
            multimodal_consistency=consistency,
            uncertainties=_list(payload.get("uncertainties")),
        )

    @staticmethod
    def _candidate_paths(candidate: Mapping[str, object]) -> list[str]:
        raw = candidate.get("image_paths", candidate.get("image_path"))
        if isinstance(raw, list):
            paths = [str(item) for item in raw]
        else:
            paths = [item.strip() for item in str(raw or "").split("|") if item.strip()]
        return [path for path in paths if Path(path).is_file()]

    @staticmethod
    def _view_name(path: str, index: int) -> str:
        name = Path(path).name.lower()
        for view in ("axial", "coronal", "sagittal"):
            if view in name:
                return view
        return f"view {index + 1}"

    @classmethod
    def _model_images(
        cls,
        images: Sequence[ImageRef],
        *,
        modality: str,
        role: str,
        visit_handle: str,
    ) -> list[ImageRef]:
        return [
            ImageRef(
                path=image.path,
                label=f"{role} {modality} {cls._view_name(image.path, index)} at {visit_handle}",
                modality=modality,
                visit_id=visit_handle,
                kind=image.kind,
            )
            for index, image in enumerate(images)
        ]

    @staticmethod
    def _match_field_score(candidate: Mapping[str, object], context: Mapping[str, object], field: str) -> tuple:
        candidate_value = candidate.get(field)
        target_value = context.get(field)
        if candidate_value is None or target_value is None:
            return 2, float("inf")
        try:
            return 0, abs(float(candidate_value) - float(target_value))
        except (TypeError, ValueError):
            candidate_text = str(candidate_value).strip().lower()
            target_text = str(target_value).strip().lower()
            return (0, 0.0) if candidate_text and candidate_text == target_text else (1, 0.0)

    def _reference_score(self, candidate: Mapping[str, object], context: Mapping[str, object]) -> tuple:
        policy = self.adapter.reference_policy
        return (
            *self._match_field_score(candidate, context, policy.match_primary),
            *self._match_field_score(candidate, context, policy.match_secondary),
            str(candidate.get("candidate_id") or ""),
        )

    def _select_reference(self, current: VisitInput, modality: str) -> tuple[str, list[ImageRef]] | None:
        policy = self.adapter.reference_policy
        matching_fields = (policy.match_primary, policy.match_secondary)
        missing_context = [
            field
            for field in matching_fields
            if field not in current.context or current.context[field] is None or str(current.context[field]).strip() == ""
        ]
        if current.references.get(modality) and missing_context:
            raise ValueError(f"Reference matching context is missing fields: {missing_context}")
        candidates = []
        for candidate in current.references.get(modality, []):
            if str(candidate.get("subject_id") or "") == current.subject_id:
                continue
            missing_candidate_fields = [
                field
                for field in matching_fields
                if field not in candidate or candidate[field] is None or str(candidate[field]).strip() == ""
            ]
            if missing_candidate_fields:
                raise ValueError(
                    f"Reference candidate is missing matching fields: {missing_candidate_fields}"
                )
            paths = self._candidate_paths(candidate)
            if paths:
                candidates.append((candidate, paths))
        if not candidates:
            return None
        candidate, paths = min(candidates, key=lambda item: self._reference_score(item[0], current.context))
        reference_id = str(candidate.get("candidate_id") or "reference")
        refs = [
            ImageRef(
                path=path,
                label=f"matched reference {modality} {self._view_name(path, index)}",
                modality=modality,
                kind="reference",
            )
            for index, path in enumerate(paths)
        ]
        return reference_id, refs

    def _calibrate(
        self,
        current: VisitInput,
        plan: InspectionPlan,
        evidence: CurrentVisitEvidence,
        trace: list[ActionTraceEntry],
    ) -> dict[str, ReferenceCalibration]:
        results: dict[str, ReferenceCalibration] = {}
        if not self.config.use_references or not self.adapter.reference_policy.enabled:
            return results
        for modality in plan.selected_modalities:
            selected = self._select_reference(current, modality)
            if selected is None:
                continue
            reference_id, reference_images = selected
            request = self.prompts.calibrate_reference(
                modality=self.adapter.modality_map[modality],
                current_images=self._model_images(
                    current.modalities[modality],
                    modality=modality,
                    role="current",
                    visit_handle=evidence.visit_id,
                ),
                reference_images=reference_images,
                current_observation=evidence.modality_observations[modality],
                reference_id="matched_reference",
            )
            payload = self._generate(request)
            results[modality] = ReferenceCalibration.from_dict(modality, reference_id, payload)
            trace.append(
                ActionTraceEntry(
                    index=len(trace) + 1,
                    action="calibrate_reference",
                    reason_code="needs_reference_comparison",
                    modality=modality,
                    visit_id=current.visit_id,
                    detail=reference_id,
                )
            )
        return results

    def _compare(
        self,
        current: VisitInput,
        plan: InspectionPlan,
        valid_anchors: Mapping[str, VisitInput],
        evidence: CurrentVisitEvidence,
        calibrations: Mapping[str, ReferenceCalibration],
        trace: list[ActionTraceEntry],
    ) -> dict[str, LocalComparison]:
        memory: dict[str, LocalComparison] = {}
        for modality in plan.selected_modalities:
            anchor = valid_anchors.get(modality)
            current_payload = to_dict(evidence.modality_observations[modality])
            if not self.config.use_anchor_comparisons:
                memory[modality] = LocalComparison.unavailable(
                    modality,
                    "Same-modality anchor comparison is disabled by the active ablation.",
                    current_payload,
                )
                continue
            if anchor is None:
                memory[modality] = LocalComparison.unavailable(
                    modality,
                    "No prior same-modality anchor is available in the visible prefix.",
                    current_payload,
                )
                continue
            gap_days = max(0, (_date(current.visit_date) - _date(anchor.visit_date)).days)
            request = self.prompts.compare_anchor(
                modality=self.adapter.modality_map[modality],
                current_images=self._model_images(
                    current.modalities[modality],
                    modality=modality,
                    role="current",
                    visit_handle=evidence.visit_id,
                ),
                anchor_images=self._model_images(
                    anchor.modalities[modality],
                    modality=modality,
                    role="prior",
                    visit_handle=plan.anchor_visit_ids.get(modality, "prior_visit"),
                ),
                current_observation=evidence.modality_observations[modality],
                reference_calibration=calibrations.get(modality),
                anchor_visit_id=plan.anchor_visit_ids.get(modality, "prior_visit"),
                anchor_date=anchor.visit_date,
                current_date=current.visit_date,
                gap_days=gap_days,
            )
            memory[modality] = LocalComparison.from_dict(
                modality,
                anchor.visit_id,
                anchor.visit_date,
                gap_days,
                self._generate(request),
                current_payload,
            )
            trace.append(
                ActionTraceEntry(
                    index=len(trace) + 1,
                    action="compare_anchor",
                    reason_code="needs_same_modality_anchor",
                    modality=modality,
                    visit_id=anchor.visit_id,
                    detail=f"gap_days={gap_days}",
                )
            )
        return memory

    def _needs_retrieval(
        self,
        current: CurrentVisitEvidence,
        memory: Mapping[str, LocalComparison],
        trajectory: TrajectoryState,
    ) -> bool:
        if not self.config.use_cross_subject_memory or self.cross_subject_bank is None:
            return False
        if not self.adapter.memory_policy.cross_subject_enabled:
            return False
        if not self.config.retrieve_when_ambiguous:
            return True
        available = [item for item in memory.values() if item.anchor_available]
        low_confidence = bool(available) and mean(item.confidence for item in available) < 0.55
        return not available or low_confidence or bool(current.uncertainties) or not trajectory.active_event_ids

    def _contradictions(
        self,
        raw_payload: Mapping[str, object],
        belief: BeliefState,
        reconciled: ReconciledEvidence,
        memory: Mapping[str, LocalComparison],
    ) -> list[str]:
        issues = []
        raw_leading = str(raw_payload.get("leading_label") or raw_payload.get("diagnosis_code") or "")
        raw_second = str(raw_payload.get("second_best_label") or raw_payload.get("second_best_diagnosis_code") or "")
        if raw_leading not in self.adapter.task.labels:
            issues.append("leading label is outside the active label space")
        if raw_second not in self.adapter.task.labels or raw_second == raw_leading:
            issues.append("second-best label is invalid or duplicates the leading label")
        if not belief.supporting_evidence:
            issues.append("belief omits supporting evidence")
        if reconciled.conflicts and not belief.conflicting_evidence:
            issues.append("belief omits reconciled conflicting evidence")
        if any(item.anchor_available for item in memory.values()) and not belief.longitudinal_evidence:
            issues.append("belief omits available within-subject longitudinal evidence")
        missing_dimensions = sorted(set(self.adapter.task.belief_dimensions) - set(belief.task_support))
        if missing_dimensions:
            issues.append(f"belief omits task-support fields: {', '.join(missing_dimensions)}")
        if belief.decision_margin == "clear" and belief.open_uncertainties and belief.confidence > 0.85:
            issues.append("clear high-confidence margin conflicts with unresolved uncertainty")
        return issues

    def run_visit(
        self,
        *,
        history: Sequence[VisitInput],
        current: VisitInput,
        trajectory: TrajectoryState,
        prior_belief_label: str | None,
    ) -> VisitResult:
        starting_calls = int(getattr(self.model, "call_count", 0))
        trace = [
            ActionTraceEntry(
                index=1,
                action="inspect_current_visit",
                reason_code="current_visit_not_inspected",
                visit_id=current.visit_id,
            )
        ]
        anchors = self._valid_anchors(history, current)
        plan = self._inspection(history, current, anchors)
        evidence = self._observe(current, plan, self._visit_handle(history, current))
        trace.append(
            ActionTraceEntry(
                index=len(trace) + 1,
                action="observe_current_visit",
                reason_code="current_visit_not_inspected",
                visit_id=current.visit_id,
            )
        )
        calibrations = self._calibrate(current, plan, evidence, trace)
        visit_memory = self._compare(current, plan, anchors, evidence, calibrations, trace)

        if trajectory.active_event_ids:
            trace.append(
                ActionTraceEntry(
                    index=len(trace) + 1,
                    action="review_trajectory",
                    reason_code="ready_for_reconciliation",
                    visit_id=current.visit_id,
                    detail=trajectory.summary,
                )
            )

        cross_context = []
        if self._needs_retrieval(evidence, visit_memory, trajectory):
            signature = build_signature(evidence, visit_memory, trajectory, trace)
            cross_context = self.cross_subject_bank.retrieve(
                subject_id=current.subject_id,
                signature=signature,
                modalities=evidence.modality_observations,
                top_k=self.config.cross_subject_top_k,
            )
            trace.append(
                ActionTraceEntry(
                    index=len(trace) + 1,
                    action="retrieve_cross_subject",
                    reason_code="insufficient_within_subject_evidence",
                    visit_id=current.visit_id,
                    detail=f"neighbors={len(cross_context)}",
                )
            )

        reconcile_request = self.prompts.reconcile(
            current_evidence=evidence,
            reference_calibrations=self._model_calibrations(calibrations),
            visit_memory=self._model_visit_memory(visit_memory),
            trajectory=self._model_trajectory(trajectory),
            cross_subject_context=cross_context,
        )
        reconciled = ReconciledEvidence.from_dict(self._generate(reconcile_request))
        trace.append(
            ActionTraceEntry(
                index=len(trace) + 1,
                action="reconcile_evidence",
                reason_code="unresolved_modality_conflict" if reconciled.conflicts else "ready_for_reconciliation",
                visit_id=current.visit_id,
            )
        )

        evidence_requirements = [
            "ground current visible evidence",
            "use only valid same-modality longitudinal comparisons",
            "preserve conflicts and missing anchors",
        ]
        if self.adapter.memory_policy.cross_subject_enabled and self.config.use_cross_subject_memory:
            evidence_requirements.append("treat cross-subject records as weak analogy-level context")
        decision_context = DecisionContext(
            task_name=self.adapter.task.name,
            label_space=list(self.adapter.task.labels),
            rubric=list(self.adapter.task.belief_rubric),
            belief_dimensions=dict(self.adapter.task.belief_dimensions),
            allowed_context=dict(current.context),
            evidence_requirements=evidence_requirements,
            evidence_snapshot={
                "current_modalities": sorted(evidence.modality_observations),
                "available_same_modality_anchors": sorted(
                    modality for modality, item in visit_memory.items() if item.anchor_available
                ),
                "missing_same_modality_anchors": sorted(
                    modality for modality, item in visit_memory.items() if not item.anchor_available
                ),
                "trajectory_summary": trajectory.summary,
                "dominant_evidence": list(reconciled.dominant_evidence),
                "conflicts": list(reconciled.conflicts),
                "unresolved_uncertainties": list(reconciled.unresolved_uncertainties),
                "cross_subject_neighbors": len(cross_context),
            },
        )
        belief_request = self.prompts.final_belief(
            current_evidence=evidence,
            visit_memory=self._model_visit_memory(visit_memory),
            trajectory=self._model_trajectory(trajectory),
            reconciled=reconciled,
            decision_context=decision_context,
            prior_belief_label=prior_belief_label,
        )
        raw_belief = self._generate(belief_request)
        belief = BeliefState.from_dict(raw_belief, self.adapter.task.labels)
        contradictions = self._contradictions(raw_belief, belief, reconciled, visit_memory)
        repair_performed = False
        if contradictions and self.config.repair_limit > 0:
            repair_request = self.prompts.repair_belief(
                belief=belief,
                current_evidence=evidence,
                visit_memory=self._model_visit_memory(visit_memory),
                trajectory=self._model_trajectory(trajectory),
                reconciled=reconciled,
                decision_context=decision_context,
                contradictions=contradictions,
            )
            belief = BeliefState.from_dict(self._generate(repair_request), self.adapter.task.labels)
            belief.repaired = True
            repair_performed = True
            trace.append(
                ActionTraceEntry(
                    index=len(trace) + 1,
                    action="repair_belief",
                    reason_code="unresolved_modality_conflict",
                    visit_id=current.visit_id,
                    detail="; ".join(contradictions),
                )
            )

        if self.config.use_audit:
            audit_request = self.prompts.audit(
                current_evidence=evidence,
                visit_memory=self._model_visit_memory(visit_memory),
                reconciled=reconciled,
                belief=belief,
                contradictions=contradictions,
                repair_performed=repair_performed,
            )
            audit_payload = self._generate(audit_request)
            audit = AuditRecord(
                decisive_evidence=_list(audit_payload.get("decisive_evidence")),
                evidence_against=_list(audit_payload.get("evidence_against")),
                missing_critical_evidence=_list(audit_payload.get("missing_critical_evidence")),
                next_best_evidence=_list(audit_payload.get("next_best_evidence")),
                contradictions=_list(audit_payload.get("contradictions")),
                repair_performed=repair_performed,
                summary=str(audit_payload.get("summary") or "").strip(),
            )
        else:
            audit = AuditRecord(repair_performed=repair_performed, summary="Audit disabled by active ablation.")
        trace.append(
            ActionTraceEntry(
                index=len(trace) + 1,
                action="finalize_and_audit" if self.config.use_audit else "finalize",
                reason_code="ready_to_finalize",
                visit_id=current.visit_id,
            )
        )
        ending_calls = int(getattr(self.model, "call_count", starting_calls))
        return VisitResult(
            visit=current,
            inspection=plan,
            current_evidence=evidence,
            reference_calibrations=calibrations,
            visit_memory=visit_memory,
            trajectory_before=trajectory,
            cross_subject_context=cross_context,
            reconciled_evidence=reconciled,
            decision_context=decision_context,
            belief=belief,
            audit=audit,
            action_trace=trace,
            model_call_count=max(0, ending_calls - starting_calls),
        )

    @staticmethod
    def build_trajectory_event(result: VisitResult, prior_belief_label: str | None) -> TrajectoryEvent:
        comparisons = [item for item in result.visit_memory.values() if item.anchor_available]
        grade = max((item.change_grade for item in comparisons), default=0)
        dominant = [item.modality for item in comparisons if item.change_grade == grade and grade > 0]
        if not dominant:
            dominant = sorted(result.current_evidence.modality_observations)[:2]
        confidences = [item.confidence for item in comparisons]
        confidences.extend(item.confidence for item in result.current_evidence.modality_observations.values())
        confidence = mean(confidences) if confidences else result.belief.confidence
        shift = ""
        if prior_belief_label:
            shift = f"{prior_belief_label} -> {result.belief.leading_label}"
        return TrajectoryEvent(
            event_id=f"{result.visit.subject_id}:{result.visit.visit_id}",
            subject_id=result.visit.subject_id,
            visit_id=result.visit.visit_id,
            visit_date=result.visit.visit_date,
            summary=result.reconciled_evidence.summary or result.current_evidence.summary,
            change_grade=grade,
            confidence=float(max(0.0, min(1.0, confidence))),
            dominant_modalities=dominant,
            conflicts=list(result.reconciled_evidence.conflicts),
            uncertainties=list(result.reconciled_evidence.unresolved_uncertainties),
            modality_changes={key: to_dict(value) for key, value in result.visit_memory.items()},
            belief_shift=shift,
            observation_only=not comparisons,
        )
