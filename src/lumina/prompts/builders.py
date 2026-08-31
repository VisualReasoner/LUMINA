from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from lumina.adapters.specs import AdapterSpec, ModalitySpec
from lumina.schemas.states import ImageRef, to_dict


JSON_RULES = (
    "Return one valid JSON object only. Do not use markdown or add prose outside JSON. "
    "Use concise evidence phrases and do not invent findings, symptoms, or future outcomes. "
    "When a label is requested, use only the supplied label space."
)


@dataclass(frozen=True)
class PromptRequest:
    stage: str
    system_prompt: str
    user_prompt: str
    images: tuple[ImageRef, ...]
    schema: Mapping[str, object]


def _dump(value: object) -> str:
    return json.dumps(to_dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class PromptBuilder:
    def __init__(self, adapter: AdapterSpec):
        self.adapter = adapter

    @property
    def role(self) -> str:
        task_name = self.adapter.task.name.replace("_", " ")
        modality_roles = "; ".join(
            f"{modality.name}: {modality.role}" for modality in self.adapter.modalities
        )
        return (
            f"You are the evidence-construction component for the {task_name} task in a longitudinal "
            "multimodal reasoning agent. "
            f"The active evidence roles are {modality_roles}. "
            "Keep current findings, within-subject change, trajectory context, and cross-subject analogies separate. "
            "Use only the supplied visible prefix. "
        )

    def inspect(
        self,
        *,
        visit_id: str,
        available_modalities: Sequence[str],
        history_inventory: Sequence[dict],
        valid_anchors: Mapping[str, dict],
        context: Mapping[str, object],
    ) -> PromptRequest:
        schema = {
            "selected_modalities": list(available_modalities),
            "anchor_visit_ids": {"modality": "visit_id"},
            "visit_questions": {"modality": ["question"]},
            "uncertainties_to_resolve": ["uncertainty"],
            "summary": "short plan",
        }
        rules = "\n".join(f"- {rule}" for rule in self.adapter.task.inspection_rules)
        user = (
            f"Plan evidence collection for visit {visit_id}. Do not predict a label.\n"
            f"Available current modalities: {_dump(list(available_modalities))}\n"
            f"Visible history inventory: {_dump(list(history_inventory))}\n"
            f"Controller-validated latest same-modality anchors: {_dump(dict(valid_anchors))}\n"
            f"Allowed pre-target context: {_dump(dict(context))}\n"
            f"Task inspection rules:\n{rules}\n"
            "Select only current modalities and only controller-validated anchors. "
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("inspect", self.role + "Plan before observing.", user, (), schema)

    def observe(
        self,
        *,
        visit_id: str,
        modality_images: Mapping[str, Sequence[ImageRef]],
        visit_questions: Mapping[str, Sequence[str]],
    ) -> PromptRequest:
        schema = {
            "modality_observations": {
                modality: {
                    "present": True,
                    "findings": ["visible finding"],
                    "severity": 0,
                    "summary": "current visual state",
                    "confidence": 0.5,
                    "uncertainty": "",
                }
                for modality in modality_images
            },
            "summary": "current visit only",
            "multimodal_consistency": "convergent|complementary|conflicting|unknown",
            "uncertainties": ["uncertainty"],
        }
        modality_rules = []
        images = []
        for modality, refs in modality_images.items():
            spec = self.adapter.modality_map[modality]
            modality_rules.append(
                {
                    "modality": modality,
                    "role": spec.role,
                    "questions": list(visit_questions.get(modality) or spec.observation_questions),
                }
            )
            images.extend(refs)
        user = (
            f"Ground what is visible at current visit {visit_id} before using longitudinal context.\n"
            f"Modality roles and questions: {_dump(modality_rules)}\n"
            "Describe current visible findings and uncertainty only. Do not infer change, progression, symptoms, "
            "diagnosis, or future outcome in this stage. Severity is an ordinal visual grade from 0 to 3, not a label.\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("observe", self.role + "Ground the current visit.", user, tuple(images), schema)

    def calibrate_reference(
        self,
        *,
        modality: ModalitySpec,
        current_images: Sequence[ImageRef],
        reference_images: Sequence[ImageRef],
        current_observation: object,
        reference_id: str,
    ) -> PromptRequest:
        schema = {
            "direction": "below|similar|subtle_above|moderate_above|marked_above|uncertain",
            "magnitude": "none|subtle|moderate|marked|uncertain",
            "summary": "reference-relative current visual read",
            "confidence": 0.5,
            "uncertainty": "",
        }
        user = (
            f"Calibrate the current {modality.name} read against reference comparator {reference_id}.\n"
            f"Modality role: {modality.role}\n"
            f"Grounded current observation: {_dump(current_observation)}\n"
            "The reference is a visual comparator, not a diagnosis-bearing exemplar. Compare visible appearance only. "
            "Do not infer longitudinal change from this comparison.\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        labeled = [*current_images, *reference_images]
        return PromptRequest("calibrate_reference", self.role + "Calibrate current perception.", user, tuple(labeled), schema)

    def compare_anchor(
        self,
        *,
        modality: ModalitySpec,
        current_images: Sequence[ImageRef],
        anchor_images: Sequence[ImageRef],
        current_observation: object,
        reference_calibration: object | None,
        anchor_visit_id: str,
        anchor_date: str,
        current_date: str,
        gap_days: int,
    ) -> PromptRequest:
        schema = {
            "direction": "decreased|stable|subtle_increase|moderate_increase|marked_increase|mixed|uncertain",
            "magnitude": "none|subtle|moderate|marked|uncertain",
            "change_grade": 0,
            "localized_evidence": ["visible same-modality change"],
            "summary": "local longitudinal comparison",
            "confidence": 0.5,
            "uncertainty": "",
            "comparison_quality": "strong|adequate|weak|uninterpretable",
        }
        user = (
            f"Compare current {modality.name} at {current_date} with the prior same-modality anchor "
            f"{anchor_visit_id} at {anchor_date} ({gap_days} days earlier).\n"
            f"Modality role: {modality.role}\n"
            f"Comparison questions: {_dump(list(modality.comparison_questions))}\n"
            f"Grounded current observation: {_dump(current_observation)}\n"
            f"Current reference calibration, if available: {_dump(reference_calibration)}\n"
            "Use only the two same-modality studies for change. Account for elapsed time. "
            "Grade 0 means no supported change, 1 subtle, 2 moderate, and 3 marked. "
            "If image quality prevents comparison, report uncertainty rather than stability.\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest(
            "compare_anchor",
            self.role + "Measure local same-modality change.",
            user,
            tuple([*current_images, *anchor_images]),
            schema,
        )

    def reconcile(
        self,
        *,
        current_evidence: object,
        reference_calibrations: object,
        visit_memory: object,
        trajectory: object,
        cross_subject_context: object,
    ) -> PromptRequest:
        schema = {
            "agreements": ["agreement"],
            "complementary_evidence": ["complementary evidence"],
            "conflicts": ["conflict"],
            "unresolved_uncertainties": ["uncertainty"],
            "dominant_evidence": ["dominant evidence"],
            "summary": "reconciled evidence state",
        }
        rules = "\n".join(f"- {rule}" for rule in self.adapter.task.reconciliation_rules)
        safe_cross_subject_context = [
            {
                "neighbor": f"N{index}",
                "similarity": getattr(item, "similarity", None),
                "summary": getattr(item, "summary", ""),
                "modalities": getattr(item, "modalities", []),
                "audit_pattern": getattr(item, "audit_pattern", ""),
            }
            for index, item in enumerate(cross_subject_context or [], start=1)
        ]
        user = (
            "Reconcile the structured evidence without averaging modalities into one score. Preserve whether sources "
            "are convergent, complementary, conflicting, or unavailable. Subject-specific evidence outranks "
            "cross-subject analogy. Do not predict the label yet.\n"
            f"Current evidence: {_dump(current_evidence)}\n"
            f"Reference calibrations: {_dump(reference_calibrations)}\n"
            f"Visit memory: {_dump(visit_memory)}\n"
            f"Active trajectory memory: {_dump(trajectory)}\n"
            f"Optional cross-subject analogies: {_dump(safe_cross_subject_context)}\n"
            f"Task reconciliation rules:\n{rules}\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("reconcile", self.role + "Construct the reconciled evidence state.", user, (), schema)

    def final_belief(
        self,
        *,
        current_evidence: object,
        visit_memory: object,
        trajectory: object,
        reconciled: object,
        decision_context: object,
        prior_belief_label: str | None,
    ) -> PromptRequest:
        labels = list(self.adapter.task.labels)
        schema = {
            "leading_label": labels[0],
            "second_best_label": labels[1],
            "decision_margin": "clear|moderate|narrow",
            "confidence": 0.5,
            "supporting_evidence": ["support"],
            "conflicting_evidence": ["conflict"],
            "longitudinal_evidence": ["within-subject evidence"],
            "open_uncertainties": ["uncertainty"],
            "task_support": dict(self.adapter.task.belief_dimensions),
            "summary": "one sentence",
        }
        user = (
            f"Form the final task-aware belief for label space {_dump(labels)}.\n"
            f"Decision context: {_dump(decision_context)}\n"
            f"Prior visit belief label, for belief-shift context only: {_dump(prior_belief_label)}\n"
            f"Current evidence: {_dump(current_evidence)}\n"
            f"Visit memory: {_dump(visit_memory)}\n"
            f"Trajectory memory: {_dump(trajectory)}\n"
            f"Reconciled evidence: {_dump(reconciled)}\n"
            "Apply the decision-context rubric in the listed order.\n"
            "Fill every task_support key with a concise evidence-based assessment; the schema values describe "
            "what each field must assess.\n"
            "Use a qualitative differential, not invented probability vectors. The second-best label must differ "
            "from the leading label.\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("final_belief", self.role + "Form a task-specific belief.", user, (), schema)

    def repair_belief(
        self,
        *,
        belief: object,
        current_evidence: object,
        visit_memory: object,
        trajectory: object,
        reconciled: object,
        decision_context: object,
        contradictions: Sequence[str],
    ) -> PromptRequest:
        labels = list(self.adapter.task.labels)
        schema = {
            "leading_label": labels[0],
            "second_best_label": labels[1],
            "decision_margin": "clear|moderate|narrow",
            "confidence": 0.5,
            "supporting_evidence": ["support"],
            "conflicting_evidence": ["conflict"],
            "longitudinal_evidence": ["within-subject evidence"],
            "open_uncertainties": ["uncertainty"],
            "task_support": dict(self.adapter.task.belief_dimensions),
            "summary": "one sentence",
        }
        user = (
            "Repair the belief once without inventing evidence. Change only fields needed to resolve the listed "
            "contradictions. Keep uncertainty when the evidence remains incomplete.\n"
            f"Current belief: {_dump(belief)}\n"
            f"Current evidence: {_dump(current_evidence)}\n"
            f"Visit memory: {_dump(visit_memory)}\n"
            f"Trajectory memory: {_dump(trajectory)}\n"
            f"Reconciled evidence: {_dump(reconciled)}\n"
            f"Decision context: {_dump(decision_context)}\n"
            f"Contradictions: {_dump(list(contradictions))}\n"
            f"Allowed labels: {_dump(labels)}\n"
            "Fill every task_support key and keep each value grounded in the supplied evidence state.\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("repair_belief", self.role + "Run one bounded belief repair.", user, (), schema)

    def audit(
        self,
        *,
        current_evidence: object,
        visit_memory: object,
        reconciled: object,
        belief: object,
        contradictions: Sequence[str],
        repair_performed: bool,
    ) -> PromptRequest:
        schema = {
            "decisive_evidence": ["decisive evidence"],
            "evidence_against": ["counterevidence"],
            "missing_critical_evidence": ["missing evidence"],
            "next_best_evidence": ["most useful additional evidence"],
            "contradictions": ["remaining contradiction"],
            "summary": "audit summary",
        }
        user = (
            "Audit whether the final belief is grounded, whether expected local comparisons were completed, and "
            "whether unresolved conflicts are represented. Do not change the prediction in this stage.\n"
            f"Current evidence: {_dump(current_evidence)}\n"
            f"Visit memory: {_dump(visit_memory)}\n"
            f"Reconciled evidence: {_dump(reconciled)}\n"
            f"Final belief: {_dump(belief)}\n"
            f"Controller contradictions before repair: {_dump(list(contradictions))}\n"
            f"Repair performed: {str(repair_performed).lower()}\n"
            f"Output schema: {_dump(schema)}\n{JSON_RULES}"
        )
        return PromptRequest("audit", self.role + "Produce an auditable trace.", user, (), schema)
