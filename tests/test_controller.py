from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from lumina.adapters import load_adapter
from lumina.agent import ControllerConfig, EvidenceController, LUMINARuntime
from lumina.models import ReplayModelClient
from lumina.memory.cross_subject import CrossSubjectBank, CrossSubjectEntry
from lumina.schemas.states import ImageRef, VisitInput
from lumina.schemas.states import TrajectoryEvent, TrajectoryState


ROOT = Path(__file__).resolve().parents[1]


def _visit(subject: str, visit_id: str, date: str, image_path: Path) -> VisitInput:
    return VisitInput(
        subject_id=subject,
        visit_id=visit_id,
        visit_date=date,
        modalities={
            "MRI": [ImageRef(path=str(image_path), label=f"MRI {visit_id}", modality="MRI", visit_id=visit_id)]
        },
    )


def test_two_visit_pipeline_preserves_stage_order(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    responses = {
        "inspect": [
            {"selected_modalities": ["MRI"], "visit_questions": {"MRI": ["Inspect structure"]}},
            {"selected_modalities": ["MRI"], "visit_questions": {"MRI": ["Inspect structure"]}},
        ],
        "observe": [
            {
                "modality_observations": {
                    "MRI": {"present": True, "findings": ["mild atrophy"], "severity": 1, "confidence": 0.7}
                },
                "summary": "Mild structural change.",
            },
            {
                "modality_observations": {
                    "MRI": {"present": True, "findings": ["greater atrophy"], "severity": 2, "confidence": 0.8}
                },
                "summary": "Moderate structural change.",
            },
        ],
        "compare_anchor": [
            {
                "direction": "moderate_increase",
                "magnitude": "moderate",
                "change_grade": 2,
                "localized_evidence": ["greater medial temporal atrophy"],
                "summary": "MRI worsened.",
                "confidence": 0.8,
                "comparison_quality": "strong",
            }
        ],
        "reconcile": [
            {"agreements": [], "conflicts": [], "summary": "Current evidence only."},
            {"agreements": ["MRI worsening"], "conflicts": [], "summary": "Supported structural worsening."},
        ],
        "final_belief": [
            {
                "leading_label": "CU_nonAD",
                "second_best_label": "Preclinical_AD",
                "supporting_evidence": ["limited current burden"],
                    "task_support": {key: "limited initial evidence" for key in adapter.task.belief_dimensions},
                "summary": "Limited evidence.",
            },
            {
                "leading_label": "Symptomatic_AD",
                "second_best_label": "Preclinical_AD",
                "supporting_evidence": ["meaningful structural worsening"],
                "longitudinal_evidence": ["MRI worsened against prior MRI"],
                    "task_support": {key: "supported target evidence" for key in adapter.task.belief_dimensions},
                "summary": "Structural worsening supports symptomatic stage.",
            },
        ],
        "audit": [
            {"decisive_evidence": ["current MRI"], "summary": "Grounded."},
            {"decisive_evidence": ["MRI worsening"], "summary": "Grounded and longitudinal."},
        ],
    }
    model = ReplayModelClient(responses)
    runtime = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False),
        )
    )
    result = runtime.run_subject(
        [
            _visit("s1", "v1", "2020-01-01", image_path),
            _visit("s1", "v2", "2021-01-01", image_path),
        ]
    )
    assert result.prediction == "Symptomatic_AD"
    assert result.visits[0].visit_memory["MRI"].current_evidence["findings"] == ["mild atrophy"]
    assert result.visits[0].visit_memory["MRI"].anchor_available is False
    assert result.visits[1].visit_memory["MRI"].anchor_visit_id == "v1"
    assert result.visits[1].visit_memory["MRI"].current_evidence["findings"] == ["greater atrophy"]
    second_actions = [item.action for item in result.visits[1].action_trace]
    assert second_actions.index("observe_current_visit") < second_actions.index("compare_anchor")
    assert result.trajectory.events[-1].change_grade == 2


def test_invalid_stage_output_gets_one_constrained_retry(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    task_support = {key: "synthetic support" for key in adapter.task.belief_dimensions}
    model = ReplayModelClient(
        {
            "inspect": [{"selected_modalities": ["MRI"]}],
            "observe": [
                {
                    "modality_observations": {"MRI": {"present": True, "findings": ["finding"]}},
                    "summary": "Grounded.",
                }
            ],
            "reconcile": [{"summary": "Reconciled."}],
            "final_belief": [
                {"summary": "Malformed belief."},
                {
                    "leading_label": "CU_nonAD",
                    "second_best_label": "Preclinical_AD",
                    "supporting_evidence": ["finding"],
                    "task_support": task_support,
                    "summary": "Valid belief.",
                },
            ],
            "audit": [{"summary": "Audited."}],
        }
    )
    result = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False, schema_retry_limit=1),
        )
    ).run_subject([_visit("s1", "v1", "2020-01-01", image_path)])
    assert result.prediction == "CU_nonAD"
    assert result.call_count == 6


def test_runtime_rejects_duplicate_visit_ids(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    runtime = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=ReplayModelClient({}),
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False),
        )
    )
    with pytest.raises(ValueError, match="duplicate visit_id"):
        runtime.run_subject(
            [
                _visit("s1", "v1", "2020-01-01", image_path),
                _visit("s1", "v1", "2021-01-01", image_path),
            ]
        )


def test_target_mode_consumes_prior_trajectory_without_replaying_history(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    model = ReplayModelClient(
        {
            "inspect": [{"selected_modalities": ["MRI"]}],
            "observe": [
                {
                    "modality_observations": {"MRI": {"present": True, "findings": ["finding"]}},
                    "summary": "Target evidence.",
                }
            ],
            "compare_anchor": [
                {
                    "direction": "stable",
                    "magnitude": "none",
                    "change_grade": 0,
                    "summary": "No supported change.",
                }
            ],
            "reconcile": [{"summary": "Reconciled target and prior trajectory."}],
            "final_belief": [
                {
                    "leading_label": "CU_nonAD",
                    "second_best_label": "Preclinical_AD",
                    "supporting_evidence": ["stable MRI"],
                    "longitudinal_evidence": ["no supported MRI change"],
                    "task_support": {key: "synthetic support" for key in adapter.task.belief_dimensions},
                    "summary": "Target belief.",
                }
            ],
            "audit": [{"summary": "Target audit."}],
        }
    )
    prior_event = TrajectoryEvent(
        event_id="prior_event",
        subject_id="s1",
        visit_id="v0",
        visit_date="2019-01-01",
        summary="Prior compact event.",
        change_grade=1,
        confidence=0.7,
    )
    prior = TrajectoryState(
        events=[prior_event],
        active_event_ids=[prior_event.event_id],
        weights={prior_event.event_id: 1.0},
        selected_mass=1.0,
        summary="One prior compact event.",
    )
    result = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False),
        )
    ).run_target(
        [_visit("s1", "v1", "2020-01-01", image_path), _visit("s1", "v2", "2021-01-01", image_path)],
        prior_trajectory=prior,
    )
    assert len(result.visits) == 1
    assert result.visits[0].trajectory_before.summary == "One prior compact event."
    assert result.visits[0].visit_memory["MRI"].anchor_visit_id == "v1"
    assert result.call_count == result.target_call_count == 6
    assert result.history_call_count == 0


def test_target_mode_ignores_cached_trajectory_when_memory_is_disabled(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    model = ReplayModelClient(
        {
            "inspect": [{"selected_modalities": ["MRI"]}],
            "observe": [
                {
                    "modality_observations": {"MRI": {"present": True, "findings": ["finding"]}},
                    "summary": "Target evidence.",
                }
            ],
            "reconcile": [{"summary": "Current target evidence only."}],
            "final_belief": [
                {
                    "leading_label": "CU_nonAD",
                    "second_best_label": "Preclinical_AD",
                    "supporting_evidence": ["current evidence"],
                    "task_support": {key: "current evidence" for key in adapter.task.belief_dimensions},
                    "summary": "Target belief.",
                }
            ],
            "audit": [{"summary": "Target audit."}],
        }
    )
    prior_event = TrajectoryEvent(
        event_id="prior_event",
        subject_id="s1",
        visit_id="v0",
        visit_date="2019-01-01",
        summary="Prior compact event.",
        change_grade=1,
        confidence=0.7,
    )
    prior = TrajectoryState(
        events=[prior_event],
        active_event_ids=[prior_event.event_id],
        weights={prior_event.event_id: 1.0},
        selected_mass=1.0,
        summary="One prior compact event.",
    )
    result = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False),
        ),
        use_trajectory_memory=False,
    ).run_target([_visit("s1", "v1", "2020-01-01", image_path)], prior_trajectory=prior)
    assert result.visits[0].trajectory_before.events == []
    assert result.trajectory.events == []


def test_reference_calibration_precedes_optional_cross_subject_retrieval(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), color="gray").save(image_path)
    Image.new("RGB", (8, 8), color="black").save(reference_path)
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    model = ReplayModelClient(
        {
            "inspect": [{"selected_modalities": ["MRI"]}],
            "observe": [
                {
                    "modality_observations": {
                        "MRI": {"present": True, "findings": ["visible pattern"], "severity": 1}
                    },
                    "summary": "Grounded current evidence.",
                }
            ],
            "calibrate_reference": [
                {
                    "direction": "subtle_above",
                    "magnitude": "subtle",
                    "summary": "Subtle reference-relative difference.",
                }
            ],
            "reconcile": [{"summary": "Reconciled subject and analogy evidence."}],
            "final_belief": [
                {
                    "leading_label": "CU_nonAD",
                    "second_best_label": "Preclinical_AD",
                    "supporting_evidence": ["limited visible evidence"],
                    "task_support": {key: "limited" for key in adapter.task.belief_dimensions},
                    "summary": "Limited evidence.",
                }
            ],
            "audit": [{"summary": "Audited."}],
        }
    )
    bank = CrossSubjectBank(
        entries=[
            CrossSubjectEntry(
                entry_id="development_entry",
                subject_id="development_subject",
                signature={"modality_count": 1.0, "has_mri": 1.0},
                summary="Label-free similar imaging evidence.",
                modalities=["MRI"],
                adapter_name=adapter.adapter_name,
            )
        ]
    )
    visit = _visit("s1", "v1", "2020-01-01", image_path)
    visit.context = {"age": 71, "sex": "F", "race": "synthetic"}
    visit.references = {
        "MRI": [
            {
                "candidate_id": "reference_subject",
                "subject_id": "reference_subject",
                "age": 70,
                "sex": "F",
                "image_paths": [str(reference_path)],
            }
        ]
    }
    result = LUMINARuntime(
        controller=EvidenceController(adapter=adapter, model=model, cross_subject_bank=bank)
    ).run_subject([visit])
    visit_result = result.visits[0]
    actions = [item.action for item in visit_result.action_trace]
    assert set(visit_result.reference_calibrations) == {"MRI"}
    assert len(visit_result.cross_subject_context) == 1
    assert actions.index("calibrate_reference") < actions.index("retrieve_cross_subject")
    assert actions.index("retrieve_cross_subject") < actions.index("reconcile_evidence")
