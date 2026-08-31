from __future__ import annotations

from math import isclose

import pytest

from lumina.memory.cross_subject import CrossSubjectEntry
from lumina.memory.cross_subject import CrossSubjectBank, build_entry
from lumina.adapters import load_adapter
from lumina.agent import EvidenceController
from lumina.models import ReplayModelClient
from pathlib import Path
from lumina.memory.trajectory import TrajectoryConfig, event_utility, update_trajectory
from lumina.schemas.states import (
    ActionTraceEntry,
    CurrentVisitEvidence,
    LocalComparison,
    ModalityObservation,
    ReconciledEvidence,
    TrajectoryEvent,
    TrajectoryState,
)


def _event(event_id: str, date: str, grade: int, confidence: float) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id=event_id,
        subject_id="subject",
        visit_id=event_id,
        visit_date=date,
        summary=event_id,
        change_grade=grade,
        confidence=confidence,
    )


def test_smc_uses_configured_half_life_equation() -> None:
    config = TrajectoryConfig(
        max_active_events=1,
        recency_half_life_days=100,
        change_grade_weight=0.0,
        confidence_weight=0.0,
        recency_weight=1.0,
    )
    older = _event("old", "2020-01-01", 0, 0.0)
    utility = event_utility(older, "2020-04-10", config)
    assert isclose(utility, 0.5, rel_tol=1e-6)


def test_top_k_events_are_retained() -> None:
    state = TrajectoryState()
    state = update_trajectory(state, _event("v1", "2020-01-01", 0, 0.5), TrajectoryConfig(max_active_events=2))
    state = update_trajectory(state, _event("v2", "2021-01-01", 3, 0.9), TrajectoryConfig(max_active_events=2))
    state = update_trajectory(state, _event("v3", "2022-01-01", 1, 0.6), TrajectoryConfig(max_active_events=2))
    assert len(state.active_event_ids) == 2
    assert "v2" in state.active_event_ids


def test_trajectory_configuration_rejects_invalid_event_budget() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TrajectoryConfig(max_active_events=0)


def test_trajectory_rejects_duplicate_event_ids() -> None:
    state = update_trajectory(TrajectoryState(), _event("v1", "2020-01-01", 1, 0.5))
    with pytest.raises(ValueError, match="already exists"):
        update_trajectory(state, _event("v1", "2021-01-01", 2, 0.7))


def test_cross_subject_entry_rejects_label_fields() -> None:
    with pytest.raises(ValueError):
        CrossSubjectEntry.from_dict(
            {
                "entry_id": "x",
                "subject_id": "s",
                "signature": {},
                "summary": "summary",
                "diagnosis_label": "forbidden",
            }
        )


def test_cross_subject_entry_rejects_signature_modalities_outside_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = load_adapter(root / "configs" / "adapters" / "adni_ad_continuum.yaml")
    bank = CrossSubjectBank(
        entries=[
            CrossSubjectEntry(
                entry_id="entry",
                subject_id="development_subject",
                signature={"has_unlisted_modality": 1.0},
                summary="Label-free development evidence.",
                modalities=["Unlisted_Modality"],
                adapter_name=adapter.adapter_name,
            )
        ]
    )
    with pytest.raises(ValueError, match="outside the active adapter"):
        EvidenceController(adapter=adapter, model=ReplayModelClient({}), cross_subject_bank=bank)


def test_controller_rejects_label_text_in_cross_subject_bank() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = load_adapter(root / "configs" / "adapters" / "adni_ad_continuum.yaml")
    bank = CrossSubjectBank(
        entries=[
            CrossSubjectEntry(
                entry_id="entry",
                subject_id="development_subject",
                signature={},
                summary="The retrieved record predicts Preclinical AD.",
                adapter_name=adapter.adapter_name,
            )
        ]
    )
    with pytest.raises(ValueError, match="label terms"):
        EvidenceController(adapter=adapter, model=ReplayModelClient({}), cross_subject_bank=bank)


def test_controller_rejects_cross_subject_entries_without_adapter_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = load_adapter(root / "configs" / "adapters" / "adni_ad_continuum.yaml")
    bank = CrossSubjectBank(
        entries=[
            CrossSubjectEntry(
                entry_id="entry",
                subject_id="development_subject",
                signature={"modality_count": 1.0},
                summary="Label-free evidence.",
            )
        ]
    )
    with pytest.raises(ValueError, match="identify their adapter"):
        EvidenceController(adapter=adapter, model=ReplayModelClient({}), cross_subject_bank=bank)


def test_trajectory_summary_is_direction_neutral_for_cross_disease_transfer() -> None:
    event = TrajectoryEvent(
        event_id="response_event",
        subject_id="s1",
        visit_id="v2",
        visit_date="2021-01-01",
        summary="Marked lesion reduction.",
        change_grade=3,
        confidence=0.9,
        dominant_modalities=["DCE_MRI"],
    )
    updated = update_trajectory(TrajectoryState(), event)
    assert "longitudinal change" in updated.summary
    assert "worsening" not in updated.summary


def test_cross_subject_entry_builder_records_adapter_and_redacts_labels() -> None:
    entry = build_entry(
        entry_id="entry",
        subject_id="development_subject",
        current=CurrentVisitEvidence(
            visit_id="V2",
            modality_observations={"MRI": ModalityObservation("MRI", True, summary="Visible pattern")},
            summary="Current evidence includes a Preclinical AD phrase from a generated trace.",
        ),
        visit_memory={"MRI": LocalComparison(modality="MRI", anchor_available=True, summary="Stable MRI")},
        trajectory=TrajectoryState(summary="No supported longitudinal change."),
        reconciled=ReconciledEvidence(summary="Reconciled evidence."),
        action_trace=[ActionTraceEntry(1, "reconcile_evidence", "ready_for_reconciliation")],
        adapter_name="adni_ad_continuum",
        forbidden_terms=("Preclinical_AD", "Preclinical AD"),
    )
    assert entry.adapter_name == "adni_ad_continuum"
    assert "Preclinical" not in entry.summary
    assert "Current evidence" in entry.summary
