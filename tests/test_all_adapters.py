from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from lumina.adapters import load_adapter
from lumina.agent import ControllerConfig, EvidenceController, LUMINARuntime
from lumina.data.io import build_subject_prefix
from lumina.data.targets import RoutedTargetConfig, build_routed_targets
from lumina.models import ReplayModelClient
from lumina.memory.cross_subject import CrossSubjectBank, CrossSubjectEntry


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (
    ("adni_ad_continuum.yaml", "Symptomatic_AD"),
    ("adni_progression_forecasting.yaml", "clear_progression"),
    ("ppmi_parkinsonian_state.yaml", "PD"),
    ("acrin6698_pcr.yaml", "pCR"),
)


class RecordingReplayModelClient(ReplayModelClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests: list[str] = []

    def generate_json(self, **kwargs):
        image_labels = " ".join(image.label for image in kwargs["images"])
        self.requests.append(f"{kwargs['system_prompt']}\n{kwargs['user_prompt']}\n{image_labels}")
        return super().generate_json(**kwargs)


def _responses(adapter, leading_label: str) -> dict[str, list[dict]]:
    modalities = [item.name for item in adapter.modalities]
    observations = {
        modality: {
            "present": True,
            "findings": [f"synthetic {modality} finding"],
            "severity": 1,
            "confidence": 0.8,
        }
        for modality in modalities
    }
    second = next(label for label in adapter.task.labels if label != leading_label)
    task_support = {key: "synthetic support" for key in adapter.task.belief_dimensions}
    return {
        "inspect": [
            {"selected_modalities": modalities, "visit_questions": {}},
            {"selected_modalities": modalities, "visit_questions": {}},
        ],
        "observe": [
            {"modality_observations": observations, "summary": "Synthetic baseline evidence."},
            {"modality_observations": observations, "summary": "Synthetic target evidence."},
        ],
        "compare_anchor": [
            {
                "direction": "moderate_increase",
                "magnitude": "moderate",
                "change_grade": 2,
                "localized_evidence": [f"synthetic {modality} change"],
                "summary": f"Synthetic {modality} change.",
                "confidence": 0.8,
                "comparison_quality": "strong",
            }
            for modality in modalities
        ],
        "reconcile": [
            {"summary": "Synthetic baseline reconciliation."},
            {"agreements": ["synthetic agreement"], "summary": "Synthetic target reconciliation."},
        ],
        "final_belief": [
            {
                "leading_label": leading_label,
                "second_best_label": second,
                "supporting_evidence": ["synthetic evidence"],
                "task_support": task_support,
                "summary": "Synthetic baseline belief.",
            },
            {
                "leading_label": leading_label,
                "second_best_label": second,
                "supporting_evidence": ["synthetic evidence"],
                "longitudinal_evidence": ["synthetic same-modality change"],
                "task_support": task_support,
                "summary": "Synthetic target belief.",
            },
        ],
        "audit": [
            {"summary": "Synthetic baseline audit."},
            {"summary": "Synthetic target audit."},
        ],
    }


@pytest.mark.parametrize(("adapter_name", "target_label"), ADAPTERS)
def test_every_adapter_routes_and_runs_the_shared_replay_pipeline(
    tmp_path: Path,
    adapter_name: str,
    target_label: str,
) -> None:
    adapter = load_adapter(ROOT / "configs" / "adapters" / adapter_name)
    image = tmp_path / f"{adapter.adapter_name}.png"
    Image.new("RGB", (8, 8), "gray").save(image)
    rows = []
    for index, date in enumerate(("2020-01-01", "2021-01-01"), start=1):
        row = {
            "subject_id": "private_subject_identifier",
            "visit_id": f"private_subject_identifier_v{index}",
            "visit_date": date,
            "age": 70 + index,
            "sex": "F",
            "race": "synthetic",
        }
        for modality in adapter.modalities:
            row[modality.path_column] = str(image)
        if adapter.task.target_filter_column:
            row[adapter.task.target_filter_column] = "T0" if index == 1 else "T1"
        rows.append(row)
    visit_index = pd.DataFrame(rows)
    labels = pd.DataFrame(
        [
            {
                "subject_id": "private_subject_identifier",
                "visit_id": "private_subject_identifier_v2",
                adapter.task.label_column: target_label,
            }
        ]
    )
    routed, report = build_routed_targets(
        visit_index=visit_index,
        label_table=labels,
        config=RoutedTargetConfig.from_adapter(adapter),
    )
    assert report["routed_targets"] == 1
    assert "history_amyloid_visible" not in routed.columns
    assert routed.iloc[0]["routed_label"] == target_label

    prefix = build_subject_prefix(
        visit_index=visit_index,
        routed_target=routed.iloc[0].to_dict(),
        adapter=adapter,
    )
    model = RecordingReplayModelClient(_responses(adapter, target_label))
    external_adapter = adapter_name in {"ppmi_parkinsonian_state.yaml", "acrin6698_pcr.yaml"}
    bank = (
        CrossSubjectBank(
            entries=[
                CrossSubjectEntry(
                    entry_id="development_entry",
                    subject_id="development_subject",
                    signature={"modality_count": 1.0},
                    summary="Label-free development evidence.",
                    adapter_name=adapter.adapter_name,
                )
            ]
        )
        if external_adapter
        else None
    )
    runtime = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=True),
            cross_subject_bank=bank,
        )
    )
    result = runtime.run_subject(prefix)
    assert result.prediction == target_label
    assert set(result.visits[-1].visit_memory) == {item.name for item in adapter.modalities}
    assert set(result.visits[-1].belief.task_support) == set(adapter.task.belief_dimensions)
    assert all("private_subject_identifier" not in request for request in model.requests)
    assert all(adapter.task.name.replace("_", " ") in request for request in model.requests)
    assert all(modality.role in model.requests[0] for modality in adapter.modalities)
    if external_adapter:
        assert adapter.memory_policy.cross_subject_enabled is False
        assert all(
            item.action != "retrieve_cross_subject"
            for visit_result in result.visits
            for item in visit_result.action_trace
        )
