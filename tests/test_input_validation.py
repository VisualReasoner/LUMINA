from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from lumina.adapters import load_adapter
from lumina.agent import EvidenceController
from lumina.data.io import build_subject_prefix
from lumina.models import ReplayModelClient
from lumina.prompts.builders import PromptRequest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller():
    adapter = load_adapter(ROOT / "configs/adapters/adni_ad_continuum.yaml")
    return EvidenceController(adapter=adapter, model=ReplayModelClient({}))


@pytest.mark.parametrize("observation", [
    None, [], {}, {"present": True}, {"findings": []},
    {"present": "true", "findings": []},
    {"present": True, "findings": "finding"},
    {"present": True, "findings": [None]},
    {"present": True, "findings": [], "severity": 4},
    {"present": True, "findings": [], "severity": True},
    {"present": True, "findings": [], "confidence": float("nan")},
    {"present": True, "findings": [], "confidence": float("inf")},
    {"present": True, "findings": [], "confidence": -0.1},
    {"present": True, "findings": [], "confidence": True},
    {"present": True, "findings": [], "summary": []},
])
def test_observation_rejects_malformed_nested_fields(controller, observation):
    with pytest.raises(ValueError):
        controller._validate_stage_payload(
            "observe", {"modality_observations": {"MRI": observation}},
            {"modality_observations": {"MRI": {}}},
        )


@pytest.mark.parametrize("exhaust_retries", [False, True])
def test_nested_observation_failure_uses_bounded_retry(controller, exhaust_retries):
    invalid = {"modality_observations": {"MRI": None}}
    valid = {"modality_observations": {" mri ": {"present": True, "findings": [], "severity": None}}}
    controller.model = ReplayModelClient({"observe": [invalid, invalid if exhaust_retries else valid]})
    request = PromptRequest("observe", "", "", (), {"modality_observations": {"MRI": {}}})
    if exhaust_retries:
        with pytest.raises(ValueError, match="after 2 attempt"):
            controller._generate(request)
    else:
        result = controller._generate(request)
        assert result["modality_observations"]["MRI"]["severity"] is None
    assert controller.model.call_count == 2


@pytest.fixture
def acrin_inputs(tmp_path):
    adapter = load_adapter(ROOT / "configs/adapters/acrin6698_pcr.yaml")
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)
    rows = []
    for index, timepoint in enumerate(("T0", "T1")):
        rows.append({
            "subject_id": "s", "visit_id": f"v{index}", "visit_date": f"202{index}-01-01",
            "timepoint": timepoint,
            **{modality.path_column: str(path) for modality in adapter.modalities},
        })
    return adapter, pd.DataFrame(rows), {"subject_id": "s", "target_visit_id": "v1", "timepoint": "T1"}


def test_complete_loaded_prefix_is_eligible(acrin_inputs):
    adapter, visits, target = acrin_inputs
    prefix = build_subject_prefix(visit_index=visits, routed_target=target, adapter=adapter)
    assert len(prefix) == 2
    assert set(prefix[-1].modalities) == set(adapter.modality_map)


@pytest.mark.parametrize("visit_index, message", [(0, "prior modalities"), (1, "usable modalities")])
def test_missing_files_rechecked_after_loading(acrin_inputs, tmp_path, visit_index, message):
    adapter, visits, target = acrin_inputs
    visits.loc[visit_index, "dwi_mri_paths"] = str(tmp_path / "missing.png")
    with pytest.raises(ValueError, match=message):
        build_subject_prefix(visit_index=visits, routed_target=target, adapter=adapter)


def test_required_any_modality_rechecked(acrin_inputs):
    adapter, visits, target = acrin_inputs
    adapter = replace(adapter, task=replace(adapter.task, required_target_modalities=(), require_any_target_modality=("DWI_MRI",)))
    visits.loc[1, "dwi_mri_paths"] = None
    with pytest.raises(ValueError, match="requires a usable modality"):
        build_subject_prefix(visit_index=visits, routed_target=target, adapter=adapter)


@pytest.mark.parametrize("source", ["visit", "routed", "missing"])
def test_target_filter_checked_at_runtime(acrin_inputs, source):
    adapter, visits, target = acrin_inputs
    if source == "visit":
        visits.loc[1, "timepoint"] = "T3"
    elif source == "routed":
        target["timepoint"] = "T3"
    else:
        visits = visits.drop(columns="timepoint")
        target.pop("timepoint")
    with pytest.raises(ValueError, match="timepoint"):
        build_subject_prefix(visit_index=visits, routed_target=target, adapter=adapter)


def test_target_filter_may_come_from_routed_label_metadata(acrin_inputs):
    adapter, visits, target = acrin_inputs
    prefix = build_subject_prefix(visit_index=visits.drop(columns="timepoint"), routed_target=target, adapter=adapter)
    assert len(prefix) == 2
