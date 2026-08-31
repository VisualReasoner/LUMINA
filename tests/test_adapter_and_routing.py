from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lumina.adapters import AdapterSpec, load_adapter
from lumina.data.references import attach_reference_metadata, load_reference_candidates, load_subject_context
from lumina.data.targets import RoutedTargetConfig, build_routed_targets
from lumina.data.visit_index import VisitIndexConfig, build_visit_index, load_study_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_adni_adapter_loads() -> None:
    adapter = load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
    assert adapter.task.labels == ("CU_nonAD", "Preclinical_AD", "Symptomatic_AD")
    assert set(adapter.modality_map) == {"MRI", "Amyloid_PET", "Tau_PET"}
    assert adapter.subject_id_aliases == ("PTID",)
    assert adapter.context_aliases["sex"] == ("PTGENDER",)


def test_routing_requires_longitudinal_prefix_and_target_modality() -> None:
    visits = pd.DataFrame(
        [
            {"subject_id": "s1", "visit_id": "s1_v1", "visit_date": "2020-01-01", "mri_paths": "a.png"},
            {"subject_id": "s1", "visit_id": "s1_v2", "visit_date": "2021-01-01", "mri_paths": "b.png"},
            {"subject_id": "s2", "visit_id": "s2_v1", "visit_date": "2021-01-01", "mri_paths": "c.png"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"subject_id": "s1", "visit_id": "s1_v1", "ad_continuum_3way": "CU_nonAD"},
            {"subject_id": "s1", "visit_id": "s1_v2", "ad_continuum_3way": "Preclinical_AD"},
            {"subject_id": "s2", "visit_id": "s2_v1", "ad_continuum_3way": "CU_nonAD"},
        ]
    )
    routed, report = build_routed_targets(
        visit_index=visits,
        label_table=labels,
        config=RoutedTargetConfig.from_adapter(
            load_adapter(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml")
        ),
    )
    assert routed["subject_id"].tolist() == ["s1"]
    assert routed.iloc[0]["target_visit_id"] == "s1_v2"
    assert report["minimum_prefix_visits"] == 2


def test_visit_builder_keeps_repeated_same_modality_as_new_visit() -> None:
    studies = pd.DataFrame(
        [
            {"subject_id": "s1", "study_date": "2020-01-01", "modality": "MRI", "image_path": "mri_a.png"},
            {"subject_id": "s1", "study_date": "2020-01-01", "modality": "MRI", "image_path": "mri_b.png"},
            {"subject_id": "s1", "study_date": "2020-06-01", "modality": "Amyloid_PET", "image_path": "pet.png"},
            {"subject_id": "s1", "study_date": "2020-12-01", "modality": "MRI", "image_path": "mri_followup.png"},
        ]
    )
    studies["study_date"] = pd.to_datetime(studies["study_date"])
    visits = build_visit_index(studies, VisitIndexConfig(within_visit_gap_days=365))
    assert len(visits) == 2
    assert visits.iloc[0]["modalities"] == "Amyloid_PET,MRI"
    assert visits.iloc[0]["mri_paths"] == "mri_a.png|mri_b.png"
    assert visits.iloc[1]["mri_paths"] == "mri_followup.png"


def test_reference_candidates_group_views_without_label_fields(tmp_path: Path) -> None:
    path = tmp_path / "references.csv"
    pd.DataFrame(
        [
            {"modality": "MRI", "candidate_id": "r1", "image_path": "axial.png", "age": 70, "sex": "F"},
            {"modality": "MRI", "candidate_id": "r1", "image_path": "coronal.png", "age": 70, "sex": "F"},
        ]
    ).to_csv(path, index=False)
    payloads = load_reference_candidates(
        path,
        modality_output_columns={"mri": "mri_reference_candidates"},
        matching_fields=("age", "sex"),
    )
    records = json.loads(payloads["mri_reference_candidates"])
    assert records == [
        {
            "candidate_id": "r1",
            "image_paths": ["axial.png", "coronal.png"],
            "sex": "F",
            "age": 70,
        }
    ]


def test_reference_candidates_reject_label_defining_columns(tmp_path: Path) -> None:
    path = tmp_path / "references.csv"
    pd.DataFrame(
        [
            {
                "modality": "MRI",
                "candidate_id": "r1",
                "image_path": "axial.png",
                "diagnosis_label": "not_allowed",
                "age": 70,
                "sex": "F",
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="label-defining or outcome columns"):
        load_reference_candidates(
            path,
            modality_output_columns={"mri": "mri_reference_candidates"},
            matching_fields=("age", "sex"),
        )


def test_reference_candidates_keep_adapter_declared_matching_fields(tmp_path: Path) -> None:
    path = tmp_path / "references.csv"
    pd.DataFrame(
        [
            {
                "modality": "MRI",
                "candidate_id": "r1",
                "image_path": "axial.png",
                "scanner_vendor": "vendor_a",
                "site": "site_1",
            }
        ]
    ).to_csv(path, index=False)
    payloads = load_reference_candidates(
        path,
        modality_output_columns={"mri": "mri_reference_candidates"},
        matching_fields=("scanner_vendor", "site"),
    )
    assert json.loads(payloads["mri_reference_candidates"])[0] == {
        "candidate_id": "r1",
        "image_paths": ["axial.png"],
        "scanner_vendor": "vendor_a",
        "site": "site_1",
    }


def test_reference_candidates_reject_inconsistent_matching_fields(tmp_path: Path) -> None:
    path = tmp_path / "references.csv"
    pd.DataFrame(
        [
            {"modality": "MRI", "candidate_id": "r1", "image_path": "a.png", "age": 70, "sex": "F"},
            {"modality": "MRI", "candidate_id": "r1", "image_path": "b.png", "age": 71, "sex": "F"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent 'age'"):
        load_reference_candidates(
            path,
            modality_output_columns={"mri": "mri_reference_candidates"},
            matching_fields=("age", "sex"),
        )


def test_study_catalog_rejects_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "studies.csv"
    pd.DataFrame(
        [{"subject_id": "s1", "study_date": "not-a-date", "modality": "MRI", "image_path": "a.png"}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid row"):
        load_study_catalog(path)


def test_subject_context_loader_uses_adapter_declared_fields(tmp_path: Path) -> None:
    path = tmp_path / "context.csv"
    pd.DataFrame(
        [{"subject_id": "s1", "scanner_vendor": "vendor_a", "site": "site_1"}]
    ).to_csv(path, index=False)
    context = load_subject_context(path, context_columns=("scanner_vendor", "site"))
    assert context.to_dict(orient="records") == [
        {"subject_id": "s1", "scanner_vendor": "vendor_a", "site": "site_1"}
    ]


def test_adapter_keeps_explicitly_empty_context() -> None:
    adapter = AdapterSpec.from_dict(
        {
            "adapter_name": "context_free",
            "context_columns": [],
            "task": {"name": "task", "label_column": "label", "labels": ["A", "B"]},
            "modalities": [{"name": "Image", "path_column": "image_paths"}],
        }
    )
    assert adapter.context_columns == ()


def test_enabled_references_require_explicit_matching_fields() -> None:
    with pytest.raises(ValueError, match="match_primary and match_secondary"):
        AdapterSpec.from_dict(
            {
                "adapter_name": "missing_reference_match",
                "context_columns": ["age", "sex"],
                "task": {"name": "task", "label_column": "label", "labels": ["A", "B"]},
                "modalities": [
                    {
                        "name": "Image",
                        "path_column": "image_paths",
                        "reference_column": "image_reference_candidates",
                    }
                ],
                "references": {"enabled": True},
            }
        )


def test_subject_context_loader_uses_adapter_declared_aliases(tmp_path: Path) -> None:
    path = tmp_path / "context.csv"
    pd.DataFrame([{"PTID": "s1", "AGE": 70, "PTGENDER": "F"}]).to_csv(path, index=False)
    context = load_subject_context(
        path,
        context_columns=("age", "sex"),
        subject_id_aliases=("PTID",),
        context_aliases={"age": ("AGE",), "sex": ("PTGENDER",)},
    )
    assert context.to_dict(orient="records") == [{"subject_id": "s1", "age": 70, "sex": "F"}]


def test_reference_metadata_fills_missing_context_without_duplicate_columns() -> None:
    routed = pd.DataFrame([{"subject_id": "s1", "age": 70, "sex": None}])
    context = pd.DataFrame([{"subject_id": "s1", "age": 71, "sex": "F"}])
    attached, _ = attach_reference_metadata(
        routed_targets=routed,
        demographics=context,
        context_columns=("age", "sex"),
        required_context_columns=("age", "sex"),
    )
    assert attached.loc[0, "age"] == 70
    assert attached.loc[0, "sex"] == "F"
    assert not any(column.endswith(("_x", "_y")) for column in attached.columns)


def test_reference_metadata_rejects_missing_matching_context() -> None:
    routed = pd.DataFrame([{"subject_id": "s1", "age": 70}])
    with pytest.raises(ValueError, match="complete routed-target context"):
        attach_reference_metadata(
            routed_targets=routed,
            required_context_columns=("age", "sex"),
        )


def test_acrin_routing_requires_all_three_prior_modalities() -> None:
    adapter = load_adapter(ROOT / "configs" / "adapters" / "acrin6698_pcr.yaml")
    visits = pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "visit_id": "t0",
                "visit_date": "2020-01-01",
                "timepoint": "T0",
                "t2_mri_paths": "t0_t2.png",
                "dwi_mri_paths": "t0_dwi.png",
            },
            {
                "subject_id": "s1",
                "visit_id": "t1",
                "visit_date": "2020-02-01",
                "timepoint": "T1",
                "t2_mri_paths": "t1_t2.png",
                "dwi_mri_paths": "t1_dwi.png",
                "dce_mri_paths": "t1_dce.png",
            },
        ]
    )
    labels = pd.DataFrame([{"subject_id": "s1", "visit_id": "t1", "pcr_binary": "pCR"}])
    routed, report = build_routed_targets(
        visit_index=visits,
        label_table=labels,
        config=RoutedTargetConfig.from_adapter(adapter),
    )
    assert routed.empty
    assert report["required_prior_modalities"] == ["T2_MRI", "DWI_MRI", "DCE_MRI"]
