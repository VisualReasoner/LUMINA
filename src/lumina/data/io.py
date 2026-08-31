from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from lumina.adapters.specs import AdapterSpec
from lumina.schemas.states import ImageRef, VisitInput


def split_paths(value: object) -> list[str]:
    if value is None or (not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            values = json.loads(text)
            if isinstance(values, list):
                return [str(item) for item in values if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split("|") if item.strip()]


def parse_reference_candidates(value: object) -> list[dict[str, object]]:
    if value is None or (not isinstance(value, (list, tuple, dict)) and bool(pd.isna(value))):
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [dict(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value or "")
    return pd.Timestamp(parsed).date().isoformat()


def _view_name(path: str, index: int) -> str:
    name = Path(path).name.lower()
    for view in ("axial", "coronal", "sagittal"):
        if view in name:
            return view
    return f"view {index + 1}"


def build_subject_prefix(
    *,
    visit_index: pd.DataFrame,
    routed_target: Mapping[str, object],
    adapter: AdapterSpec,
    require_existing_images: bool = True,
) -> list[VisitInput]:
    subject_id = str(routed_target["subject_id"])
    target_visit_id = str(routed_target["target_visit_id"])
    subject = visit_index[visit_index["subject_id"].astype(str) == subject_id].copy()
    if subject.empty:
        raise ValueError(f"No visits found for subject {subject_id}.")
    subject["visit_date"] = pd.to_datetime(subject["visit_date"], errors="coerce")
    subject = subject.sort_values(["visit_date", "visit_id"]).reset_index(drop=True)
    target_matches = subject.index[subject["visit_id"].astype(str) == target_visit_id].tolist()
    if not target_matches:
        raise ValueError(f"Target visit {target_visit_id} not found for subject {subject_id}.")
    subject = subject.iloc[: target_matches[-1] + 1].copy()

    context = {
        column: routed_target[column]
        for column in adapter.context_columns
        if column in routed_target and pd.notna(routed_target[column])
    }
    visits: list[VisitInput] = []
    for row in subject.to_dict(orient="records"):
        visit_id = str(row["visit_id"])
        visit_handle = f"V{len(visits) + 1}"
        modalities: dict[str, list[ImageRef]] = {}
        references: dict[str, list[dict[str, object]]] = {}
        for spec in adapter.modalities:
            refs = []
            for index, image_path in enumerate(split_paths(row.get(spec.path_column))):
                if require_existing_images and not Path(image_path).is_file():
                    continue
                refs.append(
                    ImageRef(
                        path=image_path,
                        label=f"{spec.name} {_view_name(image_path, index)} at visible visit {visit_handle}",
                        modality=spec.name,
                        visit_id=visit_id,
                    )
                )
            if refs:
                modalities[spec.name] = refs
            if visit_id == target_visit_id and spec.reference_column:
                references[spec.name] = parse_reference_candidates(routed_target.get(spec.reference_column))
        if not modalities:
            continue
        visits.append(
            VisitInput(
                subject_id=subject_id,
                visit_id=visit_id,
                visit_date=_date_text(row["visit_date"]),
                modalities=modalities,
                context=dict(context) if visit_id == target_visit_id else {},
                references=references,
            )
        )
    if not visits or visits[-1].visit_id != target_visit_id:
        raise ValueError(f"Target visit {target_visit_id} has no usable adapter-mapped image inputs.")
    if len(visits) < adapter.task.minimum_prefix_visits:
        raise ValueError(
            f"Target requires at least {adapter.task.minimum_prefix_visits} visible visits; found {len(visits)}."
        )
    return visits


def load_tables(visit_index_csv: str | Path, routed_targets_csv: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    visits = pd.read_csv(visit_index_csv, low_memory=False)
    targets = pd.read_csv(routed_targets_csv, low_memory=False)
    required_visits = {"subject_id", "visit_id", "visit_date"}
    required_targets = {"subject_id", "target_visit_id", "routed_label"}
    if missing := sorted(required_visits - set(visits.columns)):
        raise ValueError(f"Visit index missing columns: {missing}")
    if missing := sorted(required_targets - set(targets.columns)):
        raise ValueError(f"Routed targets missing columns: {missing}")
    return visits, targets
