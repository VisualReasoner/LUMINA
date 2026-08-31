from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


REQUIRED_STUDY_COLUMNS = (
    "subject_id",
    "study_date",
    "modality",
    "image_path",
)


@dataclass(frozen=True)
class VisitIndexConfig:
    within_visit_gap_days: int = 0
    study_id_column: str = "study_id"


def modality_to_column_key(modality: object) -> str:
    text = str(modality or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError("Modality value cannot be empty.")
    return f"{text}_paths"


def load_study_catalog(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_STUDY_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"Study catalog missing required columns: {missing}")
    working = table.copy()
    working["study_date"] = pd.to_datetime(working["study_date"], errors="coerce")
    invalid = working[["subject_id", "study_date", "modality", "image_path"]].isna().any(axis=1)
    invalid |= working["modality"].map(lambda value: not str(value).strip())
    invalid |= working["image_path"].map(lambda value: not str(value).strip())
    if invalid.any():
        raise ValueError(f"Study catalog contains {int(invalid.sum())} incomplete or invalid row(s).")
    working["subject_id"] = working["subject_id"].astype(str)
    working["modality"] = working["modality"].astype(str).str.strip()
    working["image_path"] = working["image_path"].astype(str)
    return working.sort_values(["subject_id", "study_date", "modality"]).reset_index(drop=True)


def build_visit_index(studies: pd.DataFrame, config: VisitIndexConfig | None = None) -> pd.DataFrame:
    cfg = config or VisitIndexConfig()
    rows: list[dict[str, object]] = []

    for subject_id, group in studies.groupby("subject_id", sort=False):
        subject = group.sort_values(["study_date", "modality", "image_path"]).reset_index(drop=True)
        grouping = ["study_date", "modality"]
        if cfg.study_id_column in subject.columns:
            grouping.append(cfg.study_id_column)
        study_records: list[dict[str, object]] = []
        for keys, study_group in subject.groupby(grouping, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {
                "study_date": pd.Timestamp(keys[0]),
                "modality": str(keys[1]),
                "paths": [str(value) for value in study_group["image_path"].tolist()],
            }
            if len(keys) > 2 and pd.notna(keys[2]):
                record["study_id"] = str(keys[2])
            study_records.append(record)
        study_records.sort(key=lambda item: (item["study_date"], item["modality"], item.get("study_id", "")))

        clusters: list[list[dict[str, object]]] = []
        current_rows: list[dict[str, object]] = []
        current_anchor_date: pd.Timestamp | None = None
        seen_modalities: set[str] = set()
        for record in study_records:
            study_date = pd.Timestamp(record["study_date"])
            modality = str(record["modality"])
            if current_anchor_date is None:
                current_rows = [record]
                current_anchor_date = study_date
                seen_modalities = {modality}
                continue
            gap_days = int((study_date - current_anchor_date).days)
            if gap_days <= cfg.within_visit_gap_days and modality not in seen_modalities:
                current_rows.append(record)
                seen_modalities.add(modality)
                continue
            clusters.append(current_rows)
            current_rows = [record]
            current_anchor_date = study_date
            seen_modalities = {modality}
        if current_rows:
            clusters.append(current_rows)

        for visit_counter, current_rows in enumerate(clusters, start=1):
            dates = sorted(pd.Timestamp(row["study_date"]) for row in current_rows)
            median_date = dates[len(dates) // 2]
            visit_date = min(dates, key=lambda value: (abs((value - median_date).days), value))
            row = {
                "subject_id": str(subject_id),
                "visit_id": f"{subject_id}_v{visit_counter:02d}",
                "visit_date": pd.Timestamp(visit_date),
                "modalities": ",".join(sorted(str(item["modality"]) for item in current_rows)),
                "n_studies": len(current_rows),
                "visit_span_days": int(
                    (max(dates) - min(dates)).days
                )
                if len(current_rows) > 1
                else 0,
            }
            for study in current_rows:
                modality = str(study["modality"])
                key = modality_to_column_key(modality)
                row[key] = "|".join(study["paths"])
                row[key.removesuffix("_paths") + "_date"] = pd.Timestamp(study["study_date"])
                if study.get("study_id"):
                    row[key.removesuffix("_paths") + "_study_id"] = study["study_id"]
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["subject_id", "visit_id", "visit_date", "modalities", "n_studies", "visit_span_days"]
        )
    output = pd.DataFrame(rows)
    return output.sort_values(["subject_id", "visit_date"]).reset_index(drop=True)
