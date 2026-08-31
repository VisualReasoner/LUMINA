from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class ReferenceIndexConfig:
    modality_column: str = "modality"
    candidate_id_column: str = "candidate_id"
    image_path_column: str = "image_path"


def normalize_modality(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def load_subject_context(
    path: Path,
    *,
    context_columns: tuple[str, ...],
    subject_id_aliases: tuple[str, ...] = (),
    context_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Load one subject-level context row using columns declared by an adapter."""
    table = pd.read_csv(path, low_memory=False)
    subject_col = next(
        (column for column in ("subject_id", *subject_id_aliases) if column in table.columns),
        None,
    )
    if subject_col is None:
        raise ValueError("Subject context CSV is missing the configured subject identifier column.")
    aliases = context_aliases or {}
    output = pd.DataFrame({"subject_id": table[subject_col].astype(str)})
    for field in context_columns:
        candidates = (field, *aliases.get(field, ()))
        source = next((column for column in candidates if column in table.columns), None)
        if source is None:
            raise ValueError(f"Subject context CSV is missing adapter field {field!r}.")
        output[field] = table[source]
    return output.groupby("subject_id", as_index=False).first()


def load_reference_candidates(
    path: Path,
    config: ReferenceIndexConfig | None = None,
    *,
    modality_output_columns: Mapping[str, str] | None = None,
    matching_fields: tuple[str, ...] = (),
) -> dict[str, str]:
    cfg = config or ReferenceIndexConfig()
    table = pd.read_csv(path, low_memory=False)
    required = {
        cfg.modality_column,
        cfg.candidate_id_column,
        cfg.image_path_column,
        *matching_fields,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Reference candidate CSV missing required columns: {missing}")
    forbidden_tokens = ("label", "diagnosis", "future", "outcome", "target")
    forbidden_columns = [
        column for column in table.columns if any(token in str(column).lower() for token in forbidden_tokens)
    ]
    if forbidden_columns:
        raise ValueError(
            "Reference candidate CSV contains label-defining or outcome columns that are not allowed: "
            f"{sorted(forbidden_columns)}"
        )

    payloads: dict[str, str] = {}
    output_mapping = modality_output_columns or {}
    if not output_mapping:
        raise ValueError("modality_output_columns must be provided by an adapter.")
    for modality_key, output_column in output_mapping.items():
        subset = table[table[cfg.modality_column].map(normalize_modality) == modality_key].copy()
        records: list[dict[str, object]] = []
        for candidate_id, candidate_rows in subset.groupby(cfg.candidate_id_column, sort=False):
            image_paths = [
                str(value).strip()
                for value in candidate_rows[cfg.image_path_column].tolist()
                if pd.notna(value) and str(value).strip()
            ]
            if not image_paths:
                raise ValueError(f"Reference candidate {candidate_id!r} has no image paths.")
            record = {
                "candidate_id": candidate_id,
                "image_paths": image_paths,
            }
            for optional_col in dict.fromkeys(("subject_id", *matching_fields)):
                if optional_col not in candidate_rows.columns:
                    continue
                values = candidate_rows[optional_col].dropna().unique().tolist()
                if len(values) > 1:
                    raise ValueError(
                        f"Reference candidate {candidate_id!r} has inconsistent {optional_col!r} values."
                    )
                if values:
                    record[optional_col] = values[0]
                elif optional_col in matching_fields:
                    raise ValueError(
                        f"Reference candidate {candidate_id!r} is missing matching field {optional_col!r}."
                    )
            records.append(record)
        payloads[output_column] = json.dumps(records, separators=(",", ":"))
    return payloads


def attach_reference_metadata(
    *,
    routed_targets: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
    reference_payloads: dict[str, str] | None = None,
    context_columns: tuple[str, ...] = (),
    required_context_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    table = routed_targets.copy()
    table["subject_id"] = table["subject_id"].astype(str)

    if demographics is not None and not demographics.empty:
        demo = demographics.copy()
        demo["subject_id"] = demo["subject_id"].astype(str)
        demo = demo.groupby("subject_id", as_index=False).first().set_index("subject_id")
        for column in context_columns:
            if column not in demo.columns:
                continue
            incoming = table["subject_id"].map(demo[column])
            if column in table.columns:
                table[column] = table[column].combine_first(incoming)
            else:
                table[column] = incoming

    incomplete_context: dict[str, int] = {}
    for column in required_context_columns:
        if column not in table.columns:
            incomplete_context[column] = len(table)
            continue
        values = table[column]
        missing = values.isna() | values.astype(str).str.strip().eq("")
        if missing.any():
            incomplete_context[column] = int(missing.sum())
    if incomplete_context:
        details = ", ".join(f"{field} ({count} row(s))" for field, count in incomplete_context.items())
        raise ValueError(
            "Reference matching requires complete routed-target context. "
            f"Missing values: {details}. Provide a subject-context CSV or include these fields in routed targets."
        )

    payloads = reference_payloads or {}
    for output_column, payload in payloads.items():
        table[output_column] = payload

    report = {
        "rows": int(len(table)),
        "subjects": int(table["subject_id"].nunique()) if not table.empty else 0,
        "reference_columns": sorted(payloads),
        "matching_context_columns": list(required_context_columns),
        "has_demographics": demographics is not None and not demographics.empty,
    }
    return table, report
