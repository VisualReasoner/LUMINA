from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Iterable, Mapping

import pandas as pd

from lumina.adapters.specs import AdapterSpec


@dataclass(frozen=True)
class RoutedTargetConfig:
    task_name: str
    label_column: str
    allowed_labels: tuple[str, ...]
    label_aliases: Mapping[str, str] = field(default_factory=dict)
    target_policy: str = "latest_eligible_visit"
    target_filter_column: str | None = None
    target_filter_values: tuple[str, ...] = ()
    minimum_prefix_visits: int = 2
    required_target_modalities: tuple[str, ...] = ()
    require_any_target_modality: tuple[str, ...] = ()
    required_prior_modalities: tuple[str, ...] = ()
    modality_path_columns: tuple[tuple[str, str], ...] = ()
    context_columns: tuple[str, ...] = ()

    @classmethod
    def from_adapter(cls, adapter: AdapterSpec) -> "RoutedTargetConfig":
        task = adapter.task
        return cls(
            task_name=task.name,
            label_column=task.label_column,
            allowed_labels=task.labels,
            label_aliases=task.label_aliases,
            target_policy=task.target_policy,
            target_filter_column=task.target_filter_column,
            target_filter_values=task.target_filter_values,
            minimum_prefix_visits=task.minimum_prefix_visits,
            required_target_modalities=task.required_target_modalities,
            require_any_target_modality=task.require_any_target_modality,
            required_prior_modalities=task.required_prior_modalities,
            modality_path_columns=tuple((item.name, item.path_column) for item in adapter.modalities),
            context_columns=adapter.context_columns,
        )

    def normalize_label(self, value: object) -> str | None:
        text = str(value or "").strip()
        if text in self.allowed_labels:
            return text
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        alias = self.label_aliases.get(normalized)
        if alias in self.allowed_labels:
            return alias
        for label in self.allowed_labels:
            if normalized == label.lower().replace("-", "_").replace(" ", "_"):
                return label
        return None


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _has_paths(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _visit_modalities(row: pd.Series, modality_path_columns: tuple[tuple[str, str], ...]) -> list[str]:
    return [modality for modality, column in modality_path_columns if _has_paths(row.get(column))]


def _target_candidates(subject_visits: pd.DataFrame, config: RoutedTargetConfig) -> pd.DataFrame:
    eligible = subject_visits[subject_visits["_routed_label"].notna()].copy()
    if config.target_filter_column:
        if config.target_filter_column not in eligible.columns:
            raise ValueError(
                f"Target filter column {config.target_filter_column!r} is absent from visit and label tables."
            )
        allowed = {value.strip().lower() for value in config.target_filter_values}
        eligible = eligible[
            eligible[config.target_filter_column].map(lambda value: str(value or "").strip().lower() in allowed)
        ]
    if config.target_policy == "latest_eligible_visit":
        return eligible.sort_values(["visit_date", "visit_id"], ascending=[False, False])
    if config.target_policy == "earliest_eligible_visit":
        return eligible.sort_values(["visit_date", "visit_id"], ascending=[True, True])
    raise ValueError(f"Unsupported target_policy: {config.target_policy!r}")


def _select_target(subject_visits: pd.DataFrame, config: RoutedTargetConfig) -> pd.Series | None:
    candidates = _target_candidates(subject_visits, config)
    if candidates.empty:
        return None
    ordered = subject_visits.sort_values(["visit_date", "visit_id"]).reset_index(drop=True)
    required = set(config.required_target_modalities)
    require_any = set(config.require_any_target_modality)
    for _, candidate in candidates.iterrows():
        matches = ordered.index[ordered["visit_id"].astype(str) == str(candidate["visit_id"])].tolist()
        if not matches:
            continue
        target_index = matches[-1]
        if target_index + 1 < max(1, config.minimum_prefix_visits):
            continue
        modalities = set(_visit_modalities(candidate, config.modality_path_columns))
        if not required.issubset(modalities):
            continue
        if require_any and not modalities.intersection(require_any):
            continue
        if config.required_prior_modalities:
            prior_modalities = {
                modality
                for _, prior_row in ordered.iloc[:target_index].iterrows()
                for modality in _visit_modalities(prior_row, config.modality_path_columns)
            }
            if not set(config.required_prior_modalities).issubset(prior_modalities):
                continue
        return candidate
    return None


def build_routed_targets(
    *,
    visit_index: pd.DataFrame,
    label_table: pd.DataFrame,
    config: RoutedTargetConfig,
) -> tuple[pd.DataFrame, dict]:
    required_visit_cols = {"subject_id", "visit_id", "visit_date"}
    required_label_cols = {"subject_id", "visit_id", config.label_column}
    missing_visit = sorted(required_visit_cols - set(visit_index.columns))
    missing_label = sorted(required_label_cols - set(label_table.columns))
    if missing_visit:
        raise ValueError(f"Visit index missing required columns: {missing_visit}")
    if missing_label:
        raise ValueError(f"Label table missing required columns: {missing_label}")
    missing_paths = sorted(column for _, column in config.modality_path_columns if column not in visit_index.columns)
    if len(missing_paths) == len(config.modality_path_columns):
        raise ValueError(f"Visit index contains none of the adapter modality columns: {missing_paths}")

    visits = visit_index.copy()
    labels = label_table.copy()
    for table in (visits, labels):
        table["subject_id"] = table["subject_id"].astype(str)
        table["visit_id"] = table["visit_id"].astype(str)
    if visits.duplicated(["subject_id", "visit_id"]).any():
        raise ValueError("Visit index must contain one row per subject_id and visit_id.")
    if labels.duplicated(["subject_id", "visit_id"]).any():
        raise ValueError("Label table must contain one row per subject_id and visit_id.")
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors="coerce")
    if visits["visit_date"].isna().any():
        raise ValueError("Visit index contains missing or invalid visit_date values.")

    extra_label_columns = [config.label_column]
    optional_columns = [config.target_filter_column, *config.context_columns]
    for column in optional_columns:
        if column and column in labels.columns and column not in visits.columns and column not in extra_label_columns:
            extra_label_columns.append(column)
    merged = visits.merge(
        labels[["subject_id", "visit_id", *extra_label_columns]],
        on=["subject_id", "visit_id"],
        how="left",
    )
    merged["_routed_label"] = merged[config.label_column].map(config.normalize_label)

    output_rows: list[dict[str, object]] = []
    label_counts = {label: 0 for label in config.allowed_labels}
    for subject_id, group in merged.groupby("subject_id", sort=False):
        subject_visits = group[
            group.apply(lambda row: bool(_visit_modalities(row, config.modality_path_columns)), axis=1)
        ].copy()
        subject_visits = subject_visits.sort_values(["visit_date", "visit_id"]).reset_index(drop=True)
        if subject_visits.empty:
            continue
        target_row = _select_target(subject_visits, config)
        if target_row is None:
            continue
        target_visit_id = str(target_row["visit_id"])
        target_index = subject_visits.index[subject_visits["visit_id"].astype(str) == target_visit_id].tolist()[-1]
        prefix = subject_visits.iloc[: target_index + 1].copy()
        target_label = str(target_row["_routed_label"])
        label_counts[target_label] += 1

        prefix_modalities = {
            modality
            for _, prefix_row in prefix.iterrows()
            for modality in _visit_modalities(prefix_row, config.modality_path_columns)
        }
        output = {
            "subject_id": str(subject_id),
            "task_name": config.task_name,
            "label_column": config.label_column,
            "target_visit_id": target_visit_id,
            "target_visit_date": pd.Timestamp(target_row["visit_date"]),
            "routed_label": target_label,
            "prefix_n_visits": int(len(prefix)),
            "prefix_visit_ids": _json_list(prefix["visit_id"].astype(str).tolist()),
            "prefix_modalities_seen": _json_list(sorted(prefix_modalities)),
            "target_modalities": _json_list(sorted(_visit_modalities(target_row, config.modality_path_columns))),
        }
        for column in config.context_columns:
            value = target_row.get(column)
            if value is not None and bool(pd.notna(value)):
                output[column] = value
        if config.target_filter_column:
            output[config.target_filter_column] = target_row.get(config.target_filter_column)
        output_rows.append(output)

    base_columns = [
        "subject_id",
        "task_name",
        "label_column",
        "target_visit_id",
        "target_visit_date",
        "routed_label",
        "prefix_n_visits",
        "prefix_visit_ids",
        "prefix_modalities_seen",
        "target_modalities",
    ]
    routed = pd.DataFrame(output_rows) if output_rows else pd.DataFrame(columns=base_columns)
    if not routed.empty:
        routed = routed.sort_values(["target_visit_date", "subject_id"]).reset_index(drop=True)
    report = {
        "task_name": config.task_name,
        "label_column": config.label_column,
        "allowed_labels": list(config.allowed_labels),
        "routed_targets": int(len(routed)),
        "subjects": int(routed["subject_id"].nunique()) if not routed.empty else 0,
        "routed_label_counts": label_counts,
        "target_policy": config.target_policy,
        "minimum_prefix_visits": int(config.minimum_prefix_visits),
        "required_target_modalities": list(config.required_target_modalities),
        "require_any_target_modality": list(config.require_any_target_modality),
        "required_prior_modalities": list(config.required_prior_modalities),
    }
    return routed, report
