from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


def _string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list.")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} contains duplicate values.")
    return items


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_column: str
    labels: tuple[str, ...]
    label_aliases: Mapping[str, str] = field(default_factory=dict)
    target_policy: str = "latest_eligible_visit"
    target_filter_column: str | None = None
    target_filter_values: tuple[str, ...] = ()
    minimum_prefix_visits: int = 2
    required_target_modalities: tuple[str, ...] = ()
    require_any_target_modality: tuple[str, ...] = ()
    required_prior_modalities: tuple[str, ...] = ()
    inspection_rules: tuple[str, ...] = ()
    reconciliation_rules: tuple[str, ...] = ()
    belief_rubric: tuple[str, ...] = ()
    belief_dimensions: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSpec":
        labels = _string_list(payload.get("labels"), field_name="task.labels")
        if len(labels) < 2:
            raise ValueError("task.labels must contain at least two labels.")
        normalized_labels = [label.lower().replace("-", "_").replace(" ", "_") for label in labels]
        if len(normalized_labels) != len(set(normalized_labels)):
            raise ValueError("task.labels contains normalization collisions.")
        name = str(payload.get("name") or "").strip()
        label_column = str(payload.get("label_column") or "").strip()
        if not name or not label_column:
            raise ValueError("task.name and task.label_column are required.")
        aliases = {
            str(key).strip().lower().replace("-", "_").replace(" ", "_"): str(value).strip()
            for key, value in dict(payload.get("label_aliases") or {}).items()
        }
        invalid_aliases = sorted(set(aliases.values()) - set(labels))
        if invalid_aliases:
            raise ValueError(f"task.label_aliases maps to unknown labels: {invalid_aliases}")
        target_filter = payload.get("target_filter") or {}
        if not isinstance(target_filter, Mapping):
            raise ValueError("task.target_filter must be a mapping.")
        target_filter_column = str(target_filter.get("column") or "").strip() or None
        target_filter_values = _string_list(
            target_filter.get("values"), field_name="task.target_filter.values"
        )
        if bool(target_filter_column) != bool(target_filter_values):
            raise ValueError("task.target_filter requires both column and values.")
        belief_dimensions = {
            str(key).strip(): str(value).strip()
            for key, value in dict(payload.get("belief_dimensions") or {}).items()
            if str(key).strip() and str(value).strip()
        }
        return cls(
            name=name,
            label_column=label_column,
            labels=labels,
            label_aliases=aliases,
            target_policy=str(payload.get("target_policy") or "latest_eligible_visit"),
            target_filter_column=target_filter_column,
            target_filter_values=target_filter_values,
            minimum_prefix_visits=max(1, int(payload.get("minimum_prefix_visits", 2))),
            required_target_modalities=_string_list(
                payload.get("required_target_modalities"), field_name="task.required_target_modalities"
            ),
            require_any_target_modality=_string_list(
                payload.get("require_any_target_modality"), field_name="task.require_any_target_modality"
            ),
            required_prior_modalities=_string_list(
                payload.get("required_prior_modalities"), field_name="task.required_prior_modalities"
            ),
            inspection_rules=_string_list(payload.get("inspection_rules"), field_name="task.inspection_rules"),
            reconciliation_rules=_string_list(
                payload.get("reconciliation_rules"), field_name="task.reconciliation_rules"
            ),
            belief_rubric=_string_list(payload.get("belief_rubric"), field_name="task.belief_rubric"),
            belief_dimensions=belief_dimensions,
        )

    def normalize_label(self, value: object) -> str:
        text = str(value or "").strip()
        if text in self.labels:
            return text
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if normalized in self.label_aliases:
            return self.label_aliases[normalized]
        for label in self.labels:
            if normalized == label.lower().replace("-", "_").replace(" ", "_"):
                return label
        raise ValueError(f"Unsupported label for task {self.name}: {value!r}")


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    path_column: str
    role: str
    observation_questions: tuple[str, ...]
    comparison_questions: tuple[str, ...]
    reference_column: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModalitySpec":
        name = str(payload.get("name") or "").strip()
        path_column = str(payload.get("path_column") or "").strip()
        if not name or not path_column:
            raise ValueError("Each modality requires name and path_column.")
        reference_column = str(payload.get("reference_column") or "").strip() or None
        return cls(
            name=name,
            path_column=path_column,
            role=str(payload.get("role") or "visual evidence").strip(),
            observation_questions=_string_list(
                payload.get("observation_questions"), field_name=f"modalities.{name}.observation_questions"
            ),
            comparison_questions=_string_list(
                payload.get("comparison_questions"), field_name=f"modalities.{name}.comparison_questions"
            ),
            reference_column=reference_column,
        )


@dataclass(frozen=True)
class ReferencePolicy:
    enabled: bool = False
    match_primary: str = ""
    match_secondary: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ReferencePolicy":
        data = payload or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            match_primary=str(data.get("match_primary") or "").strip(),
            match_secondary=str(data.get("match_secondary") or "").strip(),
        )


@dataclass(frozen=True)
class MemoryPolicy:
    cross_subject_enabled: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "MemoryPolicy":
        data = payload or {}
        return cls(cross_subject_enabled=bool(data.get("cross_subject_enabled", True)))


@dataclass(frozen=True)
class AdapterSpec:
    adapter_name: str
    task: TaskSpec
    modalities: tuple[ModalitySpec, ...]
    reference_policy: ReferencePolicy = field(default_factory=ReferencePolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    context_columns: tuple[str, ...] = ()
    subject_id_aliases: tuple[str, ...] = ()
    context_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterSpec":
        adapter_name = str(payload.get("adapter_name") or "").strip()
        if not adapter_name:
            raise ValueError("adapter_name is required.")
        task = TaskSpec.from_dict(dict(payload.get("task") or {}))
        modality_payloads = payload.get("modalities") or []
        if not isinstance(modality_payloads, list) or not modality_payloads:
            raise ValueError("modalities must contain at least one modality specification.")
        modalities = tuple(ModalitySpec.from_dict(item) for item in modality_payloads)
        names = [item.name for item in modalities]
        columns = [item.path_column for item in modalities]
        if len(names) != len(set(names)) or len(columns) != len(set(columns)):
            raise ValueError("Modality names and path columns must be unique.")
        normalized_names = [name.lower().replace("_", " ").strip() for name in names]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("Modality names contain normalization collisions.")
        known_modalities = set(names)
        required = (
            set(task.required_target_modalities)
            | set(task.require_any_target_modality)
            | set(task.required_prior_modalities)
        )
        unknown = sorted(required - known_modalities)
        if unknown:
            raise ValueError(f"Task eligibility references unknown modalities: {unknown}")
        forbidden_context_tokens = (
            "subject_id",
            "patient_id",
            "label",
            "diagnosis",
            "future",
            "outcome",
            "progression",
        )
        context_columns = _string_list(payload.get("context_columns"), field_name="context_columns")
        unsafe_context = sorted(
            column
            for column in context_columns
            if any(token in column.lower() for token in forbidden_context_tokens)
        )
        if unsafe_context:
            raise ValueError(f"context_columns contains label-defining or identifying fields: {unsafe_context}")
        raw_context_aliases = payload.get("context_aliases") or {}
        if not isinstance(raw_context_aliases, Mapping):
            raise ValueError("context_aliases must be a mapping from context fields to source-column lists.")
        unknown_alias_fields = sorted(set(raw_context_aliases) - set(context_columns))
        if unknown_alias_fields:
            raise ValueError(f"context_aliases contains fields absent from context_columns: {unknown_alias_fields}")
        context_aliases = {
            str(field_name): _string_list(aliases, field_name=f"context_aliases.{field_name}")
            for field_name, aliases in raw_context_aliases.items()
        }
        reference_policy = ReferencePolicy.from_dict(payload.get("references"))
        if reference_policy.enabled:
            if not reference_policy.match_primary or not reference_policy.match_secondary:
                raise ValueError("Enabled reference policy requires match_primary and match_secondary.")
            if reference_policy.match_primary == reference_policy.match_secondary:
                raise ValueError("Reference match_primary and match_secondary must be different fields.")
            if not any(item.reference_column for item in modalities):
                raise ValueError("Enabled reference policy requires at least one modality reference_column.")
            missing_match_context = sorted(
                field_name
                for field_name in (reference_policy.match_primary, reference_policy.match_secondary)
                if field_name not in context_columns
            )
            if missing_match_context:
                raise ValueError(
                    f"Reference matching fields must be present in context_columns: {missing_match_context}"
                )
        return cls(
            adapter_name=adapter_name,
            task=task,
            modalities=modalities,
            reference_policy=reference_policy,
            memory_policy=MemoryPolicy.from_dict(payload.get("memory")),
            context_columns=context_columns,
            subject_id_aliases=_string_list(payload.get("subject_id_aliases"), field_name="subject_id_aliases"),
            context_aliases=context_aliases,
        )

    @property
    def modality_map(self) -> dict[str, ModalitySpec]:
        return {item.name: item for item in self.modalities}


def load_adapter(path: str | Path) -> AdapterSpec:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter YAML must contain one mapping: {source}")
    return AdapterSpec.from_dict(payload)
