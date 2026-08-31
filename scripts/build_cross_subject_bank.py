from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.adapters import load_adapter
from lumina.memory.cross_subject import CrossSubjectBank, build_entry
from lumina.schemas.states import (
    ActionTraceEntry,
    CurrentVisitEvidence,
    LocalComparison,
    ModalityObservation,
    ReconciledEvidence,
    TrajectoryState,
)


def _load_subject_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _current(payload: dict) -> CurrentVisitEvidence:
    raw = payload.get("current_evidence") or {}
    observations = {
        str(modality): ModalityObservation.from_dict(str(modality), value)
        for modality, value in dict(raw.get("modality_observations") or {}).items()
        if isinstance(value, dict)
    }
    return CurrentVisitEvidence(
        visit_id=str(raw.get("visit_id") or ""),
        modality_observations=observations,
        summary=str(raw.get("summary") or ""),
        multimodal_consistency=str(raw.get("multimodal_consistency") or "unknown"),
        uncertainties=[str(item) for item in raw.get("uncertainties") or []],
    )


def _comparison(modality: str, payload: dict) -> LocalComparison:
    allowed = {
        "current_evidence",
        "anchor_available",
        "anchor_visit_id",
        "anchor_date",
        "gap_days",
        "direction",
        "magnitude",
        "change_grade",
        "localized_evidence",
        "summary",
        "confidence",
        "uncertainty",
        "comparison_quality",
    }
    values = {key: value for key, value in payload.items() if key in allowed}
    return LocalComparison(modality=modality, **values)


def _trajectory(payload: dict) -> TrajectoryState:
    return TrajectoryState.from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen, label-free cross-subject bank from reserved development traces."
    )
    parser.add_argument("--traces-jsonl", type=Path, required=True)
    parser.add_argument(
        "--development-subjects",
        type=Path,
        required=True,
        help="One reserved development subject ID per line. Scored evaluation subjects must not be listed.",
    )
    parser.add_argument("--adapter-yaml", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    development_ids = _load_subject_ids(args.development_subjects)
    adapter = load_adapter(args.adapter_yaml)
    forbidden_terms = set(adapter.task.labels)
    forbidden_terms.update(label.replace("_", " ") for label in adapter.task.labels)
    forbidden_terms.update(adapter.task.label_aliases)
    forbidden_terms.update(alias.replace("_", " ") for alias in adapter.task.label_aliases)
    entries = []
    for line in args.traces_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        subject = json.loads(line)
        subject_id = str(subject.get("subject_id") or "")
        if subject_id not in development_ids:
            continue
        visits = subject.get("visits") or []
        if not visits:
            continue
        latest = visits[-1]
        current = _current(latest)
        memory = {
            str(modality): _comparison(str(modality), value)
            for modality, value in dict(latest.get("visit_memory") or {}).items()
            if isinstance(value, dict)
        }
        reconciled_payload = dict(latest.get("reconciled_evidence") or {})
        reconciled = ReconciledEvidence.from_dict(reconciled_payload)
        trace = [
            ActionTraceEntry(**item)
            for item in latest.get("action_trace") or []
            if isinstance(item, dict)
        ]
        visit = dict(latest.get("visit") or {})
        entries.append(
            build_entry(
                entry_id=f"{subject_id}:{visit.get('visit_id', 'target')}",
                subject_id=subject_id,
                current=current,
                visit_memory=memory,
                trajectory=_trajectory(dict(latest.get("trajectory_before") or {})),
                reconciled=reconciled,
                action_trace=trace,
                adapter_name=adapter.adapter_name,
                forbidden_terms=forbidden_terms,
            )
        )
    bank = CrossSubjectBank(entries=entries)
    bank.save_jsonl(args.output_jsonl)
    print(json.dumps({"entries": len(entries), "reserved_subjects": len(development_ids)}, indent=2))


if __name__ == "__main__":
    main()
