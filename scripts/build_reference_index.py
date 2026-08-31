from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from lumina.adapters import load_adapter
from lumina.data.references import (
    ReferenceIndexConfig,
    attach_reference_metadata,
    load_reference_candidates,
    load_subject_context,
    normalize_modality,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach shared-reference metadata to routed LUMINA targets.")
    parser.add_argument("--routed-targets-csv", type=Path, required=True, help="Routed target CSV.")
    parser.add_argument("--reference-candidates-csv", type=Path, required=True, help="Reference candidate CSV.")
    parser.add_argument("--adapter-yaml", type=Path, required=True, help="Adapter defining reference columns.")
    parser.add_argument("--demographics-csv", type=Path, default=None, help="Optional demographics CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output reference-enabled routed target CSV.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional report JSON.")
    args = parser.parse_args()

    adapter = load_adapter(args.adapter_yaml)
    modality_outputs = {
        normalize_modality(item.name): item.reference_column
        for item in adapter.modalities
        if item.reference_column
    }
    if not adapter.reference_policy.enabled or not modality_outputs:
        raise ValueError("The selected adapter does not define an enabled reference-comparator interface.")
    demographics = (
        load_subject_context(
            args.demographics_csv,
            context_columns=adapter.context_columns,
            subject_id_aliases=adapter.subject_id_aliases,
            context_aliases=adapter.context_aliases,
        )
        if args.demographics_csv is not None
        else None
    )
    payloads = load_reference_candidates(
        args.reference_candidates_csv,
        config=ReferenceIndexConfig(),
        modality_output_columns=modality_outputs,
        matching_fields=(
            adapter.reference_policy.match_primary,
            adapter.reference_policy.match_secondary,
        ),
    )
    table, report = attach_reference_metadata(
        routed_targets=pd.read_csv(args.routed_targets_csv, low_memory=False),
        demographics=demographics,
        reference_payloads=payloads,
        context_columns=adapter.context_columns,
        required_context_columns=(
            adapter.reference_policy.match_primary,
            adapter.reference_policy.match_secondary,
        ),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
