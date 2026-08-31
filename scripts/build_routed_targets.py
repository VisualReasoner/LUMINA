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

from lumina.data.targets import RoutedTargetConfig, build_routed_targets
from lumina.adapters import load_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build subject-level routed benchmark targets for LUMINA tasks.")
    parser.add_argument("--visit-index-csv", type=Path, required=True, help="Visit index CSV.")
    parser.add_argument("--visit-labels-csv", type=Path, required=True, help="Visit label CSV.")
    parser.add_argument("--adapter-yaml", type=Path, required=True, help="Adapter defining labels and eligibility.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output routed target CSV.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional report JSON.")
    args = parser.parse_args()

    adapter = load_adapter(args.adapter_yaml)
    config = RoutedTargetConfig.from_adapter(adapter)

    routed, report = build_routed_targets(
        visit_index=pd.read_csv(args.visit_index_csv, low_memory=False),
        label_table=pd.read_csv(args.visit_labels_csv, low_memory=False),
        config=config,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    routed.to_csv(args.output_csv, index=False)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
