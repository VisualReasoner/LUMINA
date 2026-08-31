from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.data.labels import LabelBuildConfig, build_label_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADNI benchmark labels from visit and clinical tables.")
    parser.add_argument("--visit-index-csv", type=Path, required=True, help="Visit index CSV.")
    parser.add_argument("--diagnosis-csv", type=Path, required=True, help="Clinical diagnosis table with PTID, EXAMDATE, DX.")
    parser.add_argument("--cdr-csv", type=Path, required=True, help="CDR table with PTID, VISDATE, CDGLOBAL.")
    parser.add_argument("--amyloid-csv", type=Path, required=True, help="Amyloid annotation table with PTID, SCANDATE, AMYLOID_STATUS.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output label CSV.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional report JSON.")
    parser.add_argument("--max-cdr-gap-days", type=int, default=180)
    parser.add_argument("--max-dx-gap-days", type=int, default=365)
    parser.add_argument("--progression-horizon-months", type=int, default=36)
    args = parser.parse_args()

    table, report = build_label_table(
        visit_index_csv=args.visit_index_csv,
        diagnosis_csv=args.diagnosis_csv,
        cdr_csv=args.cdr_csv,
        amyloid_csv=args.amyloid_csv,
        config=LabelBuildConfig(
            max_cdr_gap_days=args.max_cdr_gap_days,
            max_dx_gap_days=args.max_dx_gap_days,
            progression_horizon_months=args.progression_horizon_months,
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
