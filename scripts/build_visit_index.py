from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.data.visit_index import VisitIndexConfig, build_visit_index, load_study_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multimodal visit index for LUMINA.")
    parser.add_argument("--study-catalog-csv", type=Path, required=True, help="CSV containing one row per study.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output visit index CSV.")
    parser.add_argument(
        "--within-visit-gap-days",
        type=int,
        default=0,
        help="Scans on the same date are grouped by default; set a positive window explicitly if required.",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optional summary JSON path.")
    args = parser.parse_args()

    studies = load_study_catalog(args.study_catalog_csv)
    visit_index = build_visit_index(studies, VisitIndexConfig(within_visit_gap_days=args.within_visit_gap_days))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    visit_index.to_csv(args.output_csv, index=False)
    report = {
        "visit_rows": int(len(visit_index)),
        "subjects": int(visit_index["subject_id"].nunique()) if not visit_index.empty else 0,
        "within_visit_gap_days": int(args.within_visit_gap_days),
        "modalities_seen": sorted(
            {
                col.removesuffix("_paths")
                for col in visit_index.columns
                if col.endswith("_paths")
            }
        ),
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report['visit_rows']} visit rows across {report['subjects']} subjects to {args.output_csv}")


if __name__ == "__main__":
    main()
