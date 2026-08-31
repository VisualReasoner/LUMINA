from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


COMPONENT_VARIANTS = (
    "full",
    "no_anchor",
    "no_trajectory",
    "no_cross_subject",
    "no_smc",
    "no_verification",
)
AVAILABLE_VARIANTS = (*COMPONENT_VARIANTS, "no_references")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LUMINA component ablations with one command.")
    parser.add_argument("--visit-index-csv", type=Path, required=True)
    parser.add_argument("--routed-targets-csv", type=Path, required=True)
    parser.add_argument("--adapter-yaml", type=Path, required=True)
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cross-subject-bank", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=AVAILABLE_VARIANTS,
        default=list(COMPONENT_VARIANTS),
        help="Run component ablations by default; no_references is a separate protocol control.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args, passthrough = parser.parse_known_args()

    runner = Path(__file__).with_name("run_benchmark.py")
    for variant in args.variants:
        command = [
            sys.executable,
            str(runner),
            "--visit-index-csv",
            str(args.visit_index_csv),
            "--routed-targets-csv",
            str(args.routed_targets_csv),
            "--adapter-yaml",
            str(args.adapter_yaml),
            "--backend",
            args.backend,
            "--output-dir",
            str(args.output_root / variant),
            "--ablation",
            variant,
        ]
        if args.model:
            command.extend(["--model", args.model])
        if args.cross_subject_bank:
            command.extend(["--cross-subject-bank", str(args.cross_subject_bank)])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        command.extend(passthrough)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
