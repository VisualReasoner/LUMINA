from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.data.preprocessing import ViewConfig, convert_dicom_series, extract_center_views, preprocess_t1_mri


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the common three-view 2.5D study representation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dicom-dir", type=Path)
    source.add_argument("--nifti", type=Path)
    parser.add_argument("--mode", choices=["native", "t1_mri_mni"], default="native")
    parser.add_argument("--mni-template", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", type=str, required=True)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument(
        "--volume-index",
        type=int,
        default=None,
        help="Required for 4D inputs; selects the volume used for the three-view representation.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nifti = args.nifti
    if args.dicom_dir is not None:
        nifti = convert_dicom_series(args.dicom_dir, args.output_dir / "nifti", stem=args.stem)
    if args.mode == "t1_mri_mni":
        if args.mni_template is None:
            raise ValueError("--mni-template is required for t1_mri_mni mode.")
        nifti = preprocess_t1_mri(
            nifti,
            args.output_dir / "mri_work",
            mni_template=args.mni_template,
            prefix=args.stem,
        )
    outputs = extract_center_views(
        nifti,
        args.output_dir / "views",
        stem=args.stem,
        config=ViewConfig(output_size=args.output_size, volume_index=args.volume_index),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
