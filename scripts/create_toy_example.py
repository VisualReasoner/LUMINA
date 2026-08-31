from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


AD_BELIEF_FIELDS = (
    "mri_current_severity",
    "mri_longitudinal_change",
    "mri_tempo",
    "mri_anchor_quality",
    "amyloid_status",
    "tau_status",
    "tau_pattern_severity",
    "pet_longitudinal_change",
    "pathology_state",
    "stage_support",
    "longitudinal_role",
    "longitudinal_pattern",
    "tempo_assessment",
    "anchor_quality",
    "overall_evidence_quality",
)


def _task_support(value: str) -> dict[str, str]:
    return {field: value for field in AD_BELIEF_FIELDS}


def _write_image(path: Path, *, inner_radius: int) -> None:
    image = Image.new("L", (64, 64), color=18)
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), fill=150, outline=220, width=2)
    center = 32
    draw.ellipse(
        (center - inner_radius, center - inner_radius, center + inner_radius, center + inner_radius),
        fill=35,
    )
    image.convert("RGB").save(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_example(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    visit_1 = root / "toy_visit_1.png"
    visit_2 = root / "toy_visit_2.png"
    _write_image(visit_1, inner_radius=5)
    _write_image(visit_2, inner_radius=10)

    visit_index = root / "visit_index.csv"
    _write_csv(
        visit_index,
        ["subject_id", "visit_id", "visit_date", "mri_paths"],
        [
            {"subject_id": "toy_subject", "visit_id": "v1", "visit_date": "2020-01-01", "mri_paths": visit_1},
            {"subject_id": "toy_subject", "visit_id": "v2", "visit_date": "2021-01-01", "mri_paths": visit_2},
        ],
    )
    routed_targets = root / "routed_targets.csv"
    _write_csv(
        routed_targets,
        ["subject_id", "target_visit_id", "routed_label", "age", "sex", "race"],
        [
            {
                "subject_id": "toy_subject",
                "target_visit_id": "v2",
                "routed_label": "Symptomatic_AD",
                "age": 72,
                "sex": "F",
                "race": "synthetic",
            }
        ],
    )

    replay = root / "replay_responses.json"
    replay.write_text(
        json.dumps(
            {
                "inspect": [
                    {"selected_modalities": ["MRI"], "visit_questions": {"MRI": ["Inspect structure"]}},
                    {"selected_modalities": ["MRI"], "visit_questions": {"MRI": ["Inspect structure"]}},
                ],
                "observe": [
                    {
                        "modality_observations": {
                            "MRI": {
                                "present": True,
                                "findings": ["synthetic mild structural pattern"],
                                "severity": 1,
                                "confidence": 0.7,
                            }
                        },
                        "summary": "Synthetic current-visit evidence.",
                    },
                    {
                        "modality_observations": {
                            "MRI": {
                                "present": True,
                                "findings": ["synthetic stronger structural pattern"],
                                "severity": 2,
                                "confidence": 0.8,
                            }
                        },
                        "summary": "Synthetic current-visit evidence with greater severity.",
                    },
                ],
                "compare_anchor": [
                    {
                        "direction": "moderate_increase",
                        "magnitude": "moderate",
                        "change_grade": 2,
                        "localized_evidence": ["larger synthetic central region"],
                        "summary": "Synthetic same-modality evidence increased.",
                        "confidence": 0.8,
                        "comparison_quality": "strong",
                    }
                ],
                "reconcile": [
                    {"agreements": [], "conflicts": [], "summary": "Current evidence only."},
                    {
                        "agreements": ["same-modality increase"],
                        "conflicts": [],
                        "summary": "Synthetic longitudinal change is supported.",
                    },
                ],
                "final_belief": [
                    {
                        "leading_label": "CU_nonAD",
                        "second_best_label": "Preclinical_AD",
                        "supporting_evidence": ["limited first-visit evidence"],
                        "task_support": _task_support("limited initial evidence"),
                        "summary": "Limited initial evidence.",
                    },
                    {
                        "leading_label": "Symptomatic_AD",
                        "second_best_label": "Preclinical_AD",
                        "supporting_evidence": ["synthetic structural increase"],
                        "longitudinal_evidence": ["current MRI differs from prior MRI"],
                        "task_support": _task_support("supported target evidence"),
                        "summary": "Synthetic longitudinal evidence supports the routed example label.",
                    },
                ],
                "repair_belief": [
                    {
                        "leading_label": "Symptomatic_AD",
                        "second_best_label": "Preclinical_AD",
                        "supporting_evidence": ["synthetic structural increase"],
                        "longitudinal_evidence": ["current MRI differs from prior MRI"],
                        "task_support": _task_support("supported target evidence without trajectory memory"),
                        "summary": "Synthetic target belief repaired from the visible anchor comparison.",
                    }
                ],
                "audit": [
                    {"decisive_evidence": ["current synthetic image"], "summary": "Grounded."},
                    {
                        "decisive_evidence": ["synthetic same-modality increase"],
                        "summary": "Grounded and longitudinal.",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"visit_index": visit_index, "routed_targets": routed_targets, "replay": replay}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a data-free replay example for the LUMINA controller.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = create_example(args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
