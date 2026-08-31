from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from lumina.adapters.specs import AdapterSpec
from lumina.agent import ControllerConfig, EvidenceController, LUMINARuntime
from lumina.eval import BenchmarkConfig, BenchmarkRunner
from lumina.models import ReplayModelClient


def test_benchmark_exports_predictions_traces_and_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "gray").save(image_path)
    adapter = AdapterSpec.from_dict(
        {
            "adapter_name": "smoke",
            "context_columns": [],
            "task": {
                "name": "smoke_task",
                "label_column": "label",
                "labels": ["A", "B"],
                "minimum_prefix_visits": 1,
                "belief_rubric": ["Use visible evidence."],
            },
            "modalities": [
                {
                    "name": "Image",
                    "path_column": "image_paths",
                    "observation_questions": ["What is visible?"],
                    "comparison_questions": ["What changed?"],
                }
            ],
        }
    )
    model = ReplayModelClient(
        {
            "inspect": [{"selected_modalities": ["Image"]}],
            "observe": [
                {
                    "modality_observations": {"Image": {"present": True, "findings": ["finding"]}},
                    "summary": "visible finding",
                }
            ],
            "reconcile": [{"agreements": ["finding"], "summary": "consistent evidence"}],
            "final_belief": [
                {
                    "leading_label": "B",
                    "second_best_label": "A",
                    "supporting_evidence": ["finding"],
                    "summary": "B supported",
                }
            ],
            "audit": [{"decisive_evidence": ["finding"], "summary": "grounded"}],
        }
    )
    runtime = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_references=False, use_cross_subject_memory=False),
        )
    )
    output = tmp_path / "results"
    runner = BenchmarkRunner(
        adapter=adapter,
        runtime=runtime,
        visit_index=pd.DataFrame(
            [
                {
                    "subject_id": "s1",
                    "visit_id": "v1",
                    "visit_date": "2020-01-01",
                    "image_paths": str(image_path),
                }
            ]
        ),
        routed_targets=pd.DataFrame(
            [{"subject_id": "s1", "target_visit_id": "v1", "routed_label": "B"}]
        ),
        output_dir=output,
        config=BenchmarkConfig(bootstrap_samples=10),
    )
    summary = runner.run()
    assert summary["balanced_accuracy"] == 0.5
    assert (output / "predictions.csv").is_file()
    assert (output / "traces.jsonl").is_file()
    assert (output / "metrics.json").is_file()
