from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_toy_replay_runs_end_to_end_without_a_model(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "toy"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "create_toy_example.py"), "--output-dir", str(fixture_dir)],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output_dir = fixture_dir / "results"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmark.py"),
            "--visit-index-csv",
            str(fixture_dir / "visit_index.csv"),
            "--routed-targets-csv",
            str(fixture_dir / "routed_targets.csv"),
            "--adapter-yaml",
            str(ROOT / "configs" / "adapters" / "adni_ad_continuum.yaml"),
            "--backend",
            "replay",
            "--replay-json",
            str(fixture_dir / "replay_responses.json"),
            "--output-dir",
            str(output_dir),
            "--fail-fast",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed"] == 1
    assert metrics["errors"] == 0
    assert metrics["ablation"] == "full"
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "traces.jsonl").is_file()
