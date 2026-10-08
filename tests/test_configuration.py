from __future__ import annotations

from pathlib import Path

import pytest

from lumina.agent import ControllerConfig
from lumina.configuration import load_yaml
from lumina.memory import TrajectoryConfig


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "benchmark" / "default.yaml"


def test_default_configuration_enables_full_pipeline() -> None:
    settings = load_yaml(BASE)
    controller = ControllerConfig(**settings["controller"])
    assert controller.use_references
    assert controller.use_anchor_comparisons
    assert controller.use_cross_subject_memory
    assert controller.use_audit
    assert controller.repair_limit == 1

    trajectory = dict(settings["trajectory"])
    assert trajectory.pop("use_trajectory_memory") is True
    assert trajectory.pop("use_smc") is True
    assert TrajectoryConfig(**trajectory).max_active_events == 4
    assert settings["evaluation"]["require_existing_images"] is True


def test_load_yaml_preserves_user_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("evaluation:\n  bootstrap_samples: 50\n", encoding="utf-8")
    assert load_yaml(path) == {"evaluation": {"bootstrap_samples": 50}}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration must be a mapping"):
        load_yaml(path)
