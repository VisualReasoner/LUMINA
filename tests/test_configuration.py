from __future__ import annotations

from pathlib import Path

from lumina.configuration import load_experiment_settings


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "benchmark" / "default.yaml"
ABLATIONS = ROOT / "configs" / "ablations"


def _settings(name: str) -> dict:
    return load_experiment_settings(BASE, ABLATIONS / f"{name}.yaml")


def test_ablation_overlays_change_only_the_named_component() -> None:
    full = _settings("full")
    checks = {
        "no_anchor": ("controller", "use_anchor_comparisons", False),
        "no_trajectory": ("trajectory", "use_trajectory_memory", False),
        "no_cross_subject": ("controller", "use_cross_subject_memory", False),
        "no_smc": ("trajectory", "use_smc", False),
        "no_references": ("controller", "use_references", False),
    }
    for name, (section, key, expected) in checks.items():
        settings = _settings(name)
        assert settings[section][key] is expected
        for stable_section in ("controller", "trajectory", "evaluation"):
            expected_section = dict(full[stable_section])
            actual_section = dict(settings[stable_section])
            if stable_section == section:
                expected_section[key] = expected
            assert actual_section == expected_section


def test_verification_ablation_disables_only_bounded_repair_and_audit() -> None:
    full = _settings("full")
    settings = _settings("no_verification")
    expected_controller = dict(full["controller"])
    expected_controller.update({"repair_limit": 0, "use_audit": False})
    assert settings["controller"] == expected_controller
    assert settings["trajectory"] == full["trajectory"]
    assert settings["evaluation"] == full["evaluation"]
