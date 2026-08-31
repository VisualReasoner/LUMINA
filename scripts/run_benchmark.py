from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.adapters import load_adapter
from lumina.agent import ControllerConfig, EvidenceController, LUMINARuntime
from lumina.configuration import load_experiment_settings
from lumina.data.io import load_tables
from lumina.eval import BenchmarkConfig, BenchmarkRunner
from lumina.memory import CrossSubjectBank, TrajectoryConfig
from lumina.models.factory import build_model_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LUMINA on routed longitudinal subject targets.")
    parser.add_argument("--visit-index-csv", type=Path, required=True)
    parser.add_argument("--routed-targets-csv", type=Path, required=True)
    parser.add_argument("--adapter-yaml", type=Path, required=True)
    parser.add_argument("--benchmark-yaml", type=Path, default=ROOT / "configs" / "benchmark" / "default.yaml")
    parser.add_argument("--backend", choices=["replay", "openai", "openai_compatible", "transformers", "hf", "local"], required=True)
    parser.add_argument("--model", type=str, default=None, help="API model name or local Hugging Face path.")
    parser.add_argument("--replay-json", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable a processor-supported thinking template; omit to use the model default.",
    )
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--cross-subject-bank", type=Path, default=None)
    parser.add_argument(
        "--ablation",
        choices=["full", "no_anchor", "no_trajectory", "no_cross_subject", "no_smc", "no_verification", "no_references"],
        default="full",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    ablation_yaml = ROOT / "configs" / "ablations" / f"{args.ablation}.yaml"
    if not ablation_yaml.is_file():
        raise FileNotFoundError(f"Ablation configuration not found: {ablation_yaml}")
    settings = load_experiment_settings(args.benchmark_yaml, ablation_yaml)
    controller_settings = dict(settings.get("controller") or {})
    trajectory_settings = dict(settings.get("trajectory") or {})
    evaluation_settings = dict(settings.get("evaluation") or {})
    adapter = load_adapter(args.adapter_yaml)
    model = build_model_client(
        backend=args.backend,
        model=args.model,
        replay_json=args.replay_json,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        device_map=args.device_map,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        enable_thinking=args.enable_thinking,
    )
    bank = CrossSubjectBank.load_jsonl(args.cross_subject_bank) if args.cross_subject_bank else None
    controller = EvidenceController(
        adapter=adapter,
        model=model,
        config=ControllerConfig(**controller_settings),
        cross_subject_bank=bank,
    )
    use_trajectory_memory = bool(trajectory_settings.pop("use_trajectory_memory", True))
    use_smc = bool(trajectory_settings.pop("use_smc", True))
    runtime = LUMINARuntime(
        controller=controller,
        trajectory_config=TrajectoryConfig(**trajectory_settings),
        use_trajectory_memory=use_trajectory_memory,
        use_smc=use_smc,
    )
    visits, targets = load_tables(args.visit_index_csv, args.routed_targets_csv)
    runner = BenchmarkRunner(
        adapter=adapter,
        runtime=runtime,
        visit_index=visits,
        routed_targets=targets,
        output_dir=args.output_dir,
        config=BenchmarkConfig(
            require_existing_images=bool(evaluation_settings.get("require_existing_images", True)),
            bootstrap_samples=int(evaluation_settings.get("bootstrap_samples", 1000)),
            bootstrap_seed=int(evaluation_settings.get("bootstrap_seed", 17)),
            fail_fast=args.fail_fast,
        ),
    )
    summary = runner.run(limit=args.limit)
    summary["ablation"] = args.ablation
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
