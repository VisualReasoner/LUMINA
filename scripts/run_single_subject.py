from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumina.adapters import load_adapter
from lumina.agent import ControllerConfig, EvidenceController, LUMINARuntime
from lumina.data.io import build_subject_prefix
from lumina.memory import CrossSubjectBank
from lumina.models.factory import build_model_client
from lumina.schemas.states import to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LUMINA for one routed subject target.")
    parser.add_argument("--visit-index-csv", type=Path, required=True)
    parser.add_argument("--routed-targets-csv", type=Path, required=True)
    parser.add_argument("--subject-id", type=str, required=True)
    parser.add_argument("--adapter-yaml", type=Path, required=True)
    parser.add_argument("--backend", choices=["replay", "openai", "openai_compatible", "transformers", "hf", "local"], required=True)
    parser.add_argument("--model", type=str, default=None)
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
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    adapter = load_adapter(args.adapter_yaml)
    visits = pd.read_csv(args.visit_index_csv, low_memory=False)
    targets = pd.read_csv(args.routed_targets_csv, low_memory=False)
    matches = targets[targets["subject_id"].astype(str) == args.subject_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one routed target for {args.subject_id}; found {len(matches)}.")
    prefix = build_subject_prefix(
        visit_index=visits,
        routed_target=matches.iloc[0].to_dict(),
        adapter=adapter,
    )
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
    runtime = LUMINARuntime(
        controller=EvidenceController(
            adapter=adapter,
            model=model,
            config=ControllerConfig(use_cross_subject_memory=bank is not None),
            cross_subject_bank=bank,
        )
    )
    result = runtime.run_subject(prefix)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(to_dict(result), indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"subject_id": result.subject_id, "prediction": result.prediction, "calls": result.call_count}, indent=2))


if __name__ == "__main__":
    main()
