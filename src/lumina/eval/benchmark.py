from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from lumina.adapters.specs import AdapterSpec
from lumina.agent.runtime import LUMINARuntime
from lumina.data.io import build_subject_prefix
from lumina.eval.metrics import bootstrap_intervals, classification_metrics
from lumina.schemas.states import to_dict
from lumina.schemas.states import TrajectoryState


@dataclass(frozen=True)
class BenchmarkConfig:
    require_existing_images: bool = True
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 17
    fail_fast: bool = False


class BenchmarkRunner:
    def __init__(
        self,
        *,
        adapter: AdapterSpec,
        runtime: LUMINARuntime,
        visit_index: pd.DataFrame,
        routed_targets: pd.DataFrame,
        output_dir: str | Path,
        config: BenchmarkConfig | None = None,
    ):
        self.adapter = adapter
        self.runtime = runtime
        self.visit_index = visit_index.copy()
        self.routed_targets = routed_targets.copy()
        self.output_dir = Path(output_dir)
        self.config = config or BenchmarkConfig()
        runtime_adapter = self.runtime.controller.adapter.adapter_name
        if runtime_adapter != self.adapter.adapter_name:
            raise ValueError(
                f"Benchmark/runtime adapter mismatch: {self.adapter.adapter_name!r} != {runtime_adapter!r}."
            )
        if self.routed_targets.duplicated(["subject_id"]).any():
            raise ValueError("A subject-level benchmark run requires one routed target per subject.")
        if "task_name" in self.routed_targets.columns:
            observed_tasks = set(self.routed_targets["task_name"].dropna().astype(str))
            if observed_tasks and observed_tasks != {self.adapter.task.name}:
                raise ValueError(
                    f"Routed target task mismatch: adapter={self.adapter.task.name!r}, table={sorted(observed_tasks)}."
                )
        for value in self.routed_targets["routed_label"].tolist():
            self.adapter.task.normalize_label(value)
        bank = self.runtime.controller.cross_subject_bank
        if bank is not None and self.runtime.controller.config.use_cross_subject_memory:
            scored_subjects = set(self.routed_targets["subject_id"].astype(str))
            bank_subjects = {entry.subject_id for entry in bank.entries}
            overlap = sorted(scored_subjects & bank_subjects)
            if overlap:
                preview = overlap[:5]
                raise ValueError(
                    "Cross-subject bank is not subject-disjoint from scored targets; "
                    f"overlap includes {preview}."
                )

    def run(self, *, limit: int | None = None) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.csv"
        traces_path = self.output_dir / "traces.jsonl"
        errors_path = self.output_dir / "errors.jsonl"
        predictions: list[dict[str, object]] = []
        targets = self.routed_targets
        if limit is not None:
            targets = targets.head(max(0, limit))
        with traces_path.open("w", encoding="utf-8") as trace_stream, errors_path.open("w", encoding="utf-8") as error_stream:
            for row in targets.to_dict(orient="records"):
                subject_id = str(row["subject_id"])
                started = perf_counter()
                try:
                    visits = build_subject_prefix(
                        visit_index=self.visit_index,
                        routed_target=row,
                        adapter=self.adapter,
                        require_existing_images=self.config.require_existing_images,
                    )
                    prior_payload = row.get("prior_trajectory_state")
                    prior_trajectory = None
                    if isinstance(prior_payload, dict):
                        prior_trajectory = TrajectoryState.from_dict(prior_payload)
                    elif isinstance(prior_payload, str) and prior_payload.strip():
                        parsed = json.loads(prior_payload)
                        if not isinstance(parsed, dict):
                            raise ValueError("prior_trajectory_state must encode one JSON object.")
                        prior_trajectory = TrajectoryState.from_dict(parsed)
                    if prior_trajectory is None:
                        result = self.runtime.run_subject(visits)
                        execution_mode = "sequential_prefix"
                    else:
                        result = self.runtime.run_target(
                            visits,
                            prior_trajectory=prior_trajectory,
                            prior_belief_label=str(row.get("prior_belief_label") or "") or None,
                        )
                        execution_mode = "target_with_prior_state"
                    truth = self.adapter.task.normalize_label(row["routed_label"])
                    elapsed = perf_counter() - started
                    record = {
                        "subject_id": subject_id,
                        "target_visit_id": str(row["target_visit_id"]),
                        "task_name": self.adapter.task.name,
                        "ground_truth": truth,
                        "prediction": result.prediction,
                        "correct": result.prediction == truth,
                        "call_count": result.target_call_count,
                        "total_call_count": result.call_count,
                        "target_call_count": result.target_call_count,
                        "history_call_count": result.history_call_count,
                        "elapsed_seconds": elapsed,
                        "execution_mode": execution_mode,
                    }
                    predictions.append(record)
                    trace_stream.write(json.dumps(to_dict(result), ensure_ascii=True, sort_keys=True) + "\n")
                    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
                except Exception as exc:
                    error = {
                        "subject_id": subject_id,
                        "target_visit_id": str(row.get("target_visit_id") or ""),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    error_stream.write(json.dumps(error, ensure_ascii=True, sort_keys=True) + "\n")
                    error_stream.flush()
                    if self.config.fail_fast:
                        raise
        if not predictions:
            summary = {"task_name": self.adapter.task.name, "completed": 0, "errors": len(targets)}
        else:
            actual = [str(item["ground_truth"]) for item in predictions]
            predicted = [str(item["prediction"]) for item in predictions]
            summary = classification_metrics(actual, predicted, self.adapter.task.labels)
            summary.update(
                bootstrap_intervals(
                    actual,
                    predicted,
                    self.adapter.task.labels,
                    samples=self.config.bootstrap_samples,
                    seed=self.config.bootstrap_seed,
                )
            )
            summary.update(
                {
                    "task_name": self.adapter.task.name,
                    "completed": len(predictions),
                    "errors": len(targets) - len(predictions),
                    "mean_model_calls": sum(int(item["call_count"]) for item in predictions) / len(predictions),
                    "mean_total_model_calls": sum(int(item["total_call_count"]) for item in predictions)
                    / len(predictions),
                    "mean_target_model_calls": sum(int(item["target_call_count"]) for item in predictions)
                    / len(predictions),
                    "mean_history_model_calls": sum(int(item["history_call_count"]) for item in predictions)
                    / len(predictions),
                    "mean_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in predictions) / len(predictions),
                }
            )
        (self.output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8"
        )
        return summary
