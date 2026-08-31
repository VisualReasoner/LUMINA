from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from lumina.agent.controller import EvidenceController
from lumina.memory.trajectory import TrajectoryConfig, update_trajectory
from lumina.schemas.states import SubjectResult, TrajectoryState, VisitInput


@dataclass
class LUMINARuntime:
    controller: EvidenceController
    trajectory_config: TrajectoryConfig = TrajectoryConfig()
    use_trajectory_memory: bool = True
    use_smc: bool = True

    def run_target(
        self,
        visits: Sequence[VisitInput],
        *,
        prior_trajectory: TrajectoryState | None = None,
        prior_belief_label: str | None = None,
    ) -> SubjectResult:
        """Run one target visit with an optional precomputed trajectory state."""
        if not visits:
            raise ValueError("A target run requires a visible subject prefix.")
        subject_ids = {visit.subject_id for visit in visits}
        if len(subject_ids) != 1:
            raise ValueError("All visits in a target run must share one subject_id.")
        ordered = sorted(visits, key=lambda item: (item.visit_date, item.visit_id))
        visit_ids = [visit.visit_id for visit in ordered]
        if len(visit_ids) != len(set(visit_ids)):
            raise ValueError("A target run cannot contain duplicate visit_id values.")
        trajectory = (prior_trajectory or TrajectoryState()) if self.use_trajectory_memory else TrajectoryState()
        if trajectory.adapter_name and trajectory.adapter_name != self.controller.adapter.adapter_name:
            raise ValueError(
                f"Prior trajectory adapter mismatch: {trajectory.adapter_name!r} != "
                f"{self.controller.adapter.adapter_name!r}."
            )
        foreign_subjects = {
            event.subject_id for event in trajectory.events if event.subject_id and event.subject_id != ordered[0].subject_id
        }
        if foreign_subjects:
            raise ValueError(f"Prior trajectory belongs to different subject(s): {sorted(foreign_subjects)}")
        starting_calls = int(getattr(self.controller.model, "call_count", 0))
        result = self.controller.run_visit(
            history=ordered[:-1],
            current=ordered[-1],
            trajectory=trajectory,
            prior_belief_label=prior_belief_label if self.use_trajectory_memory else None,
        )
        event = self.controller.build_trajectory_event(
            result, prior_belief_label if self.use_trajectory_memory else None
        )
        if self.use_trajectory_memory:
            trajectory = update_trajectory(trajectory, event, self.trajectory_config, use_smc=self.use_smc)
            trajectory.adapter_name = self.controller.adapter.adapter_name
        ending_calls = int(getattr(self.controller.model, "call_count", starting_calls))
        calls = max(0, ending_calls - starting_calls)
        return SubjectResult(
            subject_id=ordered[0].subject_id,
            task_name=self.controller.adapter.task.name,
            prediction=self.controller.adapter.task.normalize_label(result.belief.leading_label),
            visits=[result],
            trajectory=trajectory,
            call_count=calls,
            target_call_count=calls,
            history_call_count=0,
        )

    def run_subject(
        self,
        visits: Sequence[VisitInput],
        *,
        initial_trajectory: TrajectoryState | None = None,
    ) -> SubjectResult:
        if not self.use_trajectory_memory:
            return self.run_target(visits)
        if not visits:
            raise ValueError("A subject run requires at least one visible visit.")
        subject_ids = {visit.subject_id for visit in visits}
        if len(subject_ids) != 1:
            raise ValueError("All visits in a subject run must share one subject_id.")
        ordered = sorted(visits, key=lambda item: (item.visit_date, item.visit_id))
        visit_ids = [visit.visit_id for visit in ordered]
        if len(visit_ids) != len(set(visit_ids)):
            raise ValueError("A subject run cannot contain duplicate visit_id values.")
        trajectory = initial_trajectory or TrajectoryState()
        if trajectory.adapter_name and trajectory.adapter_name != self.controller.adapter.adapter_name:
            raise ValueError(
                f"Initial trajectory adapter mismatch: {trajectory.adapter_name!r} != "
                f"{self.controller.adapter.adapter_name!r}."
            )
        foreign_subjects = {
            event.subject_id for event in trajectory.events if event.subject_id and event.subject_id != ordered[0].subject_id
        }
        if foreign_subjects:
            raise ValueError(f"Initial trajectory belongs to different subject(s): {sorted(foreign_subjects)}")
        results = []
        prior_belief_label = None
        history: list[VisitInput] = []
        starting_calls = int(getattr(self.controller.model, "call_count", 0))
        for visit in ordered:
            result = self.controller.run_visit(
                history=history,
                current=visit,
                trajectory=trajectory,
                prior_belief_label=prior_belief_label,
            )
            event = self.controller.build_trajectory_event(result, prior_belief_label)
            if self.use_trajectory_memory:
                trajectory = update_trajectory(
                    trajectory,
                    event,
                    self.trajectory_config,
                    use_smc=self.use_smc,
                )
                trajectory.adapter_name = self.controller.adapter.adapter_name
            results.append(result)
            prior_belief_label = result.belief.leading_label if self.use_trajectory_memory else None
            history.append(visit)
        ending_calls = int(getattr(self.controller.model, "call_count", starting_calls))
        final = results[-1]
        target_calls = int(final.model_call_count)
        total_calls = max(0, ending_calls - starting_calls)
        return SubjectResult(
            subject_id=ordered[0].subject_id,
            task_name=self.controller.adapter.task.name,
            prediction=self.controller.adapter.task.normalize_label(final.belief.leading_label),
            visits=results,
            trajectory=trajectory,
            call_count=total_calls,
            target_call_count=target_calls,
            history_call_count=max(0, total_calls - target_calls),
        )
