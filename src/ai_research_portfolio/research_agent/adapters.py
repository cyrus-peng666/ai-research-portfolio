"""Deterministic, offline adapters for the public research-agent demo."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol, Sequence

from .models import (
    ExecutionRecord,
    ExperimentPlan,
    MutationRecord,
    WorkflowState,
)


class PlanningAdapter(Protocol):
    """Generate a read-only experiment proposal."""

    def propose(
        self,
        goal: str,
        constraints: Sequence[str],
    ) -> Sequence[ExperimentPlan]: ...


class MutationAdapter(Protocol):
    """Prepare isolated changes after approval."""

    def prepare(self, plan: Sequence[ExperimentPlan]) -> Sequence[MutationRecord]: ...


class ExecutionAdapter(Protocol):
    """Execute approved, prepared changes."""

    def execute(self, changes: Sequence[MutationRecord]) -> Sequence[ExecutionRecord]: ...


class ReportingAdapter(Protocol):
    """Summarize an executed workflow."""

    def render(self, state: WorkflowState) -> str: ...


class DeterministicPlanner:
    """Return a small, stable plan without calling an LLM."""

    def propose(
        self,
        goal: str,
        constraints: Sequence[str],
    ) -> Sequence[ExperimentPlan]:
        del goal, constraints
        return (
            ExperimentPlan(
                experiment_id="exp-baseline",
                hypothesis="Reproduce the baseline under a fixed evaluation protocol.",
                target_metric="validation_score",
                change_kind="config_only",
                parameters={"variant": "baseline", "seed": 7},
            ),
            ExperimentPlan(
                experiment_id="exp-controlled-change",
                hypothesis=(
                    "Test one isolated optimization change while holding the "
                    "evaluation protocol constant."
                ),
                target_metric="validation_score",
                change_kind="config_only",
                parameters={
                    "variant": "controlled_change",
                    "learning_rate_scale": 0.5,
                    "seed": 7,
                },
            ),
        )


class InMemoryMutationAdapter:
    """Translate plans into declarative change records without touching files."""

    def prepare(self, plan: Sequence[ExperimentPlan]) -> Sequence[MutationRecord]:
        return tuple(
            MutationRecord(
                experiment_id=item.experiment_id,
                action="prepare_isolated_config",
                target="in_memory_experiment_config",
                patch=dict(item.parameters),
            )
            for item in plan
        )


class DeterministicExecutor:
    """Produce reproducible synthetic metrics without training or network I/O."""

    @staticmethod
    def _score(change: MutationRecord) -> float:
        payload = json.dumps(
            {
                "experiment_id": change.experiment_id,
                "patch": dict(change.patch),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return round(0.65 + (int(digest[:8], 16) % 2500) / 10_000, 4)

    def execute(self, changes: Sequence[MutationRecord]) -> Sequence[ExecutionRecord]:
        return tuple(
            ExecutionRecord(
                experiment_id=item.experiment_id,
                status="simulated_success",
                metrics={"synthetic_validation_score": self._score(item)},
                notes="Offline deterministic demonstration; not an empirical result.",
            )
            for item in changes
        )


class MarkdownReporter:
    """Render a compact and deterministic audit report."""

    def render(self, state: WorkflowState) -> str:
        lines = [
            "# Research Agent Run",
            "",
            f"- Task: `{state.task_id}`",
            f"- Goal: {state.goal}",
            f"- Approval: `{state.approval_status.value}`",
            "- Mode: offline deterministic demonstration",
            "",
            "## Executions",
            "",
        ]
        for record in state.execution_records:
            score = record.metrics.get("synthetic_validation_score")
            lines.append(
                f"- `{record.experiment_id}`: {record.status}; "
                f"synthetic validation score = {score:.4f}"
            )
        lines.extend(
            [
                "",
                "> Metrics above are synthetic workflow fixtures, not research results.",
            ]
        )
        return "\n".join(lines)
