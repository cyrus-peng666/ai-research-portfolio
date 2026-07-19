from __future__ import annotations

import json

import pytest

from ai_research_portfolio.research_agent import (
    DeterministicExecutor,
    InMemoryMutationAdapter,
    InvalidTransitionError,
    ResearchWorkflow,
    WorkflowPhase,
)


class SpyMutator(InMemoryMutationAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def prepare(self, plan):
        self.calls += 1
        return super().prepare(plan)


class SpyExecutor(DeterministicExecutor):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, changes):
        self.calls += 1
        return super().execute(changes)


def test_start_pauses_before_any_mutation_or_execution() -> None:
    mutator = SpyMutator()
    executor = SpyExecutor()
    workflow = ResearchWorkflow(mutator=mutator, executor=executor)

    state = workflow.start(
        "Compare a baseline with one controlled change.",
        constraints=("Keep the evaluation protocol fixed.",),
    )

    assert state.phase is WorkflowPhase.AWAITING_APPROVAL
    assert workflow.route_after_review(state) == "approval"
    assert state.mutation_records == ()
    assert state.execution_records == ()
    assert mutator.calls == 0
    assert executor.calls == 0


def test_rejection_is_terminal_and_cannot_mutate_or_execute() -> None:
    mutator = SpyMutator()
    executor = SpyExecutor()
    workflow = ResearchWorkflow(mutator=mutator, executor=executor)
    pending = workflow.start("Evaluate a model change safely.")

    rejected = workflow.review(
        pending,
        "rejected",
        feedback="The leakage audit is incomplete.",
    )

    assert rejected.phase is WorkflowPhase.REJECTED
    assert workflow.route_after_review(rejected) == "end"
    assert rejected.mutation_records == ()
    assert rejected.execution_records == ()
    assert mutator.calls == 0
    assert executor.calls == 0
    event_kinds = [event.kind for event in rejected.audit_events]
    assert "approval.rejected" in event_kinds
    assert "workflow.stopped" in event_kinds
    assert "mutation.prepared" not in event_kinds
    assert "execution.completed" not in event_kinds

    with pytest.raises(InvalidTransitionError, match="explicitly approved"):
        workflow._run_approved_branch(rejected)
    assert mutator.calls == 0
    assert executor.calls == 0


def test_approval_is_the_only_route_to_mutation_and_execution() -> None:
    mutator = SpyMutator()
    executor = SpyExecutor()
    workflow = ResearchWorkflow(mutator=mutator, executor=executor)
    pending = workflow.start("Evaluate a model change safely.")

    completed = workflow.review(pending, "approved")

    assert completed.phase is WorkflowPhase.COMPLETED
    assert mutator.calls == 1
    assert executor.calls == 1
    assert len(completed.mutation_records) == 2
    assert len(completed.execution_records) == 2
    assert "synthetic workflow fixtures" in completed.report
    assert [event.sequence for event in completed.audit_events] == list(
        range(1, len(completed.audit_events) + 1)
    )


def test_terminal_state_cannot_be_reviewed_again() -> None:
    workflow = ResearchWorkflow.offline()
    rejected = workflow.run(
        "Evaluate a model change safely.",
        decision="rejected",
    )

    with pytest.raises(InvalidTransitionError):
        workflow.review(rejected, "approved")


def test_invalid_review_decision_is_rejected() -> None:
    workflow = ResearchWorkflow.offline()
    pending = workflow.start("Evaluate a model change safely.")

    with pytest.raises(ValueError, match="approved.*rejected"):
        workflow.review(pending, "maybe")


def test_offline_run_is_deterministic_and_json_serializable() -> None:
    first = ResearchWorkflow.offline().run(
        "Evaluate a model change safely.",
        decision="approved",
        constraints=("Keep labels fixed.",),
    )
    second = ResearchWorkflow.offline().run(
        "Evaluate a model change safely.",
        decision="approved",
        constraints=("Keep labels fixed.",),
    )

    assert first.to_dict() == second.to_dict()
    json.dumps(first.to_dict(), ensure_ascii=False)
