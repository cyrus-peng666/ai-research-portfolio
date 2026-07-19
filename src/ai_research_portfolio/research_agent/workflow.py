"""Human-in-the-loop research workflow with an enforceable approval boundary."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Literal, Sequence

from .adapters import (
    DeterministicExecutor,
    DeterministicPlanner,
    ExecutionAdapter,
    InMemoryMutationAdapter,
    MarkdownReporter,
    MutationAdapter,
    PlanningAdapter,
    ReportingAdapter,
)
from .models import ApprovalStatus, AuditEvent, WorkflowPhase, WorkflowState

ReviewRoute = Literal["approval", "mutation", "end"]


class InvalidTransitionError(RuntimeError):
    """Raised when a caller attempts to bypass the workflow state machine."""


class ResearchWorkflow:
    """Orchestrate planning, human review, mutation, execution, and reporting.

    Planning is read-only.  Mutation and execution adapters are reachable only
    from the approved route.  Rejection is terminal for the current state.
    """

    def __init__(
        self,
        *,
        planner: PlanningAdapter | None = None,
        mutator: MutationAdapter | None = None,
        executor: ExecutionAdapter | None = None,
        reporter: ReportingAdapter | None = None,
    ) -> None:
        self.planner = planner or DeterministicPlanner()
        self.mutator = mutator or InMemoryMutationAdapter()
        self.executor = executor or DeterministicExecutor()
        self.reporter = reporter or MarkdownReporter()

    @classmethod
    def offline(cls) -> "ResearchWorkflow":
        """Construct the public no-key, no-network workflow."""

        return cls()

    @staticmethod
    def _task_id(goal: str, constraints: Sequence[str]) -> str:
        payload = "\n".join((goal, *constraints)).encode("utf-8")
        return f"task-{hashlib.sha256(payload).hexdigest()[:12]}"

    @staticmethod
    def _event(
        state: WorkflowState,
        *,
        kind: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> WorkflowState:
        event = AuditEvent(
            sequence=len(state.audit_events) + 1,
            kind=kind,
            phase=state.phase.value,
            message=message,
            details=details or {},
        )
        return replace(state, audit_events=state.audit_events + (event,))

    def start(
        self,
        goal: str,
        *,
        constraints: Sequence[str] = (),
    ) -> WorkflowState:
        """Create a proposal and pause at the human approval gate."""

        normalized_goal = goal.strip()
        if not normalized_goal:
            raise ValueError("goal must not be empty")
        normalized_constraints = tuple(
            item.strip() for item in constraints if item.strip()
        )
        state = WorkflowState(
            task_id=self._task_id(normalized_goal, normalized_constraints),
            goal=normalized_goal,
            constraints=normalized_constraints,
        )
        state = self._event(
            state,
            kind="task.parsed",
            message="Research goal and constraints were normalized.",
            details={"constraint_count": len(normalized_constraints)},
        )

        plan = tuple(self.planner.propose(normalized_goal, normalized_constraints))
        if not plan:
            raise RuntimeError("planner returned an empty experiment plan")
        state = replace(state, experiment_plan=plan)
        state = self._event(
            state,
            kind="plan.generated",
            message="A read-only experiment plan was generated.",
            details={"experiment_count": len(plan)},
        )

        state = replace(state, phase=WorkflowPhase.AWAITING_APPROVAL)
        return self._event(
            state,
            kind="approval.requested",
            message="Human approval is required before mutation or execution.",
        )

    @staticmethod
    def route_after_review(state: WorkflowState) -> ReviewRoute:
        """Expose the conditional edge used by graph-style orchestrators."""

        if state.approval_status is ApprovalStatus.PENDING:
            return "approval"
        if state.approval_status is ApprovalStatus.APPROVED:
            return "mutation"
        return "end"

    def review(
        self,
        state: WorkflowState,
        decision: str,
        *,
        feedback: str = "",
    ) -> WorkflowState:
        """Resume a paused workflow with an explicit human decision."""

        if state.phase is not WorkflowPhase.AWAITING_APPROVAL:
            raise InvalidTransitionError(
                f"review requires awaiting_approval, got {state.phase.value}"
            )
        normalized = decision.strip().lower()
        if normalized not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")

        if normalized == "rejected":
            state = replace(
                state,
                phase=WorkflowPhase.REJECTED,
                approval_status=ApprovalStatus.REJECTED,
                approval_feedback=feedback.strip(),
            )
            state = self._event(
                state,
                kind="approval.rejected",
                message="The proposal was rejected; the workflow is terminal.",
                details={"feedback": state.approval_feedback},
            )
            return self._event(
                state,
                kind="workflow.stopped",
                message="No mutation or execution adapter was called.",
            )

        state = replace(
            state,
            phase=WorkflowPhase.APPROVED,
            approval_status=ApprovalStatus.APPROVED,
            approval_feedback=feedback.strip(),
        )
        state = self._event(
            state,
            kind="approval.approved",
            message="The proposal was approved for isolated preparation.",
            details={"feedback": state.approval_feedback},
        )
        if self.route_after_review(state) != "mutation":
            raise InvalidTransitionError("approved state did not route to mutation")
        return self._run_approved_branch(state)

    def _run_approved_branch(self, state: WorkflowState) -> WorkflowState:
        """Run guarded stages; this method is unreachable from rejection."""

        if (
            state.phase is not WorkflowPhase.APPROVED
            or state.approval_status is not ApprovalStatus.APPROVED
        ):
            raise InvalidTransitionError(
                "mutation and execution require an explicitly approved state"
            )

        changes = tuple(self.mutator.prepare(state.experiment_plan))
        state = replace(
            state,
            phase=WorkflowPhase.MUTATION_PREPARED,
            mutation_records=changes,
        )
        state = self._event(
            state,
            kind="mutation.prepared",
            message="Declarative in-memory changes were prepared.",
            details={"change_count": len(changes)},
        )

        executions = tuple(self.executor.execute(changes))
        state = replace(
            state,
            phase=WorkflowPhase.EXECUTED,
            execution_records=executions,
        )
        state = self._event(
            state,
            kind="execution.completed",
            message="The approved offline execution finished.",
            details={"execution_count": len(executions)},
        )

        report = self.reporter.render(state)
        state = replace(state, phase=WorkflowPhase.COMPLETED, report=report)
        state = self._event(
            state,
            kind="report.generated",
            message="A deterministic audit report was generated.",
        )
        return self._event(
            state,
            kind="workflow.completed",
            message="The approved workflow completed.",
        )

    def run(
        self,
        goal: str,
        *,
        decision: str,
        constraints: Sequence[str] = (),
        feedback: str = "",
    ) -> WorkflowState:
        """Convenience wrapper for start followed by an explicit review."""

        pending = self.start(goal, constraints=constraints)
        return self.review(pending, decision, feedback=feedback)
