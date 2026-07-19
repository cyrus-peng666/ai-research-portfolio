"""Data contracts for the offline research-agent workflow.

The contracts are intentionally independent from LangGraph and external model
providers.  They can be serialized into a graph state, inspected at an
approval boundary, and replayed in deterministic tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class WorkflowPhase(str, Enum):
    """Explicit phases used to guard workflow transitions."""

    INITIALIZED = "initialized"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    MUTATION_PREPARED = "mutation_prepared"
    EXECUTED = "executed"
    COMPLETED = "completed"


class ApprovalStatus(str, Enum):
    """Human decision recorded at the only mutation boundary."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExperimentPlan:
    """A proposed experiment produced before human approval."""

    experiment_id: str
    hypothesis: str
    target_metric: str
    change_kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MutationRecord:
    """A proposed in-memory change set.

    The public offline adapter does not edit source code or files.  A real
    integration may translate this record into a sandboxed mutation after the
    approval guard has passed.
    """

    experiment_id: str
    action: str
    target: str
    patch: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionRecord:
    """Result returned by an execution adapter."""

    experiment_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class AuditEvent:
    """Append-only audit event with deterministic sequence numbering."""

    sequence: int
    kind: str
    phase: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowState:
    """Serializable state shared by all research-agent stages."""

    task_id: str
    goal: str
    constraints: tuple[str, ...]
    phase: WorkflowPhase = WorkflowPhase.INITIALIZED
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approval_feedback: str = ""
    experiment_plan: tuple[ExperimentPlan, ...] = ()
    mutation_records: tuple[MutationRecord, ...] = ()
    execution_records: tuple[ExecutionRecord, ...] = ()
    report: str = ""
    audit_events: tuple[AuditEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation suitable for graph state."""

        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "constraints": list(self.constraints),
            "phase": self.phase.value,
            "approval_status": self.approval_status.value,
            "approval_feedback": self.approval_feedback,
            "experiment_plan": [asdict(item) for item in self.experiment_plan],
            "mutation_records": [asdict(item) for item in self.mutation_records],
            "execution_records": [asdict(item) for item in self.execution_records],
            "report": self.report,
            "audit_events": [asdict(item) for item in self.audit_events],
        }
