"""Optional LangGraph adapter for the public research workflow.

The core workflow stays framework-independent and testable without optional
packages. This module adds a real LangGraph ``interrupt`` boundary when the
``agent`` extra is installed.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, TypedDict

from .adapters import DeterministicExecutor, DeterministicPlanner, InMemoryMutationAdapter
from .models import ExperimentPlan, MutationRecord


class LangGraphResearchState(TypedDict, total=False):
    """JSON-serializable state persisted by LangGraph checkpoints."""

    goal: str
    constraints: list[str]
    experiment_plan: list[dict[str, Any]]
    approval_status: Literal["pending", "approved", "rejected"]
    approval_feedback: str
    mutation_records: list[dict[str, Any]]
    execution_records: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    report: str


def _append_event(
    state: LangGraphResearchState,
    kind: str,
    message: str,
) -> list[dict[str, Any]]:
    events = list(state.get("audit_events", []))
    events.append(
        {
            "sequence": len(events) + 1,
            "kind": kind,
            "message": message,
        }
    )
    return events


def build_langgraph():
    """Compile a checkpoint-ready graph with a real approval interrupt."""

    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "LangGraph support requires `pip install -e '.[agent]'`."
        ) from error

    planner = DeterministicPlanner()
    mutator = InMemoryMutationAdapter()
    executor = DeterministicExecutor()

    def plan_node(state: LangGraphResearchState) -> LangGraphResearchState:
        goal = state.get("goal", "").strip()
        if not goal:
            raise ValueError("goal must not be empty")
        constraints = [
            item.strip() for item in state.get("constraints", []) if item.strip()
        ]
        plan = planner.propose(goal, constraints)
        return {
            "goal": goal,
            "constraints": constraints,
            "experiment_plan": [asdict(item) for item in plan],
            "approval_status": "pending",
            "audit_events": _append_event(
                state,
                "plan.generated",
                "A read-only plan was generated before the approval gate.",
            ),
        }

    def approval_node(state: LangGraphResearchState) -> LangGraphResearchState:
        response = interrupt(
            {
                "type": "research_plan_approval",
                "allowed_decisions": ["approved", "rejected"],
                "experiment_plan": state.get("experiment_plan", []),
            }
        )
        if isinstance(response, str):
            decision = response.strip().lower()
            feedback = ""
        elif isinstance(response, dict):
            decision = str(response.get("decision", "")).strip().lower()
            feedback = str(response.get("feedback", "")).strip()
        else:
            raise ValueError("approval response must be a string or mapping")
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be 'approved' or 'rejected'")
        return {
            "approval_status": decision,
            "approval_feedback": feedback,
            "audit_events": _append_event(
                state,
                f"approval.{decision}",
                "The plan was explicitly reviewed by a human.",
            ),
        }

    def route_after_approval(state: LangGraphResearchState) -> str:
        return "mutation" if state.get("approval_status") == "approved" else "end"

    def mutation_node(state: LangGraphResearchState) -> LangGraphResearchState:
        if state.get("approval_status") != "approved":
            raise RuntimeError("mutation requires explicit approval")
        plan = [ExperimentPlan(**item) for item in state.get("experiment_plan", [])]
        changes = mutator.prepare(plan)
        return {
            "mutation_records": [asdict(item) for item in changes],
            "audit_events": _append_event(
                state,
                "mutation.prepared",
                "In-memory declarative changes were prepared.",
            ),
        }

    def execution_node(state: LangGraphResearchState) -> LangGraphResearchState:
        if state.get("approval_status") != "approved":
            raise RuntimeError("execution requires explicit approval")
        changes = [MutationRecord(**item) for item in state.get("mutation_records", [])]
        records = executor.execute(changes)
        return {
            "execution_records": [asdict(item) for item in records],
            "audit_events": _append_event(
                state,
                "execution.completed",
                "The deterministic offline execution completed.",
            ),
        }

    def report_node(state: LangGraphResearchState) -> LangGraphResearchState:
        lines = [
            "# LangGraph Research Agent Run",
            "",
            f"- Approval: `{state.get('approval_status', 'pending')}`",
            "- Mode: deterministic offline demonstration",
            "",
            "> Synthetic fixtures only; no empirical result is reported.",
        ]
        return {
            "report": "\n".join(lines),
            "audit_events": _append_event(
                state,
                "report.generated",
                "A compact audit report was generated.",
            ),
        }

    graph = StateGraph(LangGraphResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("approval", approval_node)
    graph.add_node("mutation", mutation_node)
    graph.add_node("execution", execution_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approval")
    graph.add_conditional_edges(
        "approval",
        route_after_approval,
        {"mutation": "mutation", "end": END},
    )
    graph.add_edge("mutation", "execution")
    graph.add_edge("execution", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=MemorySaver())

