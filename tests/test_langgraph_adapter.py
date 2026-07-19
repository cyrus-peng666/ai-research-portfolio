from __future__ import annotations

import pytest


pytest.importorskip("langgraph")

from langgraph.types import Command

from ai_research_portfolio.research_agent.langgraph_adapter import build_langgraph


def _start(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    pending = graph.invoke(
        {
            "goal": "Evaluate one controlled change.",
            "constraints": ["Keep the benchmark fixed."],
        },
        config=config,
    )
    assert "__interrupt__" in pending
    return config


def test_rejected_langgraph_branch_never_mutates_or_executes() -> None:
    graph = build_langgraph()
    config = _start(graph, "test-rejected")
    rejected = graph.invoke(
        Command(resume={"decision": "rejected", "feedback": "Revise the plan."}),
        config=config,
    )
    assert rejected["approval_status"] == "rejected"
    assert rejected.get("mutation_records", []) == []
    assert rejected.get("execution_records", []) == []
    assert [event["kind"] for event in rejected["audit_events"]] == [
        "plan.generated",
        "approval.rejected",
    ]


def test_approved_langgraph_branch_is_auditable() -> None:
    graph = build_langgraph()
    config = _start(graph, "test-approved")
    completed = graph.invoke(
        Command(resume={"decision": "approved", "feedback": "Proceed."}),
        config=config,
    )
    assert completed["approval_status"] == "approved"
    assert len(completed["mutation_records"]) == 2
    assert len(completed["execution_records"]) == 2
    assert completed["report"].startswith("# LangGraph Research Agent Run")
    assert [event["kind"] for event in completed["audit_events"]] == [
        "plan.generated",
        "approval.approved",
        "mutation.prepared",
        "execution.completed",
        "report.generated",
    ]

