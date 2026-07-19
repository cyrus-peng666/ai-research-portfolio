"""Run the deterministic research-agent demo without API keys or network I/O."""

from __future__ import annotations

import json

from ai_research_portfolio.research_agent import ResearchWorkflow


def summarize(label: str, state) -> None:
    print(f"\n=== {label} ===")
    print(
        json.dumps(
            {
                "task_id": state.task_id,
                "phase": state.phase.value,
                "approval": state.approval_status.value,
                "mutations": len(state.mutation_records),
                "executions": len(state.execution_records),
                "audit_events": [event.kind for event in state.audit_events],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    goal = "Evaluate one controlled model change under a fixed benchmark protocol."
    constraints = (
        "Do not change the data split or label definition.",
        "Require human approval before any mutation or execution.",
    )

    rejected_workflow = ResearchWorkflow.offline()
    rejected = rejected_workflow.review(
        rejected_workflow.start(goal, constraints=constraints),
        "rejected",
        feedback="Add a leakage check before running experiments.",
    )
    summarize("rejected branch", rejected)

    approved_workflow = ResearchWorkflow.offline()
    approved = approved_workflow.review(
        approved_workflow.start(goal, constraints=constraints),
        "approved",
        feedback="Proceed with the fixed offline protocol.",
    )
    summarize("approved branch", approved)
    print("\n" + approved.report)


if __name__ == "__main__":
    main()
