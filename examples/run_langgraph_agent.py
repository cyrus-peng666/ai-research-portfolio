"""Run the optional LangGraph adapter through an approval interrupt."""

from __future__ import annotations

from langgraph.types import Command

from ai_research_portfolio.research_agent.langgraph_adapter import build_langgraph


def main() -> None:
    graph = build_langgraph()
    config = {"configurable": {"thread_id": "public-demo-approved"}}
    pending = graph.invoke(
        {
            "goal": "Evaluate one controlled model change.",
            "constraints": ["Keep labels and the evaluation split fixed."],
        },
        config=config,
    )
    assert "__interrupt__" in pending
    completed = graph.invoke(
        Command(resume={"decision": "approved", "feedback": "Proceed."}),
        config=config,
    )
    print(completed["report"])


if __name__ == "__main__":
    main()

