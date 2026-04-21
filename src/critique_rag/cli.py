"""Command-line entry point.

    uv run python -m critique_rag.cli "How long does the Pro plan retain raw events?"
    uv run python -m critique_rag.cli "Does Northwind have an iOS app?"   # abstains
    uv run python -m critique_rag.cli "..." --no-critic                   # critic-off path
"""

import argparse

from .config import MAX_RETRIES
from .graph import build_graph
from .trace import write_trace_file


def run_query(
    question: str, critic_enabled: bool = True, max_retries: int = MAX_RETRIES
) -> dict:
    """Invoke the graph for one question and return the final state."""
    app = build_graph()
    initial_state = {
        "original_query": question,
        "current_query": question,
        "retry_count": 0,
        "max_retries": max_retries,
        "critic_enabled": critic_enabled,
        "trace_events": [],
    }
    # recursion_limit is a backstop only; the real cap is max_retries in routing.
    return app.invoke(initial_state, config={"recursion_limit": 25})


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-critiquing RAG")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="disable the critic and retry loop (single retrieve->generate pass)",
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args()

    final = run_query(
        args.question,
        critic_enabled=not args.no_critic,
        max_retries=args.max_retries,
    )
    trace_path = write_trace_file(final)

    print("\n=== ANSWER ===")
    print(final.get("final_answer"))
    print(f"\nstatus: {final.get('status')}   retries used: {final.get('retry_count', 0)}")
    verdict = final.get("critic_verdict")
    if verdict:
        print(f"critic grounded: {verdict.get('grounded')}")
    print(f"trace: {trace_path}")


if __name__ == "__main__":
    main()
