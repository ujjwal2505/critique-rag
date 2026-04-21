"""LangGraph wiring for the self-critiquing RAG workflow.

    START -> retrieve -> generate -> critique -> (conditional)
        grounded                      -> finalize -> END
        rejected & retries remain     -> reformulate_query -> retrieve (loop)
        rejected & retries exhausted  -> abstain -> END

The retry cap lives in `route_after_critique` (compared against
`max_retries`); LangGraph's `recursion_limit` is only a backstop.
"""

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import RAGState


def route_after_critique(state: RAGState) -> str:
    if state["critic_verdict"]["grounded"]:
        return "finalize"
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "reformulate_query"
    return "abstain"


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("critique", nodes.critique)
    graph.add_node("reformulate_query", nodes.reformulate_query)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("abstain", nodes.abstain)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "finalize": "finalize",
            "reformulate_query": "reformulate_query",
            "abstain": "abstain",
        },
    )
    graph.add_edge("reformulate_query", "retrieve")
    graph.add_edge("finalize", END)
    graph.add_edge("abstain", END)
    return graph.compile()
