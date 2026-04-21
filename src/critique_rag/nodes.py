"""The five workflow nodes (plus a small `finalize` node).

Each node returns a partial state update. Trace events are appended via the
additive ``trace_events`` reducer.
"""

from . import llm, vector_store
from .config import TOP_K
from .embeddings import embed_one
from .state import CriticVerdict, RAGState
from .trace import event

ABSTAIN_MESSAGE = (
    "I don't have enough information to answer that based on the provided documents."
)


def retrieve(state: RAGState) -> dict:
    query = state["current_query"]
    chunks = vector_store.query(embed_one(query), TOP_K)
    return {
        "retrieved_chunks": chunks,
        "trace_events": [
            event(
                "retrieve",
                query=query,
                retrieved=[
                    {
                        "id": c["id"],
                        "source": c["source"],
                        "distance": c["distance"],
                        "preview": c["text"][:160],
                    }
                    for c in chunks
                ],
            )
        ],
    }


def generate(state: RAGState) -> dict:
    answer = llm.generate_answer(state["current_query"], state["retrieved_chunks"])
    return {
        "generated_answer": answer,
        "trace_events": [event("generate", query=state["current_query"], answer=answer)],
    }


def critique(state: RAGState) -> dict:
    # Critic-OFF path (used by the eval): emit a pass-through verdict so the graph
    # returns the first generated answer with no retry loop.
    if not state.get("critic_enabled", True):
        verdict = {"grounded": True, "reasoning": "critic disabled", "unsupported_claims": []}
        return {
            "critic_verdict": verdict,
            "trace_events": [event("critique", enabled=False, verdict=verdict)],
        }

    verdict = llm.critique_answer(
        state["current_query"], state["generated_answer"], state["retrieved_chunks"]
    ).model_dump()
    return {
        "critic_verdict": verdict,
        "trace_events": [event("critique", enabled=True, verdict=verdict)],
    }


def reformulate_query(state: RAGState) -> dict:
    verdict = CriticVerdict(**state["critic_verdict"])
    new_query = llm.reformulate(state["original_query"], state["current_query"], verdict)
    new_count = state.get("retry_count", 0) + 1
    return {
        "current_query": new_query,
        "retry_count": new_count,
        "trace_events": [
            event("reformulate_query", new_query=new_query, retry_count=new_count)
        ],
    }


def finalize(state: RAGState) -> dict:
    return {
        "final_answer": state["generated_answer"],
        "status": "grounded" if state.get("critic_enabled", True) else "passthrough",
        "trace_events": [event("finalize", status="grounded")],
    }


def abstain(state: RAGState) -> dict:
    return {
        "final_answer": ABSTAIN_MESSAGE,
        "status": "abstained",
        "trace_events": [event("abstain", message=ABSTAIN_MESSAGE)],
    }
