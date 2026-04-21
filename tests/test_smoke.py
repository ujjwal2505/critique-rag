"""Smoke tests that need no API key — pure structure and routing logic."""

from critique_rag.graph import build_graph, route_after_critique
from critique_rag.ingest import chunk_text


def test_graph_compiles():
    assert build_graph() is not None


def test_route_grounded_goes_to_finalize():
    state = {"critic_verdict": {"grounded": True}, "retry_count": 0, "max_retries": 2}
    assert route_after_critique(state) == "finalize"


def test_route_rejected_with_retries_left_reformulates():
    state = {"critic_verdict": {"grounded": False}, "retry_count": 0, "max_retries": 2}
    assert route_after_critique(state) == "reformulate_query"


def test_route_rejected_retries_exhausted_abstains():
    state = {"critic_verdict": {"grounded": False}, "retry_count": 2, "max_retries": 2}
    assert route_after_critique(state) == "abstain"


def test_chunk_text_packs_paragraphs():
    text = "para one\n\npara two\n\npara three"
    chunks = chunk_text(text, max_chars=20)
    assert chunks
    assert all(c.strip() for c in chunks)
