"""Graph state schema and the critic's structured verdict."""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    """Structured output of the critic / judge calls."""

    grounded: bool = Field(
        description="True only if every factual claim in the answer is supported "
        "by the provided context chunks."
    )
    reasoning: str = Field(description="Brief explanation of the judgment.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims made in the answer that the context does not support.",
    )


class RAGState(TypedDict, total=False):
    """State threaded through the LangGraph workflow.

    `trace_events` uses an additive reducer so each node appends its own events
    instead of overwriting the log.
    """

    original_query: str
    current_query: str
    retrieved_chunks: list[dict]
    generated_answer: str
    critic_verdict: dict
    retry_count: int
    max_retries: int
    critic_enabled: bool
    final_answer: str
    status: str
    trace_events: Annotated[list[dict], operator.add]
