"""Anthropic API calls for the four LLM roles.

The generator and the critic are deliberately *separate* calls with separate
system prompts and separate models — the critic never grades inside the same
pass that produced the answer. The judge is an eval-only role using yet another
(independent) model.

Uses the Anthropic Python SDK per project conventions: ``anthropic.Anthropic()``
resolves ``ANTHROPIC_API_KEY`` from the environment (loaded from ``.env`` in
``config.py``), and ``messages.parse(..., output_format=...)`` returns a
validated Pydantic object via structured outputs.
"""

import anthropic

from .config import (
    CRITIC_MAX_TOKENS,
    CRITIC_MODEL,
    GEN_MAX_TOKENS,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    REFORMULATE_MAX_TOKENS,
    REFORMULATE_MODEL,
)
from .state import CriticVerdict

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no context passages were retrieved)"
    return "\n\n".join(
        f"[{i + 1}] (source: {c.get('source')})\n{c['text']}"
        for i, c in enumerate(chunks)
    )


def _text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text").strip()


# --- Generator -------------------------------------------------------------

GEN_SYSTEM = (
    "You are a careful question-answering assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Do not use outside "
    "knowledge. If the context does not contain enough information to answer, say "
    "so explicitly and do not guess. Keep the answer concise and cite passage "
    "numbers like [1], [2] where relevant."
)


def generate_answer(query: str, chunks: list[dict]) -> str:
    message = client().messages.create(
        model=GENERATOR_MODEL,
        max_tokens=GEN_MAX_TOKENS,
        system=GEN_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}",
            }
        ],
    )
    return _text(message)


# --- Critic (in-loop, separate model + prompt) -----------------------------

CRITIC_SYSTEM = (
    "You are a strict grounding critic. You are given a QUESTION, an ANSWER, and "
    "the CONTEXT passages that were available to the answerer. Decide whether "
    "every factual claim in the ANSWER is directly supported by the CONTEXT. "
    "Judge ONLY against the provided context, never against your own world "
    "knowledge. If the answer explicitly states that the context is insufficient "
    "and makes no unsupported factual claims, treat it as grounded. List any "
    "claims that are not supported by the context."
)


def critique_answer(query: str, answer: str, chunks: list[dict]) -> CriticVerdict:
    response = client().messages.parse(
        model=CRITIC_MODEL,
        max_tokens=CRITIC_MAX_TOKENS,
        system=CRITIC_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{_format_context(chunks)}\n\n"
                    f"QUESTION: {query}\n\nANSWER: {answer}"
                ),
            }
        ],
        output_format=CriticVerdict,
    )
    return response.parsed_output


# --- Query reformulation ---------------------------------------------------

REFORMULATE_SYSTEM = (
    "You rewrite search queries to improve document retrieval. The previous "
    "answer was judged not grounded in the retrieved passages. Produce a single "
    "improved search query that may broaden, rephrase, or extract a sub-question "
    "so retrieval finds more relevant passages. Respond with ONLY the rewritten "
    "query text — no preamble, no quotes."
)


def reformulate(original: str, current: str, verdict: CriticVerdict) -> str:
    feedback = (
        f"Critic reasoning: {verdict.reasoning}\n"
        f"Unsupported claims: {verdict.unsupported_claims}"
    )
    message = client().messages.create(
        model=REFORMULATE_MODEL,
        max_tokens=REFORMULATE_MAX_TOKENS,
        system=REFORMULATE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original question: {original}\n"
                    f"Current query: {current}\n{feedback}\n\nRewritten query:"
                ),
            }
        ],
    )
    return _text(message)


# --- Judge (eval only, independent model) ----------------------------------

JUDGE_SYSTEM = (
    "You are an independent grounding judge for an evaluation. Given a QUESTION, "
    "a final ANSWER, and the CONTEXT passages, decide whether the answer is fully "
    "supported by the context. Judge ONLY against the context. If the answer is "
    "an explicit refusal/abstention that makes no unsupported factual claims, "
    "mark it grounded."
)


def judge_grounded(query: str, answer: str, chunks: list[dict]) -> CriticVerdict:
    response = client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=CRITIC_MAX_TOKENS,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{_format_context(chunks)}\n\n"
                    f"QUESTION: {query}\n\nANSWER: {answer}"
                ),
            }
        ],
        output_format=CriticVerdict,
    )
    return response.parsed_output
