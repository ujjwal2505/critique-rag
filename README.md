# Self-Critiquing RAG

A retrieval-augmented generation system that **checks its own work**. Instead of
a linear `retrieve → generate` chain, it runs a stateful, cyclical
[LangGraph](https://langchain-ai.github.io/langgraph/) workflow: it generates an
answer grounded only in retrieved chunks, a *separate* critic model judges
whether that answer is actually supported by those chunks, and if not, the system
reformulates the query and retries — up to a hard limit. If it still can't ground
an answer, it abstains instead of guessing.

## Why it's not a linear chain

```
                ┌──────────┐
   START ─────► │ retrieve │ ◄───────────────────────┐
                └────┬─────┘                          │
                     ▼                                 │
                ┌──────────┐                           │ reformulated query
                │ generate │                           │ (retry_count++)
                └────┬─────┘                           │
                     ▼                                 │
                ┌──────────┐     reject &       ┌──────────────────┐
                │ critique │ ──retries remain──► │ reformulate_query│
                └────┬─────┘                     └──────────────────┘
            grounded │ │ reject & retries exhausted
                     ▼ ▼
              ┌──────────┐     ┌─────────┐
              │ finalize │     │ abstain │
              └────┬─────┘     └────┬────┘
                   ▼                ▼
                  END              END
```

The cycle (`critique → reformulate_query → retrieve`) is the whole point: the
graph can **reject its own output and try again**, which a basic chain cannot do.

### The critic is a genuinely separate step

The generator and the critic are **different models, different prompts, different
API calls**:

| Role        | Model              | Job                                                           |
| ----------- | ------------------ | ------------------------------------------------------------ |
| Generator   | `claude-sonnet-4-6` | Answer using only the retrieved chunks.                       |
| Critic      | `claude-haiku-4-5`  | Judge groundedness of that answer **against the chunks only**. Returns structured JSON. |
| Reformulator| `claude-sonnet-4-6` | Rewrite the query when the critic rejects the answer.        |
| Judge (eval)| `claude-sonnet-4-6` | Independent grader for the eval — *different model from the in-loop critic*. |

The critic never grades its own generation in the same pass. It returns:

```json
{ "grounded": true, "reasoning": "...", "unsupported_claims": ["..."] }
```

### No infinite loops

The retry cap is enforced in routing (`retry_count < max_retries`, default
`max_retries = 2`). LangGraph's `recursion_limit` is set as a backstop only.

## Project layout

```
src/critique_rag/
  config.py         model IDs, TOP_K, MAX_RETRIES, paths (loads .env)
  embeddings.py     sentence-transformers all-MiniLM-L6-v2 (local, free)
  vector_store.py   ChromaDB persistent collection (cosine)
  ingest.py         load docs -> chunk -> embed -> upsert
  state.py          RAGState (graph state) + CriticVerdict (pydantic)
  llm.py            generate / critique / reformulate / judge (Anthropic)
  nodes.py          the 5 nodes (+ finalize)
  graph.py          StateGraph wiring + conditional routing
  trace.py          per-request JSON trace logging
  cli.py            ask a question, print answer + trace path
data/
  sample_docs/      fictional "Northwind Analytics" corpus (6 docs)
  questions.json    test questions (answerable + deliberately unanswerable)
eval/
  run_eval.py       critic-on vs critic-off, judged live
traces/             one JSON trace per request (git-ignored)
tests/              smoke tests (no API key required)
```

## Setup

Requires Python 3.11+ (this repo pins 3.12 via `uv`) and an **Anthropic API key
with billing enabled** — the system makes live API calls.

```bash
# 1. Install dependencies (uv manages the Python 3.12 venv)
uv sync

# 2. Provide your API key
cp .env.example .env
#   then edit .env and set ANTHROPIC_API_KEY=...
#   (or: export ANTHROPIC_API_KEY=...)

# 3. Build the local vector store from the sample corpus
uv run python -m critique_rag.ingest
```

> Using `pip` instead of `uv`? Create a 3.12 venv, then
> `pip install -e .` and run the same commands without the `uv run` prefix.

## Run a query

```bash
# Answerable — expect a grounded answer
uv run python -m critique_rag.cli "How long does the Pro plan retain raw event data?"

# Unanswerable — expect the abstain message after the retry loop fires
uv run python -m critique_rag.cli "What is Northwind Software's annual revenue?"

# Critic-off path (single retrieve->generate pass, no loop)
uv run python -m critique_rag.cli "How long does the Pro plan retain raw event data?" --no-critic
```

Each run prints the answer, the status (`grounded` / `abstained`), retries used,
and a path to a JSON trace.

## Reading a trace

Every request writes `traces/<timestamp>-<id>.json` logging each transition: the
query used at each step, the chunks retrieved (id, source, distance, preview),
the generated answer, and the critic verdict. On the unanswerable question you'll
see the loop fire — `retrieve → generate → critique → reformulate_query →
retrieve …` — until `retry_count` hits `max_retries`, then `abstain`. This is the
artifact to show in a demo/interview that the retry loop is real.

## Point it at your own documents

1. Drop your own `.md` files into `data/sample_docs/` (or change
   `SAMPLE_DOCS_DIR` in `config.py`).
2. Re-run `uv run python -m critique_rag.ingest`.
3. Edit `data/questions.json` with questions for your corpus (including at least
   one you know is unanswerable) and re-run the CLI or eval.

Other knobs live in `config.py`: `TOP_K`, `MAX_RETRIES`, and the model IDs.

## Evaluation (real numbers, not fabricated)

`eval/run_eval.py` runs every test question through **both** the critic-on and
critic-off paths, then has an **independent judge** (`claude-sonnet-4-6` —
deliberately a *different* model from the in-loop critic `claude-haiku-4-5`, so
the critic-on answers aren't trivially graded as grounded by the same model that
selected them) label each final answer as grounded or hallucinated, judging only
against the retrieved chunks. Abstentions count as not-a-hallucination.

```bash
uv run python eval/run_eval.py
```

It prints a per-question table and the hallucination rate for each path. The
metric is **critic-judged groundedness**, and every number is computed live by
that script. No results are written into this README — run the script to
generate them for your corpus.

## Limitations

- Groundedness is judged by an LLM, not by human-labeled ground truth. The eval
  measures whether answers are *supported by the retrieved context*, not whether
  they are objectively true.
- The sample corpus is fictional on purpose: it guarantees that a correct
  specific answer must come from retrieval rather than the model's training data,
  which is what makes the grounding check meaningful.
