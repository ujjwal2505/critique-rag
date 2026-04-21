"""Honest critic-on vs critic-off comparison.

For each test question we run the graph twice:
  - critic OFF: a single retrieve -> generate pass (no critique, no retries).
  - critic ON:  the full self-critiquing loop (may answer or abstain).

An INDEPENDENT judge (a different model from the in-loop critic) then labels
each final answer as grounded or not, judging only against the chunks that were
actually retrieved. An abstention counts as not-a-hallucination.

The headline metric is "critic-judged groundedness". All numbers are computed
live here — nothing is hard-coded. Requires ANTHROPIC_API_KEY.

Run: ``uv run python eval/run_eval.py``
"""

import json

from critique_rag import llm
from critique_rag.cli import run_query
from critique_rag.config import CRITIC_MODEL, JUDGE_MODEL, MAX_RETRIES, QUESTIONS_FILE


def evaluate() -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rows = []
    for item in questions:
        question = item["q"]

        # critic OFF — judge the answer it would have returned with no critique.
        off = run_query(question, critic_enabled=False, max_retries=MAX_RETRIES)
        off_verdict = llm.judge_grounded(
            question, off["final_answer"], off["retrieved_chunks"]
        )

        # critic ON — abstentions are correct refusals, not hallucinations.
        on = run_query(question, critic_enabled=True, max_retries=MAX_RETRIES)
        print(f"{on}")
        abstained = on.get("status") == "abstained"
        if abstained:
            on_grounded = True
        else:
            on_grounded = llm.judge_grounded(
                question, on["final_answer"], on["retrieved_chunks"]
            ).grounded

        rows.append(
            {
                "q": question,
                "type": item.get("type"),
                "off_grounded": off_verdict.grounded,
                "on_grounded": on_grounded,
                "on_abstained": abstained,
                "on_retries": on.get("retry_count", 0),
            }
        )
    return rows


def main() -> None:
    rows = evaluate()
    n = len(rows)
    off_hallucinations = sum(1 for r in rows if not r["off_grounded"])
    on_hallucinations = sum(1 for r in rows if not r["on_grounded"])

    header = f"{'question':52}{'type':14}{'off':>6}{'on':>6}{'abst':>6}{'retry':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['q'][:50]:52}{str(r['type']):14}"
            f"{('OK' if r['off_grounded'] else 'HALL'):>6}"
            f"{('OK' if r['on_grounded'] else 'HALL'):>6}"
            f"{('Y' if r['on_abstained'] else '-'):>6}"
            f"{r['on_retries']:>6}"
        )

    print(f"\nN = {n}")
    print(
        f"critic OFF hallucination rate: {off_hallucinations}/{n} "
        f"= {off_hallucinations / n:.0%}"
    )
    print(
        f"critic ON  hallucination rate: {on_hallucinations}/{n} "
        f"= {on_hallucinations / n:.0%}"
    )
    print(
        "\nMetric: critic-judged groundedness. Judge model = "
        f"{JUDGE_MODEL} (independent of in-loop critic {CRITIC_MODEL}). "
        "Abstentions count as not-hallucinated. Numbers are produced by this run."
    )


if __name__ == "__main__":
    main()
