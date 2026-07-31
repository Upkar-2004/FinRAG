"""
Generation evaluation: how does generate_answer() perform end-to-end (real
retrieve() + a real Groq call per question), across the full 29-question
eval set -- not just the two hand-picked spot checks in the learning
report's Section 16.

Deliberately narrow in scope. Automated answer-CORRECTNESS scoring (exact
match against eval_set.jsonl's gold `answer` text, or an LLM-as-judge) is
NOT attempted here. Both have real, well-known failure modes -- a correct
answer can be worded completely differently from a terse gold reference
(seen directly with the Boeing legal-battles question), and an LLM judge
just introduces a second unverified model's bias instead of removing the
problem. This is flagged as a real risk in the learning report's roadmap,
not something to rush past.

What IS measured here is fully mechanical, not judgment-based:
- citation_format_ok : does the answer contain at least one
  "(Company, page N)"-shaped citation at all?
- citation_grounded  : does every page number the answer cites actually
  belong to a chunk retrieve() handed the model? (Catches a fabricated
  citation to a page the model was never given -- a distinct failure from
  just being wrong.) Checked on page NUMBER only, not company name --
  the model paraphrases company names in citations (e.g. writes "AMD"
  when the context block was labeled "AMD (semiconductors)"), so a
  strict string match on company would produce false negatives.
- cited_gold_page    : for retrieval HITS specifically, did the model's
  citation actually include the real gold page it was handed, or did it
  ignore the right passage even though retrieval found it?
- abstained          : a keyword heuristic for whether the answer says it
  lacks enough information rather than answering confidently. A heuristic,
  not a classifier -- treat it as directional, not exact.
- used_tool          : whether the calculator tool suite fired at all.

Metrics-generated questions (numeric answers) are printed in full instead
of auto-scored -- verifying a computed number against a real gold value
needs the gold value worked out by hand first, not attempted here.

Any row matching a suspicious pattern (a fabricated citation, or a
retrieval miss where the model neither cited anything nor abstained --
see _flag_reason()) also gets its full raw answer text printed in the
summary, so a boolean flag alone doesn't hide what actually happened.

A RateLimitError stops the run early and returns whatever was already
collected, instead of losing all prior results -- Groq's daily token
quota was hit mid-run building this script, discarding a completed
question's results the first time, which is why this exists. Any other
per-question failure (e.g. exhausting generate.py's tool-call retries)
is recorded as an error row instead of crashing the whole batch.

Run (makes ~29 real Groq API calls; takes a couple minutes and will hit
your Groq quota -- if you get a rate-limit error, wait a bit and rerun):
    python3 scripts/evaluate_generation.py
"""

import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from groq import RateLimitError  # noqa: E402
from src.finrag.generate import generate_answer  # noqa: E402
from src.finrag.retrieve import retrieve_with_expansion as retrieve  # noqa: E402

CITATION_RE = re.compile(r"\(([^,()]+),\s*page\s*(\d+)\)", re.IGNORECASE)
ABSTAIN_PHRASES = [
    "does not contain",
    "do not contain",
    "not contain enough",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "not enough information",
    "insufficient information",
    "context does not",
    "context doesn't",
    "no information",
]


def _extract_citations(answer: str) -> list[tuple[str, int]]:
    return [(company.strip(), int(page)) for company, page in CITATION_RE.findall(answer or "")]


def _looks_like_abstention(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(phrase in lower for phrase in ABSTAIN_PHRASES)


def evaluate() -> list[dict]:
    eval_path = config.EVAL_DIR / "eval_set.jsonl"
    eval_set = [json.loads(line) for line in eval_path.open()]

    results = []
    for row in eval_set:
        chunks = retrieve(row["question"], k=config.TOP_K)
        retrieved_pages = {c["metadata"]["page_number"] for c in chunks}
        retrieval_hit = any(
            c["metadata"]["doc_name"] == row["doc_name"] and c["metadata"]["page_index"] in row["gold_pages"]
            for c in chunks
        )
        gold_page_numbers = {p + 1 for p in row["gold_pages"]}  # page_index -> page_number

        try:
            out = generate_answer(row["question"], chunks)
        except RateLimitError as e:
            # Quota exhaustion is global, not specific to this question --
            # every remaining call would fail the same way immediately.
            # Stop here and return what's already been collected instead
            # of burning through (and reporting failures for) every
            # remaining question. Learned this the hard way: without this,
            # a single RateLimitError killed a whole 29-question run and
            # discarded every result gathered before it, since evaluate()
            # only returns `results` at the very end.
            print(f"\n[rate limit hit on {row['id']} -- stopping early. {len(results)}/{len(eval_set)} completed.]")
            print(f"  {e}")
            break
        except Exception as e:
            # Any other per-question failure (e.g. the model exhausted all
            # MAX_TOOL_ATTEMPTS retries with a malformed tool call) shouldn't
            # take down the other 28 questions -- record it and move on.
            print(f"[{row['id']} failed: {e}]")
            results.append(
                {
                    "id": row["id"],
                    "question_type": row["question_type"],
                    "retrieval_hit": retrieval_hit,
                    "used_tool": False,
                    "citation_format_ok": False,
                    "citation_grounded": False,
                    "cited_gold_page": False,
                    "abstained": False,
                    "answer": f"[ERROR: {e}]",
                }
            )
            continue

        answer = out["answer"] or ""
        citations = _extract_citations(answer)

        citation_format_ok = len(citations) > 0
        citation_grounded = citation_format_ok and all(page in retrieved_pages for _, page in citations)
        cited_gold_page = retrieval_hit and any(page in gold_page_numbers for _, page in citations)

        results.append(
            {
                "id": row["id"],
                "question_type": row["question_type"],
                "retrieval_hit": retrieval_hit,
                "used_tool": out["used_tool"],
                "citation_format_ok": citation_format_ok,
                "citation_grounded": citation_grounded,
                "cited_gold_page": cited_gold_page,
                "abstained": _looks_like_abstention(answer),
                "answer": answer,
            }
        )
    return results


def _flag_reason(r: dict) -> str | None:
    """Worth printing this row's full answer text for manual reading, or not.

    Two patterns specifically: a fabricated citation (cited a page that
    wasn't actually in the model's context -- a more serious failure than
    just being wrong), and a "silent miss" -- retrieval failed, and the
    model neither cited anything nor tripped the abstention heuristic.
    That second pattern is what financebench_id_00499 hit on a live run:
    citation_format_ok=False AND abstained=False on a retrieval MISS,
    which SYSTEM_PROMPT's rules shouldn't allow (it should always either
    cite or explicitly say it doesn't know) -- worth reading the actual
    text to see whether it quietly answered from outside knowledge
    instead.
    """
    if r["citation_format_ok"] and not r["citation_grounded"]:
        return "fabricated citation"
    if not r["retrieval_hit"] and not r["citation_format_ok"] and not r["abstained"]:
        return "miss, no citation, no abstention"
    return None


def summarize(results: list[dict]) -> None:
    n = len(results)
    print("\n" + "=" * 60)
    print(f"GENERATION EVAL COMPLETE (n={n})")
    print("=" * 60)
    print(f"Citation format present:                  {sum(r['citation_format_ok'] for r in results)}/{n}")
    print(f"Citations grounded (no fabricated pages):  {sum(r['citation_grounded'] for r in results)}/{n}")

    hits = [r for r in results if r["retrieval_hit"]]
    misses = [r for r in results if not r["retrieval_hit"]]
    if hits:
        print(
            f"\nOf {len(hits)} retrieval HITS: model cited the actual gold page in "
            f"{sum(r['cited_gold_page'] for r in hits)}/{len(hits)}"
        )
    if misses:
        print(
            f"Of {len(misses)} retrieval MISSES: model abstained (heuristic) in "
            f"{sum(r['abstained'] for r in misses)}/{len(misses)}"
        )

    flagged = [(r, _flag_reason(r)) for r in results if _flag_reason(r) is not None]
    if flagged:
        print(f"\n{len(flagged)} flagged row(s) -- full answer text, worth reading directly:")
        for r, reason in flagged:
            print(f"  [{r['id']}] ({reason})")
            print(f"    {r['answer']}")

    print(f"\nCalculator tool used: {sum(r['used_tool'] for r in results)}/{n}")

    metrics_generated = [r for r in results if r["question_type"] == "metrics-generated"]
    if metrics_generated:
        print("\nmetrics-generated answers (print only -- not auto-scored, see module docstring):")
        for r in metrics_generated:
            print(f"  [{r['id']}] used_tool={r['used_tool']}")
            print(f"    {r['answer']}")

    print(
        "\nNote: this does NOT score whether an answer's CONTENT is factually "
        "correct -- that still needs manual review (learning report, Sections 11 and 16)."
    )


if __name__ == "__main__":
    results = evaluate()
    for r in results:
        tag = "HIT " if r["retrieval_hit"] else "MISS"
        print(
            f"[{tag}] {r['id']:24s} cite_ok={str(r['citation_format_ok']):5} "
            f"grounded={str(r['citation_grounded']):5} tool={str(r['used_tool']):5}"
        )
    summarize(results)
