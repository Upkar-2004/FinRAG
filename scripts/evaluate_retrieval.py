"""
Day 2 — Retrieval evaluation: hit@k / Recall@k over the held-out eval set.

For each of the 29 labeled questions, retrieve top-k chunks from the whole
corpus (no doc filter — see src/finrag/retrieve.py) and check whether any
retrieved chunk is BOTH from the right document AND on one of its
gold_pages (page_index alone isn't enough — it resets to 0 per document,
so a coincidental page-number match against the WRONG company's chunk
would otherwise silently count as a hit).

Reports per-question hit/miss, aggregate Recall@k, a breakdown by
question_type, and the best-match similarity split by hit vs miss — real
data to calibrate config.MIN_RELEVANCE_SCORE with later, instead of
guessing.

Run once data/chroma/ exists (i.e. after running src/finrag/index.py):
    python3 scripts/evaluate_retrieval.py
"""

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.retrieve import retrieve  # noqa: E402


def evaluate(retrieve_fn=retrieve, k: int = config.TOP_K) -> list[dict]:
    """Run retrieval for every eval question and score it against gold_pages.

    retrieve_fn: any function shaped like (question: str, k: int) ->
    list[{"text", "metadata", "similarity"}]. Defaults to dense retrieval
    (src.finrag.retrieve.retrieve), but src.finrag.bm25_retrieve.retrieve
    can be passed instead to score a completely different retrieval
    strategy with this exact same hit-scoring logic, on the exact same
    eval set — that's what makes the two Recall@k numbers comparable.
    """
    eval_path = config.EVAL_DIR / "eval_set.jsonl"
    eval_set = [json.loads(line) for line in eval_path.open()]

    results = []
    for row in eval_set:
        hits = retrieve_fn(row["question"], k=k)
        # A hit requires BOTH the right document AND a gold page — page_index
        # alone is not globally unique (it resets to 0 per document), so
        # checking it without doc_name could score a same-numbered page from
        # a completely different company as a false hit.
        hit = any(
            h["metadata"]["doc_name"] == row["doc_name"]
            and h["metadata"]["page_index"] in row["gold_pages"]
            for h in hits
        )
        results.append(
            {
                "id": row["id"],
                "question_type": row["question_type"],
                "gold_pages": row["gold_pages"],
                "hit": hit,
                "best_similarity": max((h["similarity"] for h in hits), default=0.0),
            }
        )
    return results


def summarize(results: list[dict], k: int) -> None:
    n = len(results)
    n_hits = sum(r["hit"] for r in results)

    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL COMPLETE (k={k}, whole-corpus, no doc filter)")
    print("=" * 60)
    print(f"Recall@{k}: {n_hits}/{n} ({n_hits / n:.1%})")

    print("\nBy question_type:")
    for qt in sorted({r["question_type"] for r in results}):
        subset = [r for r in results if r["question_type"] == qt]
        sub_hits = sum(r["hit"] for r in subset)
        print(f"  {qt:20s}: {sub_hits}/{len(subset)} ({sub_hits / len(subset):.1%})")

    hit_sims = [r["best_similarity"] for r in results if r["hit"]]
    miss_sims = [r["best_similarity"] for r in results if not r["hit"]]
    print("\nBest-match similarity (for calibrating MIN_RELEVANCE_SCORE later):")
    if hit_sims:
        print(f"  hits   -> min={min(hit_sims):.3f}  mean={sum(hit_sims)/len(hit_sims):.3f}  max={max(hit_sims):.3f}")
    if miss_sims:
        print(f"  misses -> min={min(miss_sims):.3f}  mean={sum(miss_sims)/len(miss_sims):.3f}  max={max(miss_sims):.3f}")


if __name__ == "__main__":
    results = evaluate()
    for r in results:
        tag = "HIT " if r["hit"] else "MISS"
        print(f"[{tag}] {r['id']:24s} gold={r['gold_pages']!s:14s} best_sim={r['best_similarity']:.3f}")
    summarize(results, k=config.TOP_K)
