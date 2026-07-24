"""
Day 2 — BM25 baseline: same evaluate() harness as evaluate_retrieval.py,
pointed at src/finrag/bm25_retrieve.py instead of dense retrieval.

Whole point: get a Recall@k number for keyword-only search on the exact
same eval set, same hit-scoring rule, same k as the dense baseline
(Recall@5: 10/29, 34.5% — see docs/finrag_learning_report.md). Only once
we have both numbers, measured the same way, do we know whether dense
embeddings are actually earning their keep on this corpus, or whether a
much simpler, cheaper, keyword-only method does just as well or better.

Run:
    python3 scripts/evaluate_bm25.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.bm25_retrieve import retrieve as bm25_retrieve  # noqa: E402
from scripts.evaluate_retrieval import evaluate, summarize  # noqa: E402

if __name__ == "__main__":
    results = evaluate(retrieve_fn=bm25_retrieve, k=config.TOP_K)
    for r in results:
        tag = "HIT " if r["hit"] else "MISS"
        print(f"[{tag}] {r['id']:24s} gold={r['gold_pages']!s:14s} best_bm25={r['best_similarity']:.3f}")
    summarize(results, k=config.TOP_K)
