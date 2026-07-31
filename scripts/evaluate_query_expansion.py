"""
Query expansion evaluation: same evaluate() harness, pointed at
src.finrag.retrieve.retrieve_with_expansion instead of plain dense
retrieval.

Zero-cost proof-of-concept for the query-rewriting/HyDE direction flagged
in Section 17 as the next genuinely different lever -- everything tried
before (chunk size, doc filtering, table serialization, reranking,
hybrid RRF) operated on the document side; this is the first thing that
touches the QUERY itself. A 2-question manual check already showed a
dramatic rank change on AMD's quick-ratio question (gold page not in the
top-20 at all -> rank 5) -- this runs the same idea across the full
29-question eval set to get a real, defensible Recall@5 number instead
of a spot check.

Runs entirely locally (no Groq calls) -- safe regardless of API quota.

Run:
    python3 scripts/evaluate_query_expansion.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.retrieve import retrieve_with_expansion  # noqa: E402
from scripts.evaluate_retrieval import evaluate, summarize  # noqa: E402

if __name__ == "__main__":
    results = evaluate(retrieve_fn=retrieve_with_expansion, k=config.TOP_K)
    for r in results:
        tag = "HIT " if r["hit"] else "MISS"
        print(f"[{tag}] {r['id']:24s} gold={r['gold_pages']!s:14s} best_sim={r['best_similarity']:.3f}")
    summarize(results, k=config.TOP_K)
