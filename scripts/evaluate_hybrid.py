"""
Hybrid retrieval evaluation: same evaluate() harness, pointed at
src.finrag.hybrid_retrieve.retrieve_hybrid (RRF-fused dense + BM25)
instead of either method alone.

Compares directly against what's already measured: dense-only (34.5%),
BM25-only (13.8%), reranked (20.7%, a regression below dense-only --
see docs/finrag_learning_report.md, Section 16/17). BM25 and dense's hit
sets were shown to be complementary, not just "one strictly better" --
BM25 alone found 2 questions dense missed entirely (Section 13). This
tells us whether RRF fusion actually captures that complementary signal
in a single retriever, or whether -- like reranking -- an intuitively
promising fix doesn't move the number.

Runs entirely locally (dense + BM25, no Groq calls) -- safe regardless
of API quota state.

Run:
    python3 scripts/evaluate_hybrid.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.hybrid_retrieve import retrieve_hybrid  # noqa: E402
from scripts.evaluate_retrieval import evaluate, summarize  # noqa: E402

if __name__ == "__main__":
    results = evaluate(retrieve_fn=retrieve_hybrid, k=config.TOP_K)
    for r in results:
        tag = "HIT " if r["hit"] else "MISS"
        print(f"[{tag}] {r['id']:24s} gold={r['gold_pages']!s:14s} best_rrf_score={r['best_similarity']:.4f}")
    summarize(results, k=config.TOP_K)
