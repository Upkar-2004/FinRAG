"""
Reranking evaluation: same evaluate() harness as evaluate_retrieval.py and
evaluate_bm25.py, pointed at src.finrag.retrieve.retrieve_with_rerank
instead of plain dense retrieval.

Whole point: get a Recall@k number for dense-then-reranked search, on the
exact same eval set, same hit-scoring rule, same k as the plain-dense
baseline (Recall@5: 10/29, 34.5%) and BM25 baseline (Recall@5: 4/29,
13.8%) -- see docs/finrag_learning_report.md. Only measuring it the same
way tells us whether reranking actually rescues any of the four
diagnosed cases (Section 15/16: financial tables and ratios losing to
generic prose), or whether -- like chunk-size tuning, doc-scoped
filtering, and table serialization before it -- it's another intuitive
fix that doesn't move the number.

Runs entirely locally (cross-encoder, no Groq calls) -- safe to run
regardless of API quota state.

Run:
    python3 scripts/evaluate_rerank.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.retrieve import retrieve_with_rerank  # noqa: E402
from scripts.evaluate_retrieval import evaluate, summarize  # noqa: E402

if __name__ == "__main__":
    results = evaluate(retrieve_fn=retrieve_with_rerank, k=config.TOP_K)
    for r in results:
        tag = "HIT " if r["hit"] else "MISS"
        print(f"[{tag}] {r['id']:24s} gold={r['gold_pages']!s:14s} best_rerank_score={r['best_similarity']:.3f}")
    summarize(results, k=config.TOP_K)
