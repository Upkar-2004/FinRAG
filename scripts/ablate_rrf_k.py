"""
RRF_K ablation: does a smaller damping constant let hybrid retrieval
actually capture BM25's unique correct hits, or is RRF structurally the
wrong fusion method here regardless of tuning?

Background: a live run with the default RRF_K=60 (the standard value
from the original RRF paper, tuned for web search) showed hybrid
retrieval regressing below dense-only (24.1% vs 34.5%) AND failing to
rescue financebench_id_01226 and financebench_id_01009 -- the exact two
questions that motivated building hybrid retrieval, where BM25 alone is
confidently correct and dense has zero signal at all. Diagnosis: RRF
rewards CONSENSUS (chunks both methods rank, even weakly) over a single
method's high-confidence pick, because 1/(K+rank) summed across two
mediocre ranks can exceed 1/(K+rank) from one strong rank when K is
large. A smaller K makes rank position matter more steeply, which should
narrow or close that gap.

Unlike scripts/ablate_chunking.py, this needs NO index rebuild or cache
reset between trials -- RRF_K only affects the fusion step, applied
AFTER dense and BM25 have already returned their independent candidate
lists. Mutates config.RRF_K in memory only; the file itself is untouched.

Run:
    python3 scripts/ablate_rrf_k.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.hybrid_retrieve import retrieve_hybrid  # noqa: E402
from scripts.evaluate_retrieval import evaluate  # noqa: E402

# The two questions RRF was specifically supposed to rescue -- tracked
# individually at each K, not just the aggregate Recall@5.
WATCH_IDS = {"financebench_id_01226", "financebench_id_01009"}

K_VALUES = [1, 5, 10, 30, 60]

if __name__ == "__main__":
    print(f"{'RRF_K':>6} | {'Recall@5':>10} | watched questions")
    print("-" * 55)
    for k_value in K_VALUES:
        config.RRF_K = k_value
        results = evaluate(retrieve_fn=retrieve_hybrid, k=config.TOP_K)
        n_hits = sum(r["hit"] for r in results)
        watched_status = {r["id"]: ("HIT" if r["hit"] else "miss") for r in results if r["id"] in WATCH_IDS}
        watched_str = ", ".join(f"{qid.split('_')[-1]}={status}" for qid, status in watched_status.items())
        print(f"{k_value:>6} | {n_hits}/29 ({n_hits/29:.1%}) | {watched_str}")
