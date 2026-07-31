"""
Hybrid retrieval: fuse dense (bi-encoder) and BM25 (keyword) rankings with
Reciprocal Rank Fusion (RRF), instead of relying on either signal alone.

Why fuse instead of pick one: dense (Recall@5 34.5%) and BM25 (13.8%)
aren't just "one strictly better" -- they're measurably complementary.
BM25 uniquely hit 2 questions (financebench_id_01226, financebench_id_01009)
that dense missed entirely (docs/finrag_learning_report.md, Section 13) --
exact keyword matching succeeding where semantic embedding similarity
failed. The union of both methods' hits was 12/29 (41.4%), a real ceiling
neither method reaches alone. Reranking (Section 16/17) tried to improve
on dense with a smarter re-scorer and actively regressed it (20.7%) --
this is a structurally different idea: not "re-judge dense's own
candidates better," but "bring in a genuinely different signal."

Why RRF specifically, not averaging the two raw scores: dense's cosine
similarity (roughly 0-1) and BM25's raw score (unbounded, corpus- and
query-length-dependent) are not on comparable scales -- adding or
averaging them directly would let whichever score happens to have larger
magnitude dominate, for no principled reason (the exact trap noted in
bm25_retrieve.py's docstring: "only the resulting Recall@k numbers ...
are comparable, never the raw per-chunk scores"). RRF sidesteps this by
using ONLY each list's RANK POSITIONS, never the raw scores themselves:

    RRF_score(chunk) = sum over each ranking list L containing chunk of
                        1 / (RRF_K + rank_in_L(chunk))

A chunk ranked #1 in either list contributes close to 1/RRF_K; a chunk
appearing in BOTH lists gets credit from both, naturally boosting things
both signals agree on -- without the two methods' scales ever needing to
match. RRF_K=60 is the standard constant from the original RRF paper
(Cormack, Clarke, Buettcher, 2009) -- it dampens the impact of very top
ranks so one method's #1 doesn't completely dominate the fused ranking.
"""

import config
from . import bm25_retrieve
from .retrieve import retrieve


def _chunk_key(chunk: dict) -> tuple:
    """Identify the same underlying chunk across dense and BM25 result
    lists. Both retrieve over the exact same ingest.chunk_corpus() output
    (verified: bm25_retrieve.py builds its index directly from
    chunk_corpus(), same as index.py does for the Chroma collection), so
    (doc_name, page_index, text) is a stable, sufficient match key -- no
    need to plumb a separate chunk ID through either retriever."""
    meta = chunk["metadata"]
    return (meta["doc_name"], meta["page_index"], chunk["text"])


def retrieve_hybrid(
    question: str, k: int = config.TOP_K, candidates: int = config.HYBRID_CANDIDATES
) -> list[dict]:
    """Same {"text","metadata","similarity"} shape as retrieve() and
    bm25_retrieve.retrieve() -- drops straight into
    evaluate_retrieval.py's evaluate(retrieve_fn=...), same pattern as
    every other retriever in this project.

    "similarity" in the returned dicts is the RRF fused score -- yet
    another scale, not comparable to cosine similarity, BM25's raw score,
    or the cross-encoder's logits. Only meaningful for ranking within
    this method.
    """
    dense_hits = retrieve(question, k=candidates)
    bm25_hits = bm25_retrieve.retrieve(question, k=candidates)

    chunks_by_key: dict[tuple, dict] = {}
    rrf_scores: dict[tuple, float] = {}

    for rank_list in (dense_hits, bm25_hits):
        for rank, chunk in enumerate(rank_list, start=1):
            key = _chunk_key(chunk)
            chunks_by_key[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (config.RRF_K + rank)

    fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**chunks_by_key[key], "similarity": score} for key, score in fused[:k]]


if __name__ == "__main__":
    question = "What are the geographies that Pepsico primarily operates in as of FY2022?"
    print(f"Question: {question}\n")
    for hit in retrieve_hybrid(question):
        meta = hit["metadata"]
        print(f"[{meta['company']} p{meta['page_number']}] rrf_score={hit['similarity']:.4f}")
        print(hit["text"][:150].replace("\n", " "))
        print()
