"""
Reranking: a cross-encoder re-scores a wider dense-retrieval shortlist.

Why this exists: Section 15/16 of the learning report confirmed the same
pattern four separate times -- financial tables and ratio-heavy passages
consistently lose to generic, fluent prose in cosine-similarity space,
across every chunk size, and even when the gold passage's own document is
the only one being searched. bge-small (like most bi-encoders) embeds the
query and each passage SEPARATELY, then compares vectors -- it never
actually looks at the query and a specific passage together. A
cross-encoder does: it takes (query, passage) as one joint input and
outputs a single relevance score, so it can weigh things a bi-encoder
structurally cannot, like "this passage is mostly bare numbers, but
they're exactly the numbers this query is asking about."

Why not just use a cross-encoder for everything, then? Cost. A
cross-encoder has to run a full forward pass per (query, passage) PAIR --
comparing one query against the whole corpus would mean thousands of
forward passes per question. A bi-encoder pre-embeds every passage once,
offline, so a query only needs ONE new embedding plus a fast vector
lookup. The standard pattern (used here): let the cheap bi-encoder narrow
the whole corpus down to a shortlist (RERANK_CANDIDATES, e.g. 20), then
spend the cross-encoder's extra accuracy only on reordering that much
smaller set.
"""

from sentence_transformers import CrossEncoder

import config

# Lazily-loaded singleton, same reasoning as retrieve.py's _model/_collection
# -- rerank() may be called once per eval-set question, no need to reload
# the model each time.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(
            config.RERANK_MODEL,
            revision=config.RERANK_MODEL_REVISION,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
        )
    return _model


def rerank(question: str, candidates: list[dict], k: int) -> list[dict]:
    """Re-score `candidates` (each {"text","metadata","similarity"}, as
    returned by retrieve()) against `question` with a cross-encoder, and
    return the top-k reordered by the new score.

    The cross-encoder's raw output is NOT a cosine similarity -- it's an
    unbounded relevance logit (same caveat as bm25_retrieve.py's score:
    not comparable across methods). Still stored under the "similarity"
    key so callers (evaluate_retrieval.py) don't need to know reranking
    happened at all -- same "functions as values" pattern already used
    for the BM25 baseline.
    """
    model = _get_model()
    pairs = [[question, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [{**c, "similarity": float(score)} for c, score in reranked[:k]]


if __name__ == "__main__":
    from .retrieve import retrieve

    question = (
        "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?"
    )
    candidates = retrieve(question, k=config.RERANK_CANDIDATES)
    print(f"Dense top-5 (of {len(candidates)} candidates considered):")
    for c in candidates[:5]:
        meta = c["metadata"]
        print(f"  [{meta['company']} p{meta['page_number']}] sim={c['similarity']:.3f}")

    reranked = rerank(question, candidates, k=5)
    print("\nReranked top-5:")
    for c in reranked:
        meta = c["metadata"]
        print(f"  [{meta['company']} p{meta['page_number']}] score={c['similarity']:.3f}")
