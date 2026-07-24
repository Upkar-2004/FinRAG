"""
Stage 3b — BM25 retrieval: question -> top-k relevant chunks, keyword-based.

A classical (non-neural) baseline: ranks chunks by literal term overlap
with the question, weighted by how rare/distinctive each term is across
the corpus (rank_bm25's BM25Okapi implementation of the BM25 algorithm).
No embeddings, no notion of meaning, no model weights to load — just
tokenizing and counting.

Built specifically to be evaluated with the SAME scripts/evaluate_retrieval.py
harness used for dense retrieval (src/finrag/retrieve.py), by matching its
return shape exactly: a list of {"text", "metadata", "similarity"} dicts.
That's what makes Recall@k directly comparable between the two retrieval
strategies on the same eval set — the whole reason this file exists.

IMPORTANT: the "similarity" field here is BM25's raw relevance score, NOT
a cosine similarity. It is unbounded (not a 0-1 scale) and not comparable
in magnitude to src.finrag.retrieve's similarity values — only the
resulting Recall@k numbers from the two retrievers are comparable, never
the raw per-chunk scores against each other.
"""

import re

from rank_bm25 import BM25Okapi

import config
from .ingest import chunk_corpus

# Lazily-built, module-level singleton — same reasoning as retrieve.py's
# _model/_collection caching: building the BM25 index re-tokenizes and
# re-counts every chunk in the corpus, wasted work to repeat on every call
# when the eval harness calls retrieve() 29 times in a row.
_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. No stemming, no
    stopword removal — a deliberately simple baseline tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_index() -> tuple[BM25Okapi, list[dict]]:
    global _bm25, _chunks
    if _bm25 is None:
        _chunks = chunk_corpus()
        tokenized_corpus = [_tokenize(c["text"]) for c in _chunks]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _chunks


def reset_index_cache() -> None:
    """Force the next retrieve() call to rebuild the BM25 index from
    scratch — needed if chunk_corpus() would now produce different chunks
    (e.g. config.CHUNK_SIZE changed) since the last call in this process."""
    global _bm25, _chunks
    _bm25 = None
    _chunks = None


def retrieve(question: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks most relevant to `question` by BM25 score,
    searched across the whole corpus (no doc_name filtering — same
    whole-corpus principle as src.finrag.retrieve.retrieve()).

    Each result: {"text": str, "metadata": dict, "similarity": float}.
    "similarity" here is BM25's raw score — see module docstring.
    """
    bm25, chunks = _get_index()
    scores = bm25.get_scores(_tokenize(question))

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "text": chunks[i]["text"],
            "metadata": chunks[i]["metadata"],
            "similarity": float(scores[i]),
        }
        for i in ranked_indices
    ]


if __name__ == "__main__":
    question = "What are the major products and services that AMD sells as of FY22?"
    print(f"Question: {question}\n")
    for hit in retrieve(question):
        meta = hit["metadata"]
        print(f"[{meta['doc_name']} p{meta['page_number']}] bm25_score={hit['similarity']:.3f}")
        print(hit["text"][:200])
        print()
