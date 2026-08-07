"""
Stage 3b — BM25 retrieval: question -> top-k relevant chunks, keyword-based.
"""

import re

from rank_bm25 import BM25Okapi

import config

from .corpus_store import load_corpus, load_manifest
from .ingest import DocumentChunk

_bm25: BM25Okapi | None = None
_chunks: list[DocumentChunk] | None = None
_corpus_fingerprint: str | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_index() -> tuple[BM25Okapi, list[DocumentChunk]]:
    """Build the in-memory BM25 index on first use."""
    global _bm25, _chunks, _corpus_fingerprint

    manifest = load_manifest()
    current_fingerprint = manifest["corpus_fingerprint"]

    if _bm25 is None or _chunks is None or _corpus_fingerprint != current_fingerprint:
        _, chunks = load_corpus()

        if not chunks:
            raise ValueError("No chunks available for indexing.")

        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]

        if not any(tokenized_corpus):
            raise ValueError(
                "Tokenized corpus is empty. Check the chunking and tokenization process."
            )

        bm25 = BM25Okapi(tokenized_corpus)

        # Update the two cache values together after construction succeeds.
        _chunks = chunks
        _bm25 = bm25
        _corpus_fingerprint = current_fingerprint

    return _bm25, _chunks


def reset_index_cache() -> None:
    """Force the next retrieve() call to rebuild the BM25 index from
    scratch."""
    global _bm25, _chunks, _corpus_fingerprint
    _bm25 = None
    _chunks = None
    _corpus_fingerprint = None


def retrieve(
    question: str,
    k: int | None = None,
    doc_names: list[str] | None = None,
) -> list[dict]:
    """Return the top-k chunks most relevant to `question` by BM25 score,
    searched across the whole corpus.
    """
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if k is None:
        k = config.TOP_K

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if doc_names:
        valid_doc_names = set(config.CORPUS_DOCS)
        unknown_doc_names = sorted(set(doc_names) - valid_doc_names)

        if unknown_doc_names:
            raise ValueError(
                f"Unknown doc_names: {unknown_doc_names}. "
                f"Valid doc_names are: {sorted(valid_doc_names)}"
            )

    query_tokens = _tokenize(question)

    if not query_tokens:
        return []

    bm25, chunks = _get_index()
    scores = bm25.get_scores(query_tokens)

    allowed_doc_names = set(doc_names) if doc_names else None

    candidate_indices = [
        index
        for index, chunk in enumerate(chunks)
        if (allowed_doc_names is None or chunk["metadata"]["doc_name"] in allowed_doc_names)
        and scores[index] > 0
    ]

    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    return [
        {
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            # This is a raw BM25 score, not cosine similarity.
            "similarity": float(scores[index]),
        }
        for index in ranked_indices
    ]


def main() -> None:
    """Run a small BM25 retrieval diagnostic."""
    question = "What are the major products and services that AMD sells as of FY22?"

    print(f"Question: {question}\n")

    for hit in retrieve(question):
        metadata = hit["metadata"]
        print(
            f"[{metadata['doc_name']} "
            f"p{metadata['page_number']} "
            f"{metadata['source']}] "
            f"bm25_score={hit['similarity']:.3f}"
        )
        print(hit["text"][:200])
        print()


if __name__ == "__main__":
    main()
