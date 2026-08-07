"""Combine dense and BM25 rankings with Reciprocal Rank Fusion (RRF)."""

from typing import TypedDict

import config

from . import bm25_retrieve
from .ingest import ChunkMetadata
from .retrieve import retrieve

ChunkKey = str


class HybridHit(TypedDict):
    """One fused result with its component ranks and scores."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    similarity: float
    rrf_score: float
    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None


class ComponentScores(TypedDict):
    """Diagnostic information collected before creating a HybridHit."""

    dense_rank: int | None
    dense_score: float | None
    bm25_rank: int | None
    bm25_score: float | None


def _chunk_key(chunk: dict) -> ChunkKey:
    """Return the canonical identity shared by dense and BM25 retrieval."""
    return chunk["chunk_id"]


def retrieve_hybrid(
    question: str,
    k: int | None = None,
    candidates: int | None = None,
    doc_names: list[str] | None = None,
) -> list[HybridHit]:
    """Return the top-k results after fusing dense and BM25 ranks."""
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if k is None:
        k = config.TOP_K

    if candidates is None:
        candidates = config.HYBRID_CANDIDATES

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if candidates <= 0:
        raise ValueError("candidates must be a positive integer.")

    if candidates < k:
        raise ValueError("candidates must be greater than or equal to k.")

    dense_hits = retrieve(question, k=candidates, doc_names=doc_names)
    bm25_hits = bm25_retrieve.retrieve(
        question,
        k=candidates,
        doc_names=doc_names,
    )

    chunks_by_key: dict[ChunkKey, dict] = {}
    rrf_scores: dict[ChunkKey, float] = {}
    component_scores: dict[ChunkKey, ComponentScores] = {}

    for retrieval_method, rank_list in (
        ("dense", dense_hits),
        ("bm25", bm25_hits),
    ):
        for rank, chunk in enumerate(rank_list, start=1):
            key = _chunk_key(chunk)
            chunks_by_key[key] = chunk
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (config.RRF_K + rank)

            scores = component_scores.setdefault(
                key,
                {
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )
            if retrieval_method == "dense":
                scores["dense_rank"] = rank
                scores["dense_score"] = chunk["similarity"]
            else:
                scores["bm25_rank"] = rank
                scores["bm25_score"] = chunk["similarity"]

    fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    hits: list[HybridHit] = []

    for key, rrf_score in fused[:k]:
        chunk = chunks_by_key[key]
        scores = component_scores[key]
        hits.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "similarity": rrf_score,
                "rrf_score": rrf_score,
                **scores,
            }
        )

    return hits


def main() -> None:
    """Run a small hybrid-retrieval diagnostic."""
    question = "What are the geographies that Pepsico primarily operates in as of FY2022?"
    print(f"Question: {question}\n")

    for hit in retrieve_hybrid(question):
        meta = hit["metadata"]
        print(
            f"[{meta['company']} p{meta['page_number']}] "
            f"rrf_score={hit['rrf_score']:.4f} "
            f"dense_rank={hit['dense_rank']} "
            f"bm25_rank={hit['bm25_rank']}"
        )
        print(hit["text"][:150].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
