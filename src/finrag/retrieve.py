"""
Stage 3 — Retrieval: question -> top-k relevant chunks.
"""

from typing import TypedDict

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

import config

from .corpus_store import load_manifest
from .index import COLLECTION_NAME
from .ingest import ChunkMetadata

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class RetrievalHit(TypedDict):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    similarity: float


_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            revision=config.EMBEDDING_MODEL_REVISION,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
        )
    return _model


def _get_collection() -> chromadb.Collection:

    global _collection

    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))

        try:
            _collection = client.get_collection(
                COLLECTION_NAME,
                embedding_function=None,
            )
        except NotFoundError as error:
            raise RuntimeError(
                "The FinRAG vector index was not found. Run `python -m src.finrag.index` first."
            ) from error

    return _collection


def reset_collection_cache() -> None:

    global _collection
    _collection = None


def _validate_collection(collection: chromadb.Collection) -> None:
    """Ensure Chroma was built from the current corpus and embedding model."""
    manifest = load_manifest()
    metadata = collection.metadata or {}
    expected = {
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_revision": config.EMBEDDING_MODEL_REVISION,
    }

    mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatched:
        raise RuntimeError(
            "The Chroma index does not match the current corpus or embedding model. "
            "Rebuild it with `python -m src.finrag.index`."
        )


def retrieve(
    question: str, k: int | None = None, doc_names: list[str] | None = None
) -> list[RetrievalHit]:
    """Return the nearest dense-retrieval chunks for a question"""

    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty")

    if k is None:
        k = config.TOP_K

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if doc_names:
        valid_doc_names = set(config.CORPUS_DOCS)
        unknown_doc_names = sorted(set(doc_names) - valid_doc_names)
        if unknown_doc_names:
            raise ValueError(
                f"Unknown doc_names: {unknown_doc_names}. "
                f"Valid doc_names are: {sorted(valid_doc_names)}"
            )

    model = _get_model()
    collection = _get_collection()
    _validate_collection(collection)

    collection_count = collection.count()
    if collection_count <= 0:
        raise RuntimeError("The FinRAG vector index is empty. Rebuild the index.")

    query_embedding = model.encode(
        [QUERY_PREFIX + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    where = {"doc_name": {"$in": doc_names}} if doc_names else None
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection_count),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[RetrievalHit] = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "similarity": 1 - distance,
            }
        )
    return hits


def retrieve_with_rerank(
    question: str, k: int | None = None, doc_names: list[str] | None = None
) -> list[RetrievalHit]:

    from .rerank import rerank

    if k is None:
        k = config.TOP_K

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if k > config.RERANK_CANDIDATES:
        raise ValueError("k cannot exceed RERANK_CANDIDATES when reranking.")

    candidates = retrieve(question, k=config.RERANK_CANDIDATES, doc_names=doc_names)

    if not candidates:
        return []

    return rerank(question, candidates, k=k)


def retrieve_with_expansion(
    question: str, k: int | None = None, doc_names: list[str] | None = None
) -> list[RetrievalHit]:
    """Expand known financial terms before dense retrieval."""
    from .query_expand import expand_query

    return retrieve(
        expand_query(question),
        k=k,
        doc_names=doc_names,
    )


if __name__ == "__main__":
    question = "What are the major products and services that AMD sells as of FY22?"
    print(f"Question: {question}\n")
    for hit in retrieve(question):
        meta = hit["metadata"]
        print(f"[{meta['doc_name']} p{meta['page_number']}] similarity={hit['similarity']:.3f}")
        print(hit["text"][:200])
        print()
