"""
Stage 2 — Indexing: chunks -> embeddings -> persisted Chroma collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config

from .corpus_store import load_corpus
from .ingest import DocumentChunk

COLLECTION_NAME = "finrag_chunks"


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Encode a list of strings into normalized vectors."""
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def build_index(
    chunks: list[DocumentChunk],
    corpus_fingerprint: str | None = None,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """Embed all chunks and persist them to a fresh Chroma collection."""
    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")

    texts = [chunk["text"] for chunk in chunks]
    blank_text_positions = [position for position, text in enumerate(texts) if not text.strip()]
    if blank_text_positions:
        raise ValueError(f"Cannot index blank chunk text at positions {blank_text_positions[:10]}.")

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Chunk ID generation produced duplicate IDs.")

    metadatas = [chunk["metadata"] for chunk in chunks]

    # Finish model loading and embedding before replacing the existing index.
    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        revision=config.EMBEDDING_MODEL_REVISION,
        local_files_only=config.HF_LOCAL_FILES_ONLY,
    )
    embeddings = embed_texts(model, texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(chunks)}, received {len(embeddings)}."
        )

    embedding_dimension = model.get_sentence_embedding_dimension()
    if embedding_dimension is None:
        raise RuntimeError("The embedding model did not report its output dimension.")

    invalid_dimensions = [
        position
        for position, embedding in enumerate(embeddings)
        if len(embedding) != embedding_dimension
    ]
    if invalid_dimensions:
        raise RuntimeError(
            f"Unexpected embedding dimensions at positions {invalid_dimensions[:10]}."
        )

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))

    existing = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    # Use cosine distance because document and query embeddings are normalized.
    collection_metadata = {
        "hnsw:space": "cosine",
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_revision": config.EMBEDDING_MODEL_REVISION,
    }
    if corpus_fingerprint is not None:
        collection_metadata["corpus_fingerprint"] = corpus_fingerprint

    collection = client.create_collection(COLLECTION_NAME, metadata=collection_metadata)

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    stored_count = collection.count()
    if stored_count != len(chunks):
        raise RuntimeError(f"Index count mismatch: expected {len(chunks)}, found {stored_count}.")

    return collection, model


def main() -> None:
    """Build the index and run a small nearest-neighbour sanity check."""
    manifest, chunks = load_corpus()
    print(f"\nEmbedding and indexing {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")

    collection, model = build_index(
        chunks,
        corpus_fingerprint=manifest["corpus_fingerprint"],
    )
    print(
        f"[ok] Collection '{COLLECTION_NAME}' has {collection.count()} vectors, "
        f"persisted to {config.VECTORSTORE_DIR}"
    )

    query = (
        "Represent this sentence for searching relevant passages: "
        "What are the major products AMD sells?"
    )
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    print("\n--- smoke test: nearest chunks for an AMD products question ---")
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        strict=True,
    ):
        similarity = 1 - distance
        print(f"\n[{metadata['doc_name']} p{metadata['page_number']}] similarity={similarity:.3f}")
        print(document[:200])


if __name__ == "__main__":
    main()
