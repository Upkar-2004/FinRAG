"""
Stage 2 — Indexing: chunks -> embeddings -> persisted Chroma collection.

Pipeline: embed every chunk's text with the configured sentence-transformer,
then store (id, embedding, text, metadata) in a Chroma collection that
persists to disk at config.VECTORSTORE_DIR. Re-running this script rebuilds
the collection from scratch so it always reflects the current chunks.

Design decisions worth knowing for later:
  - We embed manually with sentence-transformers and pass vectors straight
    into Chroma, instead of using Chroma's built-in embedding_function. This
    keeps "what model embedded this" explicit and inspectable, and makes the
    retrieval stage's query-side embedding an obvious mirror of this code
    rather than something hidden inside the vector store client.
  - Collection uses cosine distance ("hnsw:space": "cosine") because
    embeddings are normalized (unit length) and bge-small is designed to be
    compared with cosine similarity. Chroma reports *distance*, not
    similarity — the retrieval stage must convert via
    similarity = 1 - distance before comparing against
    config.MIN_RELEVANCE_SCORE.
  - BGE models want an instruction prefix on the QUERY side only
    ("Represent this sentence for searching relevant passages:"), never on
    the passage/document side. No prefix is added here because these are
    passages; the retrieval stage adds it to the question text.
  - IDs are deterministic (doc_name + page_index + position within the page)
    so re-running indexing overwrites cleanly instead of accumulating
    duplicates.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config
from .ingest import chunk_corpus

COLLECTION_NAME = "finrag_chunks"


def assign_ids(chunks: list[dict]) -> list[str]:
    """Deterministic id per chunk: {doc_name}_p{page_index}_c{position in page}."""
    ids = []
    counts: dict[tuple[str, int], int] = {}
    for c in chunks:
        key = (c["metadata"]["doc_name"], c["metadata"]["page_index"])
        counts[key] = counts.get(key, 0) + 1
        ids.append(f"{key[0]}_p{key[1]}_c{counts[key] - 1}")
    return ids


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Encode a list of strings into normalized vectors."""
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def build_index(chunks: list[dict]) -> chromadb.Collection:
    """Embed all chunks and persist them to a fresh Chroma collection."""
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR)) #PersistentClient is used to persist the collection to disk so it can be reloaded later without re-embedding everything.

    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME) #Delete the existing collection if it exists to ensure a fresh start. This prevents accumulation of duplicates and ensures that the collection reflects the current chunks and embeddings.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    ids = assign_ids(chunks)
    metadatas = [c["metadata"] for c in chunks]

    # Chroma's add() has an internal batch-size ceiling; insert in fixed-size
    # batches so this works regardless of corpus size or Chroma version.
    BATCH = 500
    for start in range(0, len(chunks), BATCH):
        end = start + BATCH #Add a batch of chunks to the collection, ensuring we don't exceed the batch size limit. This loop iterates over the chunks in increments of BATCH, slicing the ids, embeddings, texts, and metadatas lists accordingly.
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
    return collection


if __name__ == "__main__":
    chunks = chunk_corpus()
    print(f"\nEmbedding + indexing {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    collection = build_index(chunks)
    print(f"[ok] Collection '{COLLECTION_NAME}' has {collection.count()} vectors, persisted to {config.VECTORSTORE_DIR}")

    # Smoke test: embed one question the same way and eyeball the top hits.
    # This is NOT the retrieval stage (no MIN_RELEVANCE_SCORE gating, no
    # gold_pages comparison) — just a sanity check that similar text lands
    # near similar text.
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    query = "Represent this sentence for searching relevant passages: What are the major products AMD sells?"
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    print("\n--- smoke test: nearest chunks for an AMD products question ---")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist
        print(f"\n[{meta['doc_name']} p{meta['page_number']}] similarity={similarity:.3f}")
        print(doc[:200])
