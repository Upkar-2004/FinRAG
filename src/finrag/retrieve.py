"""
Stage 3 — Retrieval: question -> top-k relevant chunks.

Pipeline: embed the question with the same model used for indexing, but
WITH the BGE query-side instruction prefix this time (passages were
embedded WITHOUT it in index.py — see that file's docstring for why).
Query the persisted Chroma collection, convert Chroma's distance to
similarity (cosine space: similarity = 1 - distance), and return the
top-k chunks with their metadata and similarity scores.

No relevance guardrail yet. config.MIN_RELEVANCE_SCORE is an unvalidated
placeholder — it gets calibrated once scripts/evaluate_retrieval.py has
shown us real similarity-score distributions across the eval set, instead
of guessed.

Deliberately searches the WHOLE corpus, never filtered to a known
doc_name — a real user's question doesn't arrive labeled with which
filing has the answer, so neither does this function. eval_set.jsonl's
doc_name is only ever used for SCORING a retrieval as a hit or miss, never
for narrowing what gets searched.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config
from .index import COLLECTION_NAME

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Lazily-loaded, module-level singletons. retrieve() may be called many
# times in a row (once per question, in the eval harness) — neither the
# embedding model nor the Chroma client needs to be recreated per call.
# Same lesson as the index.py double-load fix, applied here up front.
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def reset_collection_cache() -> None:
    """Force the next retrieve() call to reconnect to the Chroma collection.

    Needed whenever the underlying collection was deleted and rebuilt since
    the last retrieve() call in this process (e.g. by an ablation script
    reindexing with different chunk_size settings without restarting
    Python) — without this, retrieve() would keep querying its cached
    reference to the now-stale, pre-rebuild collection.
    """
    global _collection
    _collection = None


def retrieve(question: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks most relevant to `question`, searched
    across the whole corpus (no doc_name filtering).

    Each result: {"text": str, "metadata": dict, "similarity": float}.
    Ordered nearest-first (highest similarity first) — this is the order
    Chroma's HNSW index already returns them in.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True
    ).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "similarity": 1 - dist})
    return hits


def retrieve_with_rerank(question: str, k: int = config.TOP_K) -> list[dict]:
    """Same shape and contract as retrieve() -- {"text","metadata","similarity"}
    per result -- but widens the dense search to config.RERANK_CANDIDATES
    first, then has rerank.py's cross-encoder narrow it back down to k.
    Matches the "functions as values" pattern already used for BM25: this
    can be passed straight into evaluate_retrieval.py's evaluate(retrieve_fn=...)
    to score it with the identical harness and hit-scoring rule.
    """
    from .rerank import rerank  # local import: avoid loading the cross-encoder for callers that only need plain retrieve()

    candidates = retrieve(question, k=config.RERANK_CANDIDATES)
    return rerank(question, candidates, k=k)


def retrieve_with_expansion(question: str, k: int = config.TOP_K) -> list[dict]:
    """Same shape/contract as retrieve() -- expands the query with
    query_expand.expand_query() first (see that module's docstring for
    why), then searches normally. Matches the "functions as values"
    pattern already used for BM25/reranked/hybrid retrieval: drops
    straight into evaluate_retrieval.py's evaluate(retrieve_fn=...).
    """
    from .query_expand import expand_query

    return retrieve(expand_query(question), k=k)


if __name__ == "__main__":
    question = "What are the major products and services that AMD sells as of FY22?"
    print(f"Question: {question}\n")
    for hit in retrieve(question):
        meta = hit["metadata"]
        print(f"[{meta['doc_name']} p{meta['page_number']}] similarity={hit['similarity']:.3f}")
        print(hit["text"][:200])
        print()
