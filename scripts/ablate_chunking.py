"""
Day 2 — Chunking ablation: does chunk_size/chunk_overlap affect Recall@k?

Motivated by a concrete failure case (financebench_id_00995, "what products
does AMD sell"): the answer's section header ("Overview") and its content
landed in two different chunks, split apart by a chunk_size=1000 boundary
that fell in almost the worst possible spot on the page. This tests a few
chunk_size/chunk_overlap combinations end-to-end (re-chunk -> re-embed ->
re-index -> evaluate) to see whether moving the split point actually moves
Recall@k, or whether this is a deeper problem chunking alone can't fix.

Each trial OVERWRITES the same Chroma collection ('finrag_chunks') in
data/chroma/ — this script is for comparison, not for keeping multiple
indexes around simultaneously. After it finishes, data/chroma/ reflects
whichever config ran LAST, not necessarily the best one; once you've picked
a winner from the summary table, rebuild deliberately with that config.

Run:
    python3 scripts/ablate_chunking.py
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import config  # noqa: E402
from src.finrag.ingest import chunk_corpus  # noqa: E402
from src.finrag.index import build_index  # noqa: E402
from src.finrag.retrieve import reset_collection_cache  # noqa: E402
from scripts.evaluate_retrieval import evaluate  # noqa: E402

CONFIGS_TO_TEST = [
    (500, 100),
    (1000, 150),  # current default, rerun here for a fair side-by-side
    (1500, 300),
]


def run_trial(chunk_size: int, chunk_overlap: int) -> dict:
    # ingest.py's chunk_document() reads config.CHUNK_SIZE / CHUNK_OVERLAP
    # fresh every time it's called (not captured once at import time), so
    # mutating the module's attributes here takes effect immediately — no
    # need to edit config.py by hand or restart Python between trials.
    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = chunk_overlap

    print(f"\n{'=' * 60}\nchunk_size={chunk_size}  chunk_overlap={chunk_overlap}\n{'=' * 60}")
    chunks = chunk_corpus()
    print(f"[ok] {len(chunks)} chunks")

    build_index(chunks)  # deletes + recreates 'finrag_chunks' with the new chunks
    reset_collection_cache()  # retrieve.py must not keep querying the old collection

    results = evaluate(k=config.TOP_K)
    n_hits = sum(r["hit"] for r in results)
    recall = n_hits / len(results)
    print(f"Recall@{config.TOP_K}: {n_hits}/{len(results)} ({recall:.1%})")

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_chunks": len(chunks),
        "hits": n_hits,
        "recall": recall,
    }


if __name__ == "__main__":
    trials = [run_trial(cs, co) for cs, co in CONFIGS_TO_TEST]

    print(f"\n{'=' * 60}\nABLATION SUMMARY\n{'=' * 60}")
    header = f"{'chunk_size':>10} {'overlap':>8} {'n_chunks':>9} {'recall@' + str(config.TOP_K):>10}"
    print(header)
    for t in trials:
        print(f"{t['chunk_size']:>10} {t['chunk_overlap']:>8} {t['n_chunks']:>9} {t['recall']:>9.1%}")
