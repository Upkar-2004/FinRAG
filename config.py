"""
Central configuration for the whole project.

Why one config file? So that every knob you might change during experiments
(chunk size, which embedding model, how many chunks to retrieve, which LLM)
lives in ONE place. When you run your Day-2 ablations, you change a value here
instead of hunting through five files. Interviewers read this as "this person
designed for experimentation," which is exactly the signal we want.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# __file__ is this config.py; .parent is the repo root. Building paths this way
# means the project works no matter what directory you run it from.
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"                # the 5 source 10-K PDFs (downloaded, gitignored)
EVAL_DIR = DATA_DIR / "eval"              # our held-out labeled questions
VECTORSTORE_DIR = DATA_DIR / "chroma"     # where the vector index persists to disk
FINANCEBENCH_SRC = DATA_DIR / "_financebench_src"  # cloned source repo (gitignored)

# ---------------------------------------------------------------------------
# The corpus: which documents we build the assistant over.
# Keys are the FinanceBench doc_name; values are just a human-friendly label.
# All FY2022 10-Ks, chosen for sector spread (varied retrieval difficulty).
# ---------------------------------------------------------------------------
CORPUS_DOCS = {
    "AMD_2022_10K": "AMD (semiconductors)",
    "AMERICANEXPRESS_2022_10K": "American Express (financial services)",
    "BOEING_2022_10K": "Boeing (aerospace)",
    "PEPSICO_2022_10K": "PepsiCo (consumer staples)",
    "3M_2022_10K": "3M (industrials)",
}

# ---------------------------------------------------------------------------
# Chunking parameters (used on Day 1).
# chunk_size = characters per chunk; chunk_overlap = characters shared between
# neighbours so a sentence split across a boundary isn't lost. These are the
# FIRST things you'll ablate on Day 2.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Embedding model (runs locally, free). bge-small is a strong, small model.
# 384-dimensional vectors, fast on CPU.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Retrieval: how many chunks to pull back for a question. Another ablation knob.
# ---------------------------------------------------------------------------
TOP_K = 5

# ---------------------------------------------------------------------------
# LLM. We default to Groq (fast, free tier, OpenAI-compatible). The Ollama
# fallback lets the repo run with NO api key at all. Both are wired on Day 1.
# ---------------------------------------------------------------------------
LLM_PROVIDER = "groq"                      # "groq" or "ollama"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.1:8b"

# Retrieval-score threshold for the out-of-scope guardrail. Calibrated
# from real data, not guessed: retrieve_with_expansion()'s in-scope range
# across all 29 eval questions is ~0.660-0.810 (hits AND misses -- the
# two overlap heavily, so this threshold can only ever catch OUT-of-scope
# queries, never distinguish a will-succeed retrieval from a will-fail
# one on an in-scope question). 7 out-of-scope test questions split into
# two groups: genuinely unrelated topics (capital of France, baking,
# weather -- max 0.577) score clearly below the in-scope floor; but
# "right structure, wrong company" queries (Apple/Google revenue
# questions -- 0.720, 0.769) score WITHIN the in-scope range and are NOT
# caught by this guardrail -- a real, documented limitation, not an
# oversight (docs/finrag_learning_report.md, Section 18). 0.60 sits in
# the clean gap between the two groups.
MIN_RELEVANCE_SCORE = 0.60

# ---------------------------------------------------------------------------
# Reranking: a cross-encoder re-scores a wider dense-retrieval shortlist.
# ms-marco-MiniLM-L-6-v2 is a small, fast, standard choice for this exact
# task (trained specifically for query-passage relevance ranking), runs
# locally, no API cost. RERANK_CANDIDATES is how many dense hits to pull
# before reranking down to TOP_K -- wide enough to give the cross-encoder
# real material to work with, narrow enough to stay fast.
# ---------------------------------------------------------------------------
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Hybrid retrieval: how many candidates EACH method (dense, BM25) contributes
# to the RRF fusion pool, before fusing down to TOP_K. Separate knob from
# RERANK_CANDIDATES -- different mechanism, no reason to couple them.
#
# RRF_K: the damping constant in 1/(RRF_K + rank). 60 is the standard
# value from the original RRF paper (tuned for web search, many largely-
# agreeing systems). Ablation candidate: a live run showed RRF's default
# consensus bias (rewarding chunks BOTH methods rank, even weakly) beating
# out chunks only ONE method confidently found -- exactly the two
# questions (financebench_id_01226, financebench_id_01009) hybrid
# retrieval was built to rescue. A smaller K weights top ranks more
# steeply, which should favor a single method's high-confidence pick over
# two methods' mediocre agreement -- see scripts/ablate_rrf_k.py.
# ---------------------------------------------------------------------------
HYBRID_CANDIDATES = 20
RRF_K = 60
