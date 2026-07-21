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

# Retrieval-score threshold for the out-of-scope guardrail (tuned on Day 2).
# If the best retrieved chunk is less similar than this, we refuse instead of
# guessing. Starts as a placeholder; you'll set it from data.
MIN_RELEVANCE_SCORE = 0.30
