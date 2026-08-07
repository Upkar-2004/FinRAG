"""Central control panel for FinRAG."""

import os  
from pathlib import Path

from dotenv import load_dotenv  # Loads environment variables from a .env file

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent  # The root directory of the project 
DATA_DIR = PROJECT_ROOT / "data" 
PDF_DIR = DATA_DIR / "pdfs"  
EVAL_DIR = DATA_DIR / "eval"  # Contains the evaluation questions and the generated evaluation results
VECTORSTORE_DIR = DATA_DIR / "chroma"  
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
FINANCEBENCH_SRC = DATA_DIR / "_financebench_src"  # cloned source repo (gitignored)


# Fixed FinanceBench corpus used by the application and evaluations.
# Keys match FinanceBench document IDs and PDF filename stems.
# Values are stored in chunk metadata and displayed as company labels.
CORPUS_DOCS: dict[str, str] = {
    "AMD_2022_10K": "AMD (semiconductors)",
    "AMERICANEXPRESS_2022_10K": "American Express (financial services)",
    "BOEING_2022_10K": "Boeing (aerospace)",
    "PEPSICO_2022_10K": "PepsiCo (consumer staples)",
    "3M_2022_10K": "3M (industrials)",
}


# Chunking parameters
# chunk_size = characters per chunk; chunk_overlap = characters shared between
# neighbours so a sentence split across a boundary isn't lost
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150  # this increases the chance that one chunk preserves the whole sentence

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be at least 0 and less than CHUNK_SIZE")


# Embedding model (no per request API cost). bge-small is a strong, small model.
# 384-dimensional vectors, fast on CPU.
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5" #Converts the text into a numerical vector
EMBEDDING_MODEL_REVISION: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a" # This identifies the specific version
HF_LOCAL_FILES_ONLY: bool = os.getenv("FINRAG_HF_LOCAL_FILES_ONLY", "0").lower() in {
    "1",
    "true",
    "yes",
}
# Both the document chunks and the questions use the same embedding model and configuration.

# Final number of retrieved chunks sent to generation
# Retrieval: how many chunks to pull back for a question
TOP_K: int = 5

if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than 0.")


GROQ_MODEL: str = os.getenv("FINRAG_GROQ_MODEL", "openai/gpt-oss-120b").strip() # This selects the language model to be used after retrieval
GROQ_REASONING_EFFORT: str = os.getenv("FINRAG_GROQ_REASONING_EFFORT", "low").strip().lower()
GROQ_TIMEOUT_SECONDS: float = float(os.getenv("FINRAG_GROQ_TIMEOUT_SECONDS", "60"))
GROQ_MAX_COMPLETION_TOKENS: int = int(os.getenv("FINRAG_GROQ_MAX_COMPLETION_TOKENS", "1600"))

if not GROQ_MODEL:
    raise ValueError("FINRAG_GROQ_MODEL cannot be empty.")

if GROQ_REASONING_EFFORT not in {"low", "medium", "high"}:
    raise ValueError(
        "FINRAG_GROQ_REASONING_EFFORT must be low, medium, or high."
    )

if GROQ_TIMEOUT_SECONDS <= 0:
    raise ValueError("FINRAG_GROQ_TIMEOUT_SECONDS must be greater than 0.")

if GROQ_MAX_COMPLETION_TOKENS <= 0:
    raise ValueError(
        "FINRAG_GROQ_MAX_COMPLETION_TOKENS must be greater than 0."
    )

# Conservative best-chunk cosine threshold for query-expanded dense retrieval.
# It rejects weak topical matches; it is not answer confidence.
MIN_RELEVANCE_SCORE: float = 0.60

if not -1.0 <= MIN_RELEVANCE_SCORE <= 1.0:
    raise ValueError("MIN_RELEVANCE_SCORE must be between -1.0 and 1.0.")


# Putting a retrieve and rerank pipeline to use

# Experimental query-time reranker; not used by the Streamlit default path.
# Changing this model requires reevaluation but not a Chroma rebuild.
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL_REVISION: str = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
RERANK_CANDIDATES: int = 20

if RERANK_CANDIDATES < TOP_K:
    raise ValueError("RERANK_CANDIDATES must be greater than or equal to TOP_K.")


# This combines: Dense semantic retrieval + BM25 keyword retrieval 
# Experimental dense + BM25 fusion; not used by the Streamlit default path.
# Both settings affect query-time ranking and do not require a Chroma rebuild.
HYBRID_CANDIDATES: int = 20
RRF_K: int = 60

if HYBRID_CANDIDATES < TOP_K:
    raise ValueError("HYBRID_CANDIDATES must be greater than or equal to TOP_K.")

if RRF_K < 0:
    raise ValueError("RRF_K must be greater than or equal to 0.")
 
