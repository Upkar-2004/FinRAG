# FinRAG — Retrieval-Augmented Q&A over Financial Filings

A RAG assistant that answers natural-language questions about company 10-K
filings, grounding every answer in retrieved passages with source citations
(company, document, page), and using a sandboxed calculator tool for
derived numbers (ratios, margins, YoY change) that don't appear verbatim in
the text. Built to be evaluated honestly: retrieval quality and answer
quality are measured separately, against a held-out labeled question set,
and every design decision below is backed by a number, not a guess.

> **Status:** feature-complete. Ingestion, retrieval (6 techniques
> compared), tool-augmented generation, an out-of-scope guardrail, and a
> Streamlit demo UI are all built and wired together. See
> [`docs/finrag_learning_report.md`](docs/finrag_learning_report.md) for
> the full build log — every experiment, bug, and decision, in order.

## What it does

Ask a question like *"Does AMD have a reasonably healthy liquidity profile
based on its quick ratio for FY22?"* and FinRAG:

1. Retrieves the most relevant passages from the 5-filing corpus (dense
   embedding search, expanded with domain vocabulary — see Retrieval below).
2. Checks the best match's similarity against a calibrated threshold and
   refuses up front if the question is out of scope, instead of guessing.
3. Sends the retrieved context to an LLM (Groq, `llama-3.3-70b-versatile`)
   with a system prompt that requires citing every claim as
   `(Company, page N)` and reaching for a calculator tool — not mental math —
   for any derived number.
4. Returns a grounded answer with citations, plus the underlying evidence
   passages, in a Streamlit UI.

## Results

Retrieval quality, measured as Recall@5 (does any of the top-5 retrieved
chunks land on a gold answer page) across all 29 held-out FinanceBench
questions. Six techniques were implemented and measured against the same
eval set with the same scoring rule:

| Technique | Recall@5 |
|---|---|
| **Query expansion (shipped)** | **37.9% (11/29)** |
| Dense embeddings only (baseline) | 34.5% (10/29) |
| Hybrid BM25 + dense, RRF-tuned | 31.0% (9/29) |
| Hybrid BM25 + dense, RRF default | 24.1% (7/29) |
| Cross-encoder reranking | 20.7% (6/29) |
| BM25 keyword search only | 13.8% (4/29) |

Reranking and hybrid retrieval both *looked* like they should help and
didn't — both are diagnosed with hit-set comparisons and RRF-math tracing
in Section 17 of the learning report, not just reported as negative
results. Query expansion won because it's the only technique that touches
the query's vocabulary instead of reprocessing the documents — see
"Retrieval" below.

## Retrieval: what was tried, and why the winner won

Every question the retriever misses shares a pattern: the question uses
abstract financial terminology ("quick ratio," "liquidity profile") that
never appears in the filing itself — filings state raw line items
("Cash and cash equivalents: $4,835"), not their named ratios. A query
embedding for "quick ratio" ends up closer to *any* company's prose that
happens to discuss liquidity than to the specific numbers that would let
you compute it.

`src/finrag/query_expand.py` hand-maps financial metric names (quick
ratio, operating margin, gross margin, effective tax rate, EBITDA) to the
raw line-item vocabulary their standard definitions are built from, and
appends it to matching queries before embedding — a zero-cost proxy for
HyDE (Hypothetical Document Embeddings). It rescued one gold page into the
top-5 with zero regressions, and is the production retriever
(`retrieve_with_expansion()`). It's deliberately narrow — 5 hand-mapped
metrics, not a general solution — with full LLM-generated HyDE as the
natural next step.

## Generation: tool-augmented, not free-text arithmetic

LLMs predict plausible next tokens, not computed results — they're well
documented to make arithmetic errors even when they've picked out the
right numbers. `src/finrag/generate.py` gives the model three tools instead:
`percent_change`, `ratio`, and a general `calculate` expression evaluator.
None of them use Python's `eval()` — tool arguments come from the model,
which is untrusted input, so `calculate()` parses with `ast.parse()` and
walks the tree through a hand-written evaluator that only permits numeric
constants and a whitelisted set of operators (`+ - * / ** %`); anything
else raises instead of executing.

## Guardrail: calibrated from real out-of-scope data

`config.MIN_RELEVANCE_SCORE = 0.60` isn't a guess — it's set from the gap
between measured similarity scores on 7 real out-of-scope test questions
(max 0.577) and the real in-scope range across all 29 eval questions
(0.660–0.810). It reliably catches genuinely unrelated questions ("What's
the capital of France?"). It does **not** catch "right structure, wrong
company" questions (a well-formed question about Apple's revenue, when
Apple isn't in the corpus) — documented as a real, known limitation rather
than silently ignored, since no similarity threshold can structurally
distinguish those two cases.

## Corpus

Five FY2022 10-K filings across different sectors (AMD, American Express,
Boeing, PepsiCo, 3M), chosen for varied retrieval difficulty. Documents and
29 labeled evaluation questions come from the
[FinanceBench](https://github.com/patronus-ai/financebench) open-source
subset. **Review FinanceBench's terms before any commercial use.** This
repo does not redistribute the filings — `scripts/prepare_data.py`
downloads them.

## Setup

Requires Python 3.10+ and git.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the corpus + build the eval set (one command, ~1 min)
python scripts/prepare_data.py

# 4. LLM access -- pick ONE:
#    (a) Groq (recommended, free, fast): get a key at https://console.groq.com
cp .env.example .env               # then paste your key into .env
#    (b) Fully local, no key: install Ollama (https://ollama.com) then:
#        ollama pull llama3.1:8b

# 5. Build the vector index (parses PDFs, chunks, embeds, persists to Chroma)
python -m src.finrag.index

# 6. Run the demo UI
streamlit run app.py
```

## Project layout

```
config.py                          # every tunable knob: chunking, models, top-k,
                                    #   guardrail threshold, RRF/rerank params
app.py, ui.py                      # Streamlit demo UI
src/finrag/
  ingest.py, index.py              # parse PDFs -> chunk -> embed -> persist to Chroma
  retrieve.py                      # dense retrieval + retrieve_with_expansion() (shipped)
  bm25_retrieve.py                 # BM25 keyword baseline
  hybrid_retrieve.py               # BM25 + dense, RRF fusion
  rerank.py                        # cross-encoder reranking
  query_expand.py                  # the metric-vocabulary expansion that won
  generate.py                      # tool-augmented generation + guardrail
scripts/
  prepare_data.py                  # download corpus + build eval set
  evaluate_retrieval.py            # Recall@5 for dense-only
  evaluate_bm25.py, evaluate_hybrid.py,
  evaluate_rerank.py, evaluate_query_expansion.py
  evaluate_generation.py           # citation/groundedness checks on generated answers
  ablate_chunking.py, ablate_rrf_k.py
docs/finrag_learning_report.md     # the full build log: every experiment, bug, decision
data/                               # downloaded/generated, gitignored
```

## What's left

- **Generation-quality eval hasn't been run to completion** — the script
  (`scripts/evaluate_generation.py`) is built and measures citation
  formatting, groundedness, and whether the cited page matches the gold
  page, but no results have been saved yet.
- **Latency/cost logging** isn't built yet — currently in progress.
- Full LLM-generated HyDE, generalizing query expansion beyond 5 hand-mapped
  metrics, is the natural next step for retrieval.
