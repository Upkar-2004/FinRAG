# FinRAG — Retrieval-Augmented Q&A over Financial Filings

A RAG assistant that answers natural-language questions about company 10-K
filings, grounding every answer in retrieved passages with source citations
(company, document, page). Built to be evaluated honestly: it measures
retrieval quality and answer quality separately on a held-out labeled set.

> Status: **work in progress.** Day 0 (data prep) complete. Ingestion,
> retrieval, generation, evaluation, and guardrails to follow.

## Corpus

Five FY2022 10-K filings across different sectors (AMD, American Express,
Boeing, PepsiCo, 3M). Documents and 29 labeled evaluation questions come from
the [FinanceBench](https://github.com/patronus-ai/financebench) open-source
subset. **Review FinanceBench's terms before any commercial use.** This repo
does not redistribute the filings — `scripts/prepare_data.py` downloads them.

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

# 4. LLM access — pick ONE:
#    (a) Groq (recommended, free, fast): get a key at https://console.groq.com
cp .env.example .env               # then paste your key into .env
#    (b) Fully local, no key: install Ollama (https://ollama.com) then:
#        ollama pull llama3.1:8b
```

After setup you'll have:
- `data/pdfs/` — the five source 10-K PDFs
- `data/eval/eval_set.jsonl` — 29 questions with gold answers and gold pages

## Project layout

```
config.py                 # all tunable knobs (chunking, models, top-k, paths)
scripts/prepare_data.py   # Day 0: download + build eval set
src/finrag/               # the pipeline (ingestion, retrieval, generation) — WIP
data/                     # downloaded, gitignored
```

## Roadmap

- [x] Day 0 — data acquisition + held-out eval set
- [ ] Day 1 — ingestion (parse → chunk → embed → index) + grounded answers with citations
- [ ] Day 2 — evaluation harness, guardrails, hybrid retrieval, failure analysis, ablation
      (retrieval hit@k = any retrieved chunk's `page_index` is in the question's `gold_pages`)
- [ ] Day 3 — calculator tool, cost/latency logging, Streamlit demo, final README
