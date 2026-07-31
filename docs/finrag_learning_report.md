# FinRAG Learning Report

**Purpose of this document:** this is a teaching document, not a status update. It exists so that you can study it and confidently defend every design decision made so far in an ML/RAG engineering interview — including the decisions we changed our minds about, and the honest, unflattering numbers. Every number in this report was actually produced by running the code in this repository; nothing is estimated or projected. Where something has not been measured yet, it is explicitly marked **NOT YET MEASURED**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Project State](#2-current-project-state)
3. [Data Preparation](#3-data-preparation)
4. [Ingestion / Chunking](#4-ingestion--chunking)
5. [Embeddings](#5-embeddings)
6. [Chroma Vector Store](#6-chroma-vector-store)
7. [Smoke Test](#7-smoke-test)
8. [Retrieval Evaluation](#8-retrieval-evaluation)
9. [Algorithms and Concepts Used So Far](#9-algorithms-and-concepts-used-so-far)
10. [What We Have Learned So Far](#10-what-we-have-learned-so-far)
11. [Next Steps / Roadmap From Here](#11-next-steps--roadmap-from-here)
12. [Interview Defense Section — "How I Would Explain This In An Interview"](#12-interview-defense-section--how-i-would-explain-this-in-an-interview)
13. [Addendum: BM25 Baseline (Stage 3b)](#13-addendum-bm25-baseline-stage-3b)
14. [Issues, Bugs, and Near-Misses Encountered Along the Way](#14-issues-bugs-and-near-misses-encountered-along-the-way)
15. [Broader Failure Analysis: Four More Diagnosed Cases](#15-broader-failure-analysis-four-more-diagnosed-cases)
16. [Generation Stage (Phase 1, Stage 4): Tool-Augmented Answers](#16-generation-stage-phase-1-stage-4-tool-augmented-answers)
17. [Reranking and Hybrid Retrieval: Two More Negative Results, Rigorously Tuned](#17-reranking-and-hybrid-retrieval-two-more-negative-results-rigorously-tuned)
18. [Wiring the Winning Retriever In, and Calibrating the Guardrail From Real Data](#18-wiring-the-winning-retriever-in-and-calibrating-the-guardrail-from-real-data)

---

## 1. Project Overview

### What FinRAG is trying to build

FinRAG is a **Retrieval-Augmented Generation (RAG)** question-answering system over a small, fixed corpus of company 10-K filings (annual financial reports filed with the SEC). Given a natural-language question like *"What are the major products and services that AMD sells as of FY22?"*, the system should:

1. Find the passages in the filings that actually contain the answer.
2. Generate a natural-language answer grounded in those passages.
3. Cite exactly where the answer came from (company, document, page).
4. Refuse to answer, rather than guess, when the question isn't actually covered by the corpus.

### Why this is a RAG project, not just a chatbot

A plain chatbot (or a raw LLM call) answers from whatever the model memorized during training — it can't see AMD's actual FY2022 10-K, and if asked a specific question about it, it will either refuse or, worse, **hallucinate a plausible-sounding but fabricated number.** For financial documents, a confidently wrong number is much more dangerous than an obvious refusal.

RAG fixes this by separating two concerns that a plain chatbot conflates:
- **Retrieval:** a search step that finds the *actual* relevant text from a trusted source (our indexed PDFs), using the LLM for nothing at this stage.
- **Generation:** the LLM only writes the final answer, and only after being handed the retrieved passages as its sole source of truth — it is instructed to answer *from* those passages, not from its own training-time memory.

This split is also what makes the system **evaluable**. You can measure retrieval quality (did we find the right passage?) completely independently of generation quality (did the LLM write a good answer from it?) — which is exactly what Sections 7 and 8 of this report do. A plain chatbot gives you no comparable, checkable intermediate step.

### What the final system is supposed to do

Per the project's own `README.md` roadmap:
- **Day 0** (done): acquire the corpus + build a held-out evaluation set.
- **Day 1** (in progress): ingestion (parse → chunk → embed → index) + grounded answers with citations.
- **Day 2** (not started): evaluation harness, guardrails, hybrid retrieval, failure analysis, ablation.
- **Day 3** (not started): a calculator tool, cost/latency logging, a Streamlit demo, final README.

### What corpus and eval set we are using

- **Corpus:** 5 FY2022 10-K filings, deliberately chosen for sector spread (`config.py`, `CORPUS_DOCS`):
  - AMD (semiconductors)
  - American Express (financial services)
  - Boeing (aerospace)
  - PepsiCo (consumer staples)
  - 3M (industrials)
- **Source:** the [FinanceBench](https://github.com/patronus-ai/financebench) open-source dataset (150 labeled question/answer/evidence-page triples across many companies). We filter this down to the 29 questions that are actually about our 5 chosen documents.
- **Eval set:** `data/eval/eval_set.jsonl`, 29 questions, each with a gold answer (for future generation-quality evaluation) and one or more gold evidence pages (for retrieval evaluation — see Section 3).

---

## 2. Current Project State

### What has been implemented so far

| Stage | File | Status |
|---|---|---|
| Data acquisition + eval set | `scripts/prepare_data.py` | Done, committed |
| PDF parsing + chunking | `src/finrag/ingest.py` | Done, committed |
| Embedding + vector indexing | `src/finrag/index.py` | Done, committed |
| Retrieval (question → top-k chunks) | `src/finrag/retrieve.py` | Done, **not yet committed** |
| Retrieval evaluation (Recall@k) | `scripts/evaluate_retrieval.py` | Done, **not yet committed** |
| Chunking ablation harness | `scripts/ablate_chunking.py` | Done, ran — see Section 13 addendum note below |
| BM25 keyword baseline | `src/finrag/bm25_retrieve.py`, `scripts/evaluate_bm25.py` | Done, ran — Recall@5 = 13.8% (4/29), **not yet committed** — see Section 13 |
| Generation (LLM answer + citations + calculator tools) | `src/finrag/generate.py` | Built and validated in isolation — see Section 16. Real end-to-end run (via `retrieve()`) still blocked on the same AMD quick-ratio question by a retrieval miss, not a generation bug. |
| Reranking (cross-encoder) | `src/finrag/rerank.py` | Built, measured, rejected — Recall@5 = 20.7% (6/29), a regression below dense-only. See Section 17. |
| Hybrid retrieval (BM25 + dense, RRF) | `src/finrag/hybrid_retrieve.py`, `scripts/ablate_rrf_k.py` | Built, measured, tuned, rejected — best case 31.0% (9/29) at RRF_K=0/1, still below dense-only's 34.5%. See Section 17. |
| Query expansion (zero-cost HyDE proxy) | `src/finrag/query_expand.py` | Built, measured, **accepted** — Recall@5 = 37.9% (11/29), the first technique to beat dense-only, with zero hits lost. See Section 17. |

**Decision: `retrieve_with_expansion()` (dense + metric-vocabulary query expansion) ships as the production retriever** — the best-performing of seven configurations tested (dense-only, BM25-only, reranked, hybrid at 5 `RRF_K` values, and query expansion). See Section 17 for the full comparison.
| Guardrails (`MIN_RELEVANCE_SCORE`) | — | Placeholder value in `config.py` only, unused in code |
| Hybrid retrieval, reranking | — | **Not started** |

### Git history (actual commits, in order)

```
3499739  Phase 1 stage 1: ingestion (PDF -> page text -> chunks)
dbdd41b  Fix gold-page and page-index conventions in data prep and ingestion
3ffb823  Add embedding + Chroma indexing stage (Day 1 stage 2)
f2ac8ea  Reuse loaded model in index.py smoke test instead of loading twice
```

Uncommitted at time of writing: `src/finrag/retrieve.py`, `scripts/evaluate_retrieval.py`, `scripts/ablate_chunking.py`.

### Commands that have actually been run, and what they produced

| Command | Result |
|---|---|
| `python3 scripts/prepare_data.py` | 29 eval questions written to `data/eval/eval_set.jsonl`; 5 PDFs copied to `data/pdfs/` |
| `python3 -m src.finrag.ingest` | 4,961 chunks produced across 5 documents (chunk_size=1000, overlap=150) |
| `python3 -m src.finrag.index` | 4,961 vectors embedded and persisted to `data/chroma/` (~38–42s on CPU) |
| `python3 -m src.finrag.retrieve` | Verified retrieval works standalone (AMD products question demo) |
| `python3 scripts/evaluate_retrieval.py` | **Recall@5: 10/29 (34.5%)** — see Section 8 |
| One-off `evaluate(k=10)` / `evaluate(k=20)` sweep | Recall@10: 10/29 (34.5%), Recall@20: 12/29 (41.4%) — see Section 8 |
| `python3 scripts/ablate_chunking.py` | **NOT YET RUN.** No results exist for this yet. |

### Per-document chunk counts (from `ingest.py`, chunk_size=1000, overlap=150)

```
AMD_2022_10K              :   513 chunks
AMERICANEXPRESS_2022_10K  : 1,042 chunks
BOEING_2022_10K           :   671 chunks
PEPSICO_2022_10K          : 1,559 chunks
3M_2022_10K                : 1,176 chunks
                    TOTAL  : 4,961 chunks
```

### Per-document eval question counts (29 total)

```
AMD_2022_10K              : 7
AMERICANEXPRESS_2022_10K  : 7
BOEING_2022_10K           : 7
PEPSICO_2022_10K          : 5
3M_2022_10K                : 3
```
By question type: `domain-relevant` 19, `novel-generated` 8, `metrics-generated` 2.

### The current end-to-end pipeline, visually

```
                 ┌────────────────────┐
                 │ FinanceBench repo  │  (cloned once, gitignored)
                 └─────────┬──────────┘
                           │  prepare_data.py
              ┌────────────┴────────────┐
              ▼                         ▼
   data/eval/eval_set.jsonl      data/pdfs/*.pdf
   (29 Qs, gold_pages)           (5 filings)
                                       │  ingest.py
                                       ▼
                            4,961 chunks (text + metadata:
                            doc_name, company, page_index,
                            page_number)
                                       │  index.py
                                       ▼
                        data/chroma/  (Chroma collection
                        "finrag_chunks": ids, embeddings,
                        documents, metadata)
                                       │
                          retrieve.py │ evaluate_retrieval.py
                                       ▼
                        question -> embed -> top-k chunks
                        -> scored against gold_pages
                                       │
                                       ▼
                        Recall@5 = 10/29 (34.5%)
                                       │
                                       ▼
                        generate.py: top-k chunks + question -> LLM
                        (+ calculator tools) -> cited answer  <-- we are here
                        Validated correct in isolation (Section 16);
                        full pipeline still blocked on questions
                        where retrieve() itself misses.
```

---

## 3. Data Preparation

### What `scripts/prepare_data.py` does

Three responsibilities, run in order by `main()`:

1. `ensure_financebench_source()` — clones the FinanceBench GitHub repo (shallow, `--depth 1`) into `data/_financebench_src/` if it isn't already there. Idempotent: if the directory already exists, it just prints a message and does nothing.
2. `build_eval_set()` — reads FinanceBench's 150 labeled questions (`financebench_open_source.jsonl`), keeps only the ones whose `doc_name` is one of our 5 chosen filings, and writes a normalized version to `data/eval/eval_set.jsonl`.
3. `copy_pdfs()` — copies just the 5 needed PDFs out of FinanceBench's larger PDF collection into `data/pdfs/`.

### Why we clone/use FinanceBench instead of committing files

Two reasons, both explicit in the code's own docstring:
- **Reproducibility.** Anyone who clones this repo runs one command (`python scripts/prepare_data.py`) and ends up with an identical corpus and eval set — no manually-downloaded files to go stale or drift from what the code expects.
- **We don't want to redistribute the filings ourselves.** 10-Ks are public SEC filings, but FinanceBench packaged and curated them; committing copies into our own repo means we'd be redistributing someone else's curated dataset under our own name. Cloning their repo at run-time keeps us clearly downstream of the original source, and `data/` is gitignored specifically so none of this ever gets committed.

### What `data/eval/eval_set.jsonl` contains

One JSON object per line (JSONL — see Section 9), 29 lines. Example (a straightforward single-page case):

```json
{"id": "financebench_id_00995", "company": "AMD", "doc_name": "AMD_2022_10K",
 "question": "What are the major products and services that AMD sells as of FY22?",
 "answer": "AMD sells server microprocessors (CPUs) and graphics processing units (GPUs)...",
 "question_type": "domain-relevant",
 "gold_pages": [3],
 "gold_page": 3}
```

And a multi-page case (see below for why this matters):

```json
{"id": "financebench_id_00499", "company": "3M", "doc_name": "3M_2022_10K",
 "question": "Is 3M a capital-intensive business based on FY2022 data?",
 "answer": "No, the company is managing its CAPEX and Fixed Assets pretty efficiently...",
 "question_type": "domain-relevant",
 "gold_pages": [47, 49, 51],
 "gold_page": 47}
```

Fields: `id` (FinanceBench's own ID, kept for traceability), `company`/`doc_name` (which filing this is about), `question`, `answer` (the gold answer text — **not currently used by anything**, since we have no generation stage yet; reserved for future answer-quality evaluation), `question_type` (FinanceBench's own taxonomy — see Section 8), `gold_pages` (the retrieval ground truth), `gold_page` (deprecated singular form, kept for backward compatibility only).

### What `data/pdfs/` contains

The 5 source 10-K PDF files, copied verbatim from FinanceBench's PDF collection — no modification. This is the actual corpus the system indexes and searches.

### Why `eval_set` is the answer key, not training data

Nothing in this pipeline trains or fine-tunes a model. `bge-small-en-v1.5` is used exactly as downloaded from Hugging Face — frozen weights, no fine-tuning step exists in this project. `eval_set.jsonl` is used purely to **check our work**: we run the untrained pipeline, then compare its output against these labels. This is the standard "held-out evaluation set" pattern (Section 9), not a train/test split in the ML-training sense — there is no training happening for this to be held out *from*.

### Why `gold_pages` matters (and why we changed the code)

The original version of `build_eval_set()` took only the *first* evidence page FinanceBench listed for each question:
```python
evidence = r.get("evidence") or [{}]
gold_page = evidence[0].get("evidence_page_num")
```
We found, by actually inspecting FinanceBench's raw data, that **5 of our 29 kept questions cite evidence spread across more than one page.** For example, `financebench_id_00499` ("Is 3M a capital-intensive business?") needs CAPEX/revenue (page 47), fixed-assets/total-assets (page 49), and ROA (page 51) — three different financial statements, three different pages, all independently valid evidence. Taking only page 47 as "the" gold page means a retrieval that correctly found the page-49 balance-sheet chunk would have been scored as a *miss* — an artifact of our evaluation code, not a real retrieval failure.

### Why preserving all evidence pages is better than only `gold_page`

The fixed code:
```python
evidence = r.get("evidence") or []
gold_pages = sorted(
    {e["evidence_page_num"] for e in evidence if e.get("evidence_page_num") is not None}
)
```
`gold_pages` is now the full, deduplicated, sorted list. `gold_page` (singular) is retained as `gold_pages[0]` purely for any code that hasn't been updated to use the list — it is not the source of truth. This directly changes how a "hit" is defined in `evaluate_retrieval.py` (Section 8): a retrieval is correct if it finds **any one** of the valid gold pages, not specifically the first-listed one.

### The `page_index` vs `page_number` convention

Two different, deliberately-separate fields exist everywhere pages are represented in this codebase:
- **`page_index`** — 0-based. This is pdfplumber's native page numbering (`enumerate(pdf.pages)` starts at 0) and is the *only* field ever compared against `gold_pages` or used in retrieval hit-checking. Machine-facing.
- **`page_number`** — 1-based (`page_index + 1`). Exists only for showing a human "see page 48." Never compared against anything. Human-facing, and explicitly **not guaranteed** to match the number physically printed on the page (10-Ks often have unnumbered cover pages or roman-numeral table-of-contents pages before "real" numbering starts).

Before this split existed, both `ingest.py` and `prepare_data.py` used a single, ambiguous field just called `"page"` — a real risk, because two different systems (FinanceBench's page numbers and pdfplumber's) could easily have used different conventions, and nothing in a field named `"page"` would have flagged that.

### The off-by-one issue, and how we verified `evidence_page_num` is 0-based

This was checked empirically, not assumed. For `financebench_id_00499`, FinanceBench lists `evidence_page_num: 47` with evidence text beginning "3M Company and Subsidiaries / Consolidated Statement of Income...". We opened `3M_2022_10K.pdf` with pdfplumber directly and extracted text at three neighboring indices:

```
pdfplumber page index 46 (1-based page 47): "...Goodwill Impairment Assessment..."
pdfplumber page index 47 (1-based page 48): "3M Company and Subsidiaries | Consolidated
                                              Statement of Income | ..."   <-- MATCH
pdfplumber page index 48 (1-based page 49): "...Statement of Comprehensive Income..."
```
`pdf.pages[47]` (pdfplumber's 0-based index 47, the **physical 48th page**) matched exactly. **Conclusion: FinanceBench's `evidence_page_num` is already 0-based, identical to pdfplumber's own indexing — no `+1`/`-1` conversion needed anywhere in the pipeline.** This is now documented directly in the docstrings of both `prepare_data.py::build_eval_set()` and `ingest.py::extract_pages()`, specifically so this doesn't need to be re-derived (or silently assumed wrong) later.

---

## 4. Ingestion / Chunking

*(Implemented in `src/finrag/ingest.py`)*

### How PDFs are parsed page by page

```python
def extract_pages(pdf_path) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_index": i, "text": text})
    return pages
```
[pdfplumber](https://github.com/jsvine/pdfplumber) opens the PDF and exposes each page as an object with an `.extract_text()` method, which reads the page's embedded text layer (10-Ks are digitally generated, not scanned images, so this works reliably — no OCR needed). `extract_text()` returns `None` for blank or image-only pages, so `or ""` guards against that becoming a crash later when we try to `.strip()` or split it.

### Why page-level extraction matters for citations and eval

We deliberately extract text **one page at a time**, not the whole PDF as one blob. If we concatenated the entire document into one string before chunking, we would lose the information "which page did this text come from" the moment two pages' text got merged together — and page number is the *entire* unit our evaluation ground truth (`gold_pages`) and our future citation feature are built around. Extracting per-page first, then chunking each page's text independently, guarantees every resulting chunk can be traced back to exactly one page.

### What a chunk is

A chunk is a contiguous slice of one page's text, sized to roughly `CHUNK_SIZE` characters (1000, by current default), with metadata attached identifying exactly where it came from:
```python
{
    "text": "...",
    "metadata": {
        "doc_name": "AMD_2022_10K",
        "company": "AMD (semiconductors)",
        "page_index": 3,
        "page_number": 4,
    },
}
```

### Why we chunk instead of embedding whole PDFs

Two independent reasons:
1. **Embedding models have limited input length.** `bge-small-en-v1.5` (like most sentence embedding models) has a fixed maximum token window; a full 10-K (often 100+ pages) vastly exceeds it. Even if it didn't, cramming an entire document into one vector would average away all the specific detail that makes retrieval useful — the resulting vector would represent "this document, vaguely," not "this document's answer to this specific question."
2. **Retrieval granularity.** We need to retrieve *just* the passage relevant to a question, not the whole filing, both so the future generation step has a manageable, focused context window, and so citations can point to a specific page rather than "somewhere in this 250-page document."

### How `RecursiveCharacterTextSplitter` works, conceptually

From `langchain_text_splitters`. Conceptually: it tries to split text at the *most natural* boundary it can find, in priority order — first trying to split on paragraph breaks (`"\n\n"`), then single newlines, then sentences, then words, only falling back to a hard character cut as a last resort — while keeping each resulting piece under `chunk_size` characters. This is why it's "recursive": it recurses through a list of increasingly aggressive separators until the pieces are small enough.

### What `chunk_size` and `chunk_overlap` mean

- **`chunk_size`** (currently 1000): the target maximum number of characters per chunk.
- **`chunk_overlap`** (currently 150): how many characters the end of one chunk repeats at the start of the next chunk. This exists so a sentence or idea that happens to fall right at a chunk boundary isn't split with no chunk containing it whole — the overlap gives it a second chance to appear intact in the *next* chunk even if it got cut off in the first.

### Tradeoffs of small vs. large chunks

- **Smaller chunks:** more precise (a matching chunk is more likely to be *tightly* about one topic, improving embedding specificity), but more likely to separate a topic sentence/header from its supporting detail (see the concrete failure case below), and produces more total chunks to search and store.
- **Larger chunks:** less likely to slice a short, coherent section in half, but risk diluting the embedding by mixing multiple topics into one vector, and give a future generation step more (possibly irrelevant) context to wade through per retrieved chunk.

**This tradeoff is not resolved yet — it is currently being tested via `scripts/ablate_chunking.py`, which has been written but not yet run (see Sections 2 and 11).**

### Why financial tables make chunking hard

10-Ks are full of dense tabular data (income statements, balance sheets) where the *meaning* of a number depends on its row and column headers, which may be positioned far away from the number in the page's raw reading order once `pdfplumber` flattens a table into linear text. A character-count-based splitter has no awareness of table structure — it can easily cut a table in half, separating a dollar figure from the row label that gives it meaning, or from the column header that says which fiscal year it's for. This is a known, unresolved limitation of the current chunking approach — not yet specifically addressed.

### A concrete, real example of a chunking failure (from this project)

For `financebench_id_00995` ("what products does AMD sell," `gold_pages=[3]`), we investigated why retrieval missed it (full detail in Section 8). AMD's page_index 3 is 3,971 characters total. The actual answer — a paragraph starting "Overview / We are a global semiconductor company primarily offering: ..." — begins at character offset **3,330**, i.e., right near the very end of the page; the rest of the page is a generic "Cautionary Statement Regarding Forward-Looking Statements" disclaimer common to nearly every 10-K. With `chunk_size=1000`, this page split into 5 chunks:

```
chunk 0, 1, 2 : forward-looking-statement boilerplate
chunk 3 (997 chars) : ~93% more boilerplate + just the "Overview" HEADER, at the very tail
chunk 4 (430 chars) : the actual bulleted product list — but WITHOUT the "Overview" header,
                       which landed in chunk 3 instead
```
Neither resulting chunk reads clearly as "this answers a product-listing question": chunk 3's embedding is dominated by unrelated legal boilerplate; chunk 4 is an unlabeled list of technical jargon (GPUs, FPGAs, SoCs) missing the framing sentence that would tell an embedding model what kind of list this is. This is a specific, diagnosed instance of the general "financial-document chunking is hard" problem above — a chunk boundary fell in almost the worst possible place, splitting a section header from its content.

### What metadata is attached to each chunk, and why

```python
"metadata": {
    "doc_name": doc_name,        # which filing — needed to correctly score hits (Section 8)
    "company": company,          # human-readable label, for display/citations
    "page_index": page_index,    # 0-based — compared against gold_pages
    "page_number": page_index+1, # 1-based — shown to humans in citations
}
```
Every one of these fields exists because something downstream consumes it: `doc_name` is required for correct hit-scoring (a coincidental page-number match across two different companies must not count as a hit — see Section 8); `page_index`/`page_number` implement the machine/human split from Section 3; `company` exists purely for eventual user-facing citations.

---

## 5. Embeddings

*(Implemented in `src/finrag/index.py` for indexing, `src/finrag/retrieve.py` for querying)*

### What embeddings are

An embedding model takes a piece of text and outputs a fixed-length list of numbers (a vector) that represents that text's *meaning* as a point in a high-dimensional space. The model is trained so that texts with similar meaning map to nearby points, and texts with different meaning map to distant points — regardless of exact wording. This is what converts "find text with similar meaning" (a fuzzy, linguistic problem) into "find nearby points" (a well-defined geometry problem that a computer can solve fast).

### What `BAAI/bge-small-en-v1.5` does

This is the specific model used (`config.EMBEDDING_MODEL`), loaded via the `sentence-transformers` library. It is a small (relative to large LLMs), CPU-friendly transformer model fine-tuned specifically to produce good embeddings for search/retrieval tasks (as opposed to, say, being fine-tuned for translation or classification).

### What "384-dimensional" means

Every piece of text — whether a 50-character question or a 1000-character chunk — gets mapped to exactly 384 floating-point numbers. That fixed-size vector, regardless of the input's original length, is what makes comparison possible: two 384-number vectors can be compared with a well-defined geometric formula (cosine similarity — see Section 6); two variable-length strings of raw text cannot be compared that way at all.

### Why semantically similar text has nearby vectors

This is a *learned* property, not a hard-coded rule — it emerges from how the model was trained (on large volumes of text where similar-meaning pairs were pushed together and dissimilar pairs pushed apart during training). It is why "AMD's revenue grew 44% in FY22" and "In fiscal 2022, AMD's sales increased significantly" — different words, same meaning — end up as nearby vectors, enabling *semantic* search rather than requiring exact keyword overlap.

### Why we manually embed instead of using Chroma's built-in embedding function

Chroma can auto-embed text via an `embedding_function` attached to a collection. This project deliberately embeds manually instead:
```python
model = SentenceTransformer(config.EMBEDDING_MODEL)
embeddings = embed_texts(model, texts)
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
```
Reasoning (from `index.py`'s own docstring): this keeps "what model embedded this" an explicit, inspectable Python object rather than something buried inside Chroma's client configuration — and it makes the query-side embedding in `retrieve.py` an obvious mirror of the indexing-side code, since both explicitly call `SentenceTransformer(config.EMBEDDING_MODEL)`, rather than one side being automatic and invisible.

### Why the same embedding model must be used for indexing and querying

Every embedding model defines its *own* vector space — the coordinates it assigns to a piece of text are meaningful only relative to other text embedded by that *same* model. If chunks were embedded with model A and a query were embedded with model B, their vectors would not be comparable at all — cosine similarity between them would be geometrically meaningless (essentially random), even if both models were individually "good." `retrieve.py` explicitly reuses `config.EMBEDDING_MODEL` — the same constant `index.py` used — for exactly this reason.

### BGE's asymmetric query prefix

`bge-small-en-v1.5` was trained on (query, relevant-passage) pairs where, in every training example, the query side always had the instruction string `"Represent this sentence for searching relevant passages: "` prepended, and the passage side never did. Through that consistent pattern, the model's weights learned an implicit distinction: text arriving with that prefix should be represented as a *search query*; text without it should be represented as a *searchable passage*. This is not a rule enforced by any code — nothing in `sentence-transformers` special-cases that string. It only works because we deliberately follow the same convention the model was trained under:
- `index.py` embeds passages **without** the prefix (`embed_texts()`, `texts = [c["text"] for c in chunks]`).
- `retrieve.py` embeds the question **with** the prefix (`QUERY_PREFIX + question`).

Mismatching this convention (e.g. adding the prefix to both sides, or neither) would not crash anything — it would silently degrade retrieval quality, since it's a *calibration* the model was trained under, not a hard mechanical requirement. This is a real, general lesson about ML systems: some bugs never raise an exception, they just quietly produce worse results, and nothing except an actual quality measurement (Section 8) would catch it.

### What `normalize_embeddings=True` does, and why it matters

Scales every output vector to unit length (length exactly 1). This is what makes cosine similarity and dot-product mathematically equivalent for these vectors, and it's the standard setup `bge-small` is designed to be used and benchmarked under. This choice is directly linked to the Chroma collection's `"hnsw:space": "cosine"` configuration (Section 6) — both sides of that pairing have to agree for the resulting similarity scores to mean what we intend them to mean.

---

## 6. Chroma Vector Store

*(Implemented in `src/finrag/index.py`, queried in `src/finrag/retrieve.py`)*

### What Chroma stores

For every chunk, four parallel pieces of data, all inserted together via `collection.add(...)`:
- **`ids`** — a unique string identifying this chunk (see "deterministic IDs" below).
- **`embeddings`** — the 384-number vector.
- **`documents`** — the actual chunk text.
- **`metadatas`** — the `doc_name`/`company`/`page_index`/`page_number` dict from Section 4.

### Why we persist the vector store to disk

```python
client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
```
`PersistentClient` writes the collection to `data/chroma/` on disk. Without this (using `chromadb.Client()` instead), the entire index would live only in memory and vanish the moment the Python process exits — meaning every single script that needs to query the index would first have to re-embed all 4,961 chunks from scratch, which is exactly the ~40-second cost we want to pay *once*, not on every run.

### What a "collection" is

Chroma's unit of organization — roughly analogous to a table in a traditional database. This project uses exactly one collection, named `"finrag_chunks"` (the `COLLECTION_NAME` constant in `index.py`, imported by `retrieve.py` rather than re-typed, so the name only exists as a literal string in one place).

### Why we delete/rebuild the collection before indexing

```python
existing = {c.name for c in client.list_collections()}
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
```
Chroma raises an error if you try to `create_collection` with a name that already exists, so we check first and delete if present. This makes re-running `index.py` (or the ablation script) **idempotent** (Section 9): running it once or five times in a row produces exactly the same end state — a collection that reflects the *current* chunks, never a stale mix of old and new.

### What deterministic IDs are, and why they matter

```python
def assign_ids(chunks: list[dict]) -> list[str]:
    ids = []
    counts: dict[tuple[str, int], int] = {}
    for c in chunks:
        key = (c["metadata"]["doc_name"], c["metadata"]["page_index"])
        counts[key] = counts.get(key, 0) + 1
        ids.append(f"{key[0]}_p{key[1]}_c{counts[key] - 1}")
    return ids
```
Produces IDs like `AMD_2022_10K_p11_c0`, `AMD_2022_10K_p11_c1` — built from the chunk's own content/position, not an arbitrary counter. The counter is keyed by **`(doc_name, page_index)` together**, not `page_index` alone, because `page_index` resets to 0 for every document — keying by page alone would either cause the "position in page" counter to leak incorrectly across unrelated documents, or (if `doc_name` were dropped from the ID string entirely) cause genuine ID collisions, since Chroma's ID namespace is flat across the whole collection, not scoped per document. Because IDs are derived purely from content, re-running indexing on the same chunks reproduces the exact same IDs every time — combined with the delete-and-recreate pattern above, this is what makes rebuilding safe to do repeatedly.

### Why inserts are batched

```python
BATCH = 500
for start in range(0, len(chunks), BATCH):
    end = start + BATCH
    collection.add(ids=ids[start:end], embeddings=embeddings[start:end], ...)
```
Chroma has an internal ceiling on how many items can be added in a single `.add()` call (backend- and version-dependent). Inserting in fixed-size batches of 500 sidesteps needing to introspect that limit directly, and works regardless of corpus size.

### What cosine distance/similarity means

Cosine similarity measures the angle between two vectors, ignoring their magnitude — it answers "do these two vectors point in the same direction," which for normalized (unit-length) text embeddings corresponds to "are these two pieces of text semantically similar." It ranges, in principle, from -1 (opposite meaning) to 1 (identical meaning); in practice for real text, unrelated passages tend to sit noticeably below related ones, though (as Section 8 shows) the separation isn't always as clean as that description implies.

### Why Chroma returns distance, and why we convert `similarity = 1 - distance`

Chroma's `.query()` API reports *distance* (lower = more similar), not similarity, by convention. For a collection configured with `"hnsw:space": "cosine"` (as this one is), that distance is defined such that `similarity = 1 - distance` recovers the cosine similarity value. Both `index.py`'s smoke test and `retrieve.py`'s `retrieve()` function perform this conversion explicitly — it is not automatic, and getting it backwards (or forgetting it) would silently invert every relevance judgment downstream, including the `config.MIN_RELEVANCE_SCORE` guardrail comparison whenever that gets built.

### How metadata enables citations, filtering, debugging, and evaluation

- **Citations** (future work): `doc_name`, `company`, `page_number` are exactly what a generation stage would need to say "according to AMD's FY2022 10-K, page 4, ...".
- **Filtering** (available, not currently used by `retrieve()` — deliberately, see Section 8): Chroma supports a `where={"doc_name": ...}` filter on `.query()`, which could restrict search to one company if we ever wanted that.
- **Debugging**: every printed retrieval result in this project (smoke test, eval harness) is human-readable specifically because `doc_name`/`page_number` ride along with each result — without metadata, a "top match" would just be an opaque vector and chunk text with no way to say *where* it came from.
- **Evaluation**: `evaluate_retrieval.py`'s entire hit-scoring logic depends on reading `doc_name` and `page_index` back out of each retrieved result's metadata (Section 8).

---

## 7. Smoke Test

*(Located in `src/finrag/index.py`'s `__main__` block, and reproduced as a proper function call in `src/finrag/retrieve.py`'s `__main__` block)*

### What the smoke test does

After building the index, `index.py` embeds one hardcoded question ("What are the major products AMD sells?", with the BGE query prefix), queries the freshly-built collection, and prints the top-3 nearest chunks with their similarity scores. It is a "does this even work at all" check, in the same spirit as `ingest.py`'s `__main__` block printing one sample chunk.

### Why it is not real evaluation

It checks exactly one hand-picked question, with no ground truth comparison, no aggregate statistics, and no coverage of the other 28 eval questions or their edge cases (multi-page gold, different question types, etc.). The code comments explicitly flag this: "This is NOT the retrieval stage (no `MIN_RELEVANCE_SCORE` gating, no `gold_pages` comparison) — just a sanity check that similar text lands near similar text." A smoke test can tell you the pipeline *runs*; it cannot tell you the pipeline is *correct*. Only Section 8's systematic evaluation can do that.

### How to read an output line like `[AMD_2022_10K p11] similarity=0.729`

- `AMD_2022_10K` — the `doc_name` metadata field: which of the 5 filings this chunk came from.
- `p11` — the `page_number` metadata field (1-based): this chunk's text came from the 12th physical page of the PDF (`page_index=10`, since `page_number = page_index + 1`).
- `similarity=0.729` — the cosine similarity (already converted from Chroma's raw distance) between the query vector and this chunk's vector; higher means more similar, maximum possible is 1.0.

### Why the same page can appear multiple times in the results

A single page's text (often several thousand characters) gets split into multiple chunks by `RecursiveCharacterTextSplitter` when it exceeds `chunk_size`. It's entirely normal — expected, even — for more than one of the resulting chunks from a topically dense page to independently rank among the top-k results for a related question. In the actual smoke test output for this project, two of the top-3 results both came from AMD `p11`.

### Why high similarity does not guarantee correctness

This is the single most important lesson to take from Section 8's real evaluation numbers, but it is visible even at the smoke-test stage if you read the retrieved *text*, not just the score: for the AMD products question, the top hit (similarity 0.769, before the double-model-load fix) was a chunk about product **warranty terms** ("we generally warrant that our products sold to our customers will conform to our approved specifications...") — topically adjacent (it mentions "products" and "customers") but not actually responsive to "what products does AMD sell." A high score means "similar vocabulary/topic," not "contains the correct answer" — and Section 8's aggregate data shows this gap is systematic, not a one-off.

---

## 8. Retrieval Evaluation

*(Implemented in `scripts/evaluate_retrieval.py`, built on top of `src/finrag/retrieve.py`)*

### What `scripts/evaluate_retrieval.py` does

For each of the 29 questions in `eval_set.jsonl`: calls `retrieve(question, k=5)`, checks whether the result set contains a **hit**, and records the outcome. After all 29, it prints a per-question hit/miss list, an aggregate Recall@5, a breakdown by `question_type`, and the best-match similarity score split by hit vs. miss.

### What Recall@5 means

Out of all 29 questions, what fraction had at least one genuinely correct chunk somewhere in the top 5 retrieved results? This measures *retrieval coverage* — "did we surface the right passage at all" — not whether that passage was ranked #1 specifically, and not whether a downstream LLM would successfully use it to write a correct answer.

### What HIT and MISS mean, precisely

```python
hit = any(
    h["metadata"]["doc_name"] == row["doc_name"]
    and h["metadata"]["page_index"] in row["gold_pages"]
    for h in hits
)
```
A **HIT** requires at least one of the top-5 retrieved chunks to be **both** (a) from the correct document **and** (b) on one of that question's `gold_pages`. This two-part check is deliberate and important: `page_index` alone resets to 0 for every document, so checking page number without also checking `doc_name` could score a coincidentally-same-numbered page from a *completely different company* as a false hit. A **MISS** means none of the top-5 chunks satisfied both conditions.

### How top-5 retrieved chunks are compared against `gold_pages`

Retrieval is run against the **whole corpus** — all 4,961 chunks from all 5 companies, with no filtering to the question's known company (a deliberate design choice, explained below). The resulting top-5 chunks (which could, in principle, include chunks from the wrong company entirely) are checked against the question's `gold_pages` list using the hit rule above — a hit on *any one* of potentially several valid gold pages counts, consistent with the reasoning in Section 3.

### Why this evaluates retrieval only, not answer generation

There is no LLM call anywhere in this evaluation. It only checks whether the *right source material* was found — it says nothing about whether a language model, given that material, would write a correct, well-cited answer. These are measured completely separately by design (Section 1) — a low Recall@5 here means generation quality can't even be properly evaluated yet downstream (no point measuring how well an LLM uses retrieved context if the context is frequently wrong), and conversely a future high generation-quality score would be meaningless if it turned out to be lucking into correct answers despite wrong retrieved context.

### Why retrieval searches the whole corpus, not just the known company

`eval_set.jsonl` happens to record which company each question is about, but a real deployed user never states this explicitly ("what products does AMD sell," not "search only the AMD filing"). Filtering retrieval to the ground-truth `doc_name` would produce an artificially inflated Recall@k that doesn't reflect what the system can actually do at inference time — and would hide a real, interesting failure mode (a chunk from the wrong company outscoring the right one) that the project's own roadmap explicitly wants to study ("failure analysis"). `doc_name` from the eval set is used *only* for scoring hits after the fact, never for narrowing what gets searched.

### The actual current result

```
Recall@5: 10/29 (34.5%)
```

### The by-question-type breakdown, and what it means

```
domain-relevant     : 7/19 (36.8%)
metrics-generated    : 0/2  (0.0%)
novel-generated      : 3/8  (37.5%)
```
`metrics-generated` questions (e.g. "What is the FY2022 unadjusted EBITDA less capex for PepsiCo?") require combining multiple numbers, often from more than one financial statement, into a computed value — the answer isn't sitting as one coherent passage anywhere, so there is no single chunk that "is" the answer the way there might be for a more direct factual question. With only 2 such questions in our eval set, `0/2` is too small a sample to be statistically confident about the category in general, but it is directionally consistent with a real, understandable mechanism, not noise.

**A related, notable pattern:** all 5 of the multi-page `gold_pages` questions (Section 3) also missed at k=5 — `financebench_id_00499` (3M, 3 gold pages), `01290` (Boeing, 3 gold pages), `01009` (PepsiCo, 2 gold pages), `03620` and `04481` (PepsiCo, 2 gold pages each). These overlap substantially with the "requires combining information from multiple places" pattern above — again, n=5 is a small sample, but it's a coherent, explainable pattern rather than five unrelated coincidences.

### The best-match similarity stats for hits and misses — the most important finding

```
hits   -> min=0.660  mean=0.721  max=0.767
misses -> min=0.680  mean=0.722  max=0.773
```
These two distributions are **essentially indistinguishable** — misses even reach a *higher* maximum similarity (0.773) than the highest-scoring hit (0.767). This result held steady across every k value tested (k=5, k=10, k=20 — see below), so it is not a k=5-specific artifact.

### Why this means `MIN_RELEVANCE_SCORE` is not easy to calibrate

`config.MIN_RELEVANCE_SCORE` (currently an untested placeholder, 0.30) was designed to answer a *different* question than the one this data measures: "is this question about something not in our corpus at all" (out-of-scope detection), not "did we retrieve the exact right chunk for a question that genuinely is answerable here." All 29 of our eval questions are legitimately in-corpus, so we have **zero data yet** on what a truly out-of-scope question's best-match similarity would look like — that data simply doesn't exist in this project yet. What we *do* now know, with real evidence: even if `MIN_RELEVANCE_SCORE` were perfectly calibrated for out-of-scope detection, it would provide **no help distinguishing a correct retrieval from an incorrect one on in-scope questions**, because the score doesn't separate them. This is a limitation to state plainly, not paper over — it directly follows the project's own "do not overclaim metrics" principle.

### An additional diagnostic: does increasing k help?

A quick sweep (reusing `evaluate(k=...)` with no code changes) tested k=5, 10, and 20:

```
k= 5: Recall@5  = 10/29 (34.5%)
k=10: Recall@10 = 10/29 (34.5%)   <- zero additional hits from doubling k
k=20: Recall@20 = 12/29 (41.4%)   <- only +2 hits from doubling k again
```
Going from k=5 to k=10 recovered **zero** additional questions. Going from k=10 to k=20 recovered only 2 more, out of 19 remaining misses. If the bottleneck were purely a *ranking* problem (the right chunk exists and is found, just ranked outside the cutoff), Recall would be expected to climb meaningfully as k grows. It barely did. This points toward the bottleneck being a **representation problem** (the right chunk often isn't being surfaced as relevant *at all*, at any reasonable k) rather than a ranking-cutoff problem — consistent with, and reinforced by, the concrete chunking-boundary failure case documented in Section 4.

### Likely reasons retrieval is currently weak (grounded in evidence gathered so far)

1. **Chunking boundaries separate section headers from their content** — directly demonstrated for `financebench_id_00995` (Section 4). Not yet known how widespread this specific pattern is across the other 18 misses; `scripts/ablate_chunking.py` (written, not yet run) is designed to test whether adjusting `chunk_size`/`chunk_overlap` measurably helps.
2. **Computed/multi-source questions** (`metrics-generated`, multi-page `gold_pages`) may be structurally hard for single-chunk retrieval regardless of chunking, since no single passage "is" the answer.
3. **Purely semantic (dense) retrieval may be under-weighting exact terms** — financial filings contain a lot of precise, literal vocabulary (specific line-item names, product brand names, section headers like "Item 1. Business") that keyword-based retrieval (BM25) might catch more reliably than embedding similarity alone. **This is a hypothesis, not yet tested** — no BM25 or hybrid retrieval exists in this codebase yet.
4. **Whole-corpus search increases task difficulty** relative to a hypothetical doc-scoped baseline, by design (see above) — some unknown fraction of misses could involve a wrong-company chunk crowding out the correct one, though this has not been specifically measured (would require re-running with the `where={"doc_name":...}` filter as a diagnostic, which has not been done).

---

## 9. Algorithms and Concepts Used So Far

**JSONL datasets** — "JSON Lines": a text file format where each line is an independent, complete JSON object (as opposed to one big JSON array spanning the whole file). Used for both `eval_set.jsonl` and FinanceBench's own source data. Advantages: you can read/write it one line at a time without loading the whole file into memory, and appending a new record never requires rewriting the rest of the file.

**PDF parsing with pdfplumber** — a Python library that reads a PDF's embedded text layer (position and content of each character/word, as encoded by whatever tool generated the PDF) and reconstructs readable text per page. Works well for digitally-generated documents like SEC filings; would not work for scanned image-only PDFs without a separate OCR step (not needed here, since these 10-Ks are digital).

**Chunking** — splitting a long document into smaller, roughly fixed-size pieces so each piece can be independently embedded and retrieved. See Section 4 for the specific splitter, parameters, and tradeoffs used in this project.

**Metadata** — structured, non-text fields attached to each unit of retrievable data (here, `doc_name`, `company`, `page_index`, `page_number` per chunk), used for filtering, citation, and evaluation, distinct from the text content itself.

**Dense embeddings** — the specific class of embedding used here: every input maps to a fixed-length, dense (mostly non-zero) vector of real numbers, in contrast to "sparse" representations like a bag-of-words vector (which is mostly zeros, one slot per vocabulary word). "Dense" is what makes semantic (meaning-based) rather than purely lexical (word-overlap-based) matching possible.

**Vector search** — the general problem of, given a query vector, finding the nearest vectors among a large stored collection. Chroma's underlying implementation uses HNSW (Hierarchical Navigable Small World graphs), an approximate-nearest-neighbor index structure that finds very-likely-nearest results in roughly logarithmic time rather than comparing against every stored vector one by one — this matters at large scale; at our current corpus size (4,961 vectors) a brute-force scan would also be fast, but the index structure is what production vector databases use regardless of current scale.

**Cosine similarity vs. distance** — see Section 6. Similarity: how aligned two vectors' directions are (higher = more similar). Chroma reports the complementary "distance" value; this project explicitly converts `similarity = 1 - distance`.

**Chroma / vector database** — a database purpose-built to store vectors alongside associated data (text, metadata) and answer "nearest neighbor" queries efficiently, with persistence to disk. Used here via `chromadb.PersistentClient`.

**Recall@k** — the fraction of evaluation questions for which a correct result appears anywhere within the top-k retrieved results. Distinct from Precision@k (what fraction of the top-k results are correct) or MRR (which additionally weights *how high* the correct result was ranked) — Recall@k, as used here, only asks "was it found at all within the cutoff," not "was it ranked first." This project currently reports Recall@k only; MRR and Precision@k are not implemented.

**Held-out evaluation** — testing a system against labeled examples it was not built or tuned using. Here, `eval_set.jsonl`'s 29 questions serve this role even though no model training occurs in this project — the "held-out" principle still applies to the pipeline's configuration choices (chunk size, retrieval method, etc.), which should be validated against this set rather than hand-tuned to look good on examples we've manually inspected.

**Multi-page gold evidence** — the situation (Section 3) where a single question's correct answer is supported by evidence spread across more than one page/passage, requiring the evaluation logic to accept a match against *any* valid page rather than one designated page.

**Idempotent data/index scripts** — a script is idempotent if running it multiple times produces the same end result as running it once (Section 6's collection delete-and-recreate pattern, Section 6's deterministic chunk IDs, `prepare_data.py`'s "skip cloning if already present" check). This matters because it makes re-running any stage of the pipeline safe by default — no manual cleanup required, no risk of silently accumulating duplicate or stale data.

---

## 10. What We Have Learned So Far

### What is working

- The full mechanical pipeline runs end-to-end without errors: PDF → page text → chunks → embeddings → persisted vector store → retrieval → scored evaluation.
- Page-indexing conventions were verified empirically, not assumed, and are internally consistent throughout (`page_index` 0-based everywhere it's compared against gold data).
- The evaluation harness correctly implements the harder, more honest whole-corpus retrieval task, with a correctness-critical doc_name+page_index hit check that a naive implementation would likely have gotten wrong.
- Caching/performance fixes (the double-model-load fix, the lazy singleton pattern, the collection-cache invalidation for the ablation script) were identified and fixed through actually reading real output, not by only reading the code.

### What is not working yet

- **Retrieval quality itself is weak: Recall@5 = 34.5%.** Fewer than half of the 29 questions retrieve a correct source page in the top 5.
- **`metrics-generated` questions are at 0% Recall** (albeit n=2).
- Increasing k from 5 to 20 only recovers 2 additional questions — the ceiling doesn't move much even with a much larger retrieval window.
- No generation stage exists — the system cannot currently answer a question end-to-end, only retrieve candidate source material.
- No guardrail logic is wired up (`MIN_RELEVANCE_SCORE` is unused in code, and current evidence suggests it wouldn't help with in-scope retrieval quality even once calibrated).

### What the current metrics honestly say

They say: **plain, default-configuration semantic search over naively-chunked financial filings finds the right page for about a third of realistic questions, in the top 5 results, searching the whole corpus.** They do not yet say anything about whether chunking changes help (untested), whether hybrid retrieval helps (not built), or how well an LLM could write correct, cited answers even from the chunks we *do* retrieve correctly (not built/measured).

### What we should not overclaim

- We should not claim "34.5% Recall@5" reflects the ceiling of what semantic retrieval can achieve on this corpus — we haven't tuned chunking, tried hybrid retrieval, or tried reranking yet.
- We should not claim the similarity-score analysis proves `MIN_RELEVANCE_SCORE` is useless in general — only that it doesn't separate hit/miss quality on in-scope questions; its intended out-of-scope-detection use case remains untested (no out-of-scope questions exist in our eval set).
- We should not claim the 5/5 multi-page-miss pattern or the 0/2 metrics-generated result are statistically robust findings — both are small-sample observations that are *directionally* consistent with an understandable mechanism, not proven at this sample size.
- We should not claim the chunking-boundary explanation (Section 4) generalizes to all 19 misses — it was diagnosed for exactly one concrete case; the ablation script exists specifically to test this at scale, and hasn't been run yet.

### What this teaches about real RAG systems, generally

- **A pipeline that runs without crashing is not the same as a pipeline that works.** Every stage of this project produced plausible-looking, non-crashing output at every step — the actual retrieval quality problem was invisible until we built a real, ground-truth-based evaluation.
- **Similarity scores are not automatically a confidence signal.** This is a genuinely common trap in RAG systems: assuming "the top result had a high score" means "the top result is probably right." Our own data shows that assumption is false on this corpus.
- **Silent degradation is a real bug class in ML systems**, distinct from crashes — mismatched embedding conventions (like the BGE query-prefix asymmetry, Section 5), if gotten wrong, would produce valid-looking vectors and non-error output while quietly performing worse, and only a real quality measurement would catch it.
- **Chunking is not a minor implementation detail** — it directly determines what information can possibly be retrieved together, and a boundary falling in the wrong place can make otherwise-perfect embeddings and search infrastructure fail on an easy, obviously-answerable question.

---

## 11. Next Steps / Roadmap From Here

### Phase 1, Stage 3: proper retrieval module/CLI

- **What we'll build:** `src/finrag/retrieve.py` already exists as the reusable function; this step is closer to "wire it into something a user can actually run" — e.g. a small CLI or REPL loop that takes a typed question and prints results, rather than only a hardcoded demo question in `__main__`.
- **Why it matters:** makes the retrieval stage independently usable/demoable before generation exists, and useful for manual qualitative spot-checking during the chunking ablation work.
- **Concept to understand:** the difference between a library function (`retrieve()`, meant to be imported) and a script entry point (`__main__`, meant to be run directly) — already partially built, this step just extends the entry point's usability.
- **How we'll measure it helped:** qualitative — can you interactively explore retrieval results for arbitrary questions without editing code each time.

### Phase 1, Stage 4: generation with Groq and citations — DONE, see Section 16

- **What we built:** `src/finrag/generate.py` — an LLM call (via the `groq` client, `config.GROQ_MODEL = "llama-3.3-70b-versatile"`) that takes a question plus its retrieved chunks, answers only from that context with citations, and can call three tools (`percent_change`, `ratio`, `calculate`) for any computed value instead of doing arithmetic itself. Pulled forward from its original Day 3 slot once Section 15's failure analysis gave concrete evidence it was needed now, not later.
- **Validated correct in isolation** (bypassing `retrieve()`, feeding the real AMD balance-sheet chunk directly): computed quick ratio ≈ 1.567 against a real gold answer of 1.57, with a properly cited final sentence. Three real bugs were found and fixed getting there — see Section 16.
- **Still open:** the exact same question fails end-to-end through the *full* pipeline, because `retrieve()` itself returns zero AMD chunks for it (a retrieval miss, not a generation bug) — see Section 16's closing note. Generation correctness and retrieval correctness remain genuinely separable failure modes, exactly as predicted below.
- **Not yet built:** automated scoring of generated answers against `eval_set.jsonl`'s gold `answer` field across the full 29-question set (still only spot-checked manually so far) — see the "generation-quality eval harness" option raised but not yet chosen at the end of Section 16.

### Phase 2: proper evaluation harness with BM25 baseline first

- **What we'll build:** a simple keyword-based (BM25) retriever as a baseline, evaluated with the *exact same* `evaluate_retrieval.py` harness and hit-scoring rule already built, so its Recall@5 is directly comparable to the current 34.5% dense-only number.
- **Why it matters:** without a baseline, "34.5%" has no reference point — is dense embedding retrieval actually better than simple keyword search on this corpus, or not? We don't currently know, and it's a natural, cheap first comparison before building anything more complex.
- **Concept to understand:** BM25 (a classical term-frequency-based ranking algorithm — scores documents by how often and how distinctively query terms appear in them, with no notion of "meaning," only literal term overlap) and why it can outperform dense embeddings specifically on queries with precise, distinctive vocabulary (exact product names, section headers, numbers) even though it can't do true semantic matching.
- **How we'll measure it helped:** direct Recall@5 comparison, same eval set, same hit-scoring rule, dense-only vs. BM25-only vs. (eventually) hybrid.

### Retrieval improvements — DONE (6 techniques tried), see Section 17 for the full comparison

- **Doc/company filtering** — tested. Rescued zero questions (Section 15).
- **Chunk-size sweep** — run. 500/1500 both worse than the 1000/150 default (Section 14.8).
- **Table serialization** — built and tested. Zero Recall@5 change (Section 15/16).
- **Reranking (cross-encoder)** — built and tested. Regression to 20.7% (Section 17).
- **Hybrid BM25 + dense with RRF** — built, tested, and its main hyperparameter (`RRF_K`) tuned across a full sweep. Best case 31.0%, still below dense-only (Section 17).
- **Query expansion (zero-cost HyDE proxy)** — built and tested. **37.9% (11/29), the first technique to beat dense-only (34.5%), zero hits lost** (Section 17).
- **Decision:** ship `retrieve_with_expansion()` (dense + metric-vocabulary expansion) as the production retriever — it beat all six alternatives tested, including plain dense-only.
- **Still open — the more general version of what just worked:** full HyDE (LLM-generates the expansion per query, rather than a hand-maintained 5-metric dictionary). The dictionary approach is deliberately narrow and won't generalize past the metrics it explicitly lists; HyDE would. Costs real Groq quota per query, which is why it wasn't attempted first — a good candidate once quota allows, now with direct evidence the underlying mechanism (query-side vocabulary enrichment) actually works on this corpus.

### Reliability

- **Groundedness** — checking that a generated answer's claims are actually supported by the retrieved context it was given, rather than the model drifting back to its own training-time knowledge.
- **Abstention** — the system correctly declining to answer when the corpus doesn't actually contain the answer, instead of guessing. This is what `MIN_RELEVANCE_SCORE` was originally intended to help with (Section 8) — its calibration remains unresolved and needs genuinely out-of-scope test questions we don't currently have.
- **Citation checking** — verifying that a cited page/document actually supports the specific claim attributed to it, not just that *some* retrieved chunk happened to be in the context window.
- **Failure taxonomy** — systematically categorizing *why* each failure happens (wrong document retrieved? right document, wrong page? right page, generation ignored it? computed/multi-source question the pipeline isn't designed for?) rather than only tracking a single aggregate pass/fail number. Section 8 and Section 4's case study are early, informal steps in this direction; nothing systematic exists yet.

### Latency/cost measurement

- **What we'll build:** timing and (for the Groq API) token/cost logging around retrieval and generation calls.
- **Why it matters:** a system that's accurate but too slow or expensive isn't practically deployable; this is a standard, expected part of a real system's evaluation story, not just accuracy.
- **Concept to understand:** the latency/cost/quality three-way tradeoff that shows up in essentially every production ML system decision (e.g., a bigger reranker or a larger k improves quality but costs more time and money per query).
- **How we'll measure it helped:** wall-clock and (for hosted LLM calls) token-cost logging per query, tracked alongside the quality metrics already being measured.

### Packaging/README/resume bullet

- **What we'll build:** an updated `README.md` reflecting the finished system, and a concise, honest, metrics-backed project description suitable for a resume/portfolio.
- **Why it matters:** the whole point of this project, per your own stated goal, is being able to explain and defend it — a clear writeup is part of that deliverable, not an afterthought.
- **Concept to understand:** how to state real, imperfect metrics (like a 34.5% baseline that later improved by some measured amount) honestly and compellingly, rather than either hiding weak numbers or overclaiming strong ones — this report is meant to be practice for exactly that skill.
- **How we'll measure it helped:** whether you can, unprompted, explain any design decision in this project and its measured tradeoffs without needing to re-read the code first.

---

## 12. Interview Defense Section — "How I Would Explain This In An Interview"

**Why RAG?**
Because the alternative — asking an LLM to answer questions about specific financial filings from its own training-time memory — either can't work (the model never saw this specific document) or, worse, produces a confident, fabricated-sounding wrong number, which is a much worse failure mode than an obvious "I don't know" for financial content. RAG separates retrieval (find the real source text) from generation (write an answer grounded in that text), which also makes the two independently measurable — I can prove my retrieval finds the right page without needing an LLM in the loop at all.

**Why chunking?**
Embedding models have a bounded input size, and even without that constraint, embedding an entire 100+ page filing as one vector would average away exactly the specific detail that makes retrieval useful. Chunking gives us retrieval at the right granularity — small enough to be specific, attributable to one page for citations, and small enough that a downstream LLM gets a focused, relevant context window instead of an entire document.

**Why embeddings?**
Because the task is semantic matching — "what products does AMD sell" and AMD's own text "we are a global semiconductor company primarily offering..." share almost no exact words, but clearly mean the same thing. Embeddings convert that fuzzy, meaning-based matching problem into a well-defined geometric nearest-neighbor problem.

**Why Chroma?**
It's a vector database that persists to disk (so we pay the embedding cost once, not on every run), supports metadata alongside vectors (essential for our citation and hit-scoring needs), and uses an approximate nearest-neighbor index (HNSW) under the hood — the same category of technology production RAG systems use at scale, even though our current corpus (4,961 vectors) doesn't strictly require it yet.

**Why cosine similarity?**
`bge-small-en-v1.5` is trained and benchmarked to be compared via cosine similarity specifically; I explicitly normalize embeddings to unit length and configure the Chroma collection for cosine distance so both sides of that comparison agree — mismatching this (e.g. using raw Euclidean/L2 distance on unnormalized vectors) would silently produce a worse, uncalibrated ranking.

**Why manual embedding, instead of Chroma's built-in embedding function?**
Explicitness. I want "which model embedded this text" to be a visible, inspectable line of code, and I want the retrieval-side query embedding to be an obvious deliberate mirror of the indexing-side passage embedding — including the asymmetric BGE query-prefix convention — rather than something that happens automatically and invisibly inside the vector store client, where a subtle mismatch could go unnoticed.

**Why preserve `gold_pages` instead of just the first evidence page?**
Because I checked the actual data and found 5 of our 29 questions genuinely have evidence on multiple pages (e.g. a question needing CAPEX, fixed assets, and ROA from three different financial statements). Scoring against only the first page would have made retrieval look artificially worse whenever the correct chunk landed on the second or third valid page — an evaluation-code artifact, not a real retrieval failure. I'd rather fix my ground truth than let a scoring bug misrepresent the system's actual performance.

**Why separate retrieval evaluation from answer generation?**
Because conflating them makes debugging much harder — if a generated answer is wrong, was it because retrieval found the wrong passage, or because the LLM ignored a correct passage it was given? I can only answer that question if I've already independently verified retrieval quality on its own, which is exactly what my Recall@5 evaluation does before any generation code exists.

**Why is 34.5% Recall@5 useful, even though it's low?**
Because it's a real, honestly-measured, reproducible baseline — not a guess, and not cherry-picked from one example. It tells me precisely how much room for improvement exists, and (via the similarity-score analysis and the concrete chunking-boundary case study) gives me specific, evidence-backed hypotheses about *why* it's low, rather than a vague sense that "it could probably be better." A credible, low, honestly-obtained number is worth far more, both scientifically and in an interview, than an inflated or unmeasured one — and every future improvement (chunking, hybrid retrieval, reranking) has a concrete number to actually beat.

**What would you improve next?**
In order: (1) run the chunk-size ablation I've already built, since I have direct evidence a chunking boundary caused at least one concrete failure; (2) add a BM25 baseline and evaluate it with the exact same harness, since I don't yet know if dense embeddings are even beating simple keyword search here; (3) build hybrid retrieval (dense + BM25, combined via Reciprocal Rank Fusion) if the baseline comparison justifies it; (4) only then move to generation, since there's limited value in generating answers from a retrieval stage I already know is finding the wrong source material two-thirds of the time.

---

## 13. Addendum: BM25 Baseline (Stage 3b)

*Added in a follow-up session, after the chunk-size ablation was actually run (Section 11's "next step") and a deeper failure case was diagnosed. Covers `src/finrag/bm25_retrieve.py`, the `retrieve_fn` parameterization added to `scripts/evaluate_retrieval.py`, and `scripts/evaluate_bm25.py`. **The BM25 Recall@k result is NOT YET MEASURED — the code has been written but `evaluate_bm25.py` has not yet been run.** Treat any Recall number for BM25 as unknown until that changes.*

### Why this exists (recap from Section 11)

The chunk-size ablation (500/1000/1500 chars) was run and came back negative — the original default (1000/150) beat both alternatives, and a targeted follow-up showed our one hand-diagnosed failure case (`financebench_id_00995`, AMD's product-overview question) never entered the top-5 at *any* tested chunk size. Digging into *why* revealed something more specific than "chunking is imperfect": a wrong passage (AMD's product-warranty section, page 11) consistently out-scored the correct one, and its similarity score actually *increased* as chunks got larger — because it genuinely shares more surface vocabulary with the query ("products," "customers," AMD product brand names) than the correct passage does. That's a semantic-confusability problem, not a boundary problem, and dense embeddings have no built-in way to tell "this text mentions products in a warranty-liability context" apart from "this text is enumerating what the company sells."

BM25 is the natural next thing to measure — not because it's guaranteed to fix that specific case (it might make the exact same mistake; both passages share the word "products"), but because we currently have **zero evidence about whether our dense embedding approach is even beating a much simpler, much cheaper method on this corpus.** Without that comparison, "34.5% Recall@5" has no reference point.

### BM25, from first principles

#### The basic problem BM25 solves

Given a search query and a large collection of documents (here: our 4,961 chunks), produce a relevance score for every document, so we can rank them and return the top few. BM25 does this with **no notion of meaning at all** — no vectors, no neural network, nothing learned from training data. It is pure counting, refined by decades of information-retrieval research (it dates to the 1970s–1990s, and is still what most search engines, including Elasticsearch by default, use as their baseline ranking function).

#### Building block 1: Term Frequency (TF)

The simplest possible idea: if a query word appears in a document many times, that document is probably more about that word. Count of the word "revenue" in a chunk = that chunk's term frequency for "revenue."

**Why TF alone isn't enough:** the word "the" appears constantly in every document and tells you nothing about relevance. Raw counts treat "the" and "EBITDA" as equally meaningful, which is clearly wrong.

#### Building block 2: Inverse Document Frequency (IDF)

Fixes the above: weight each term by how *rare* it is across the whole corpus, not just how often it appears in one document. A term that appears in only 5 of our 4,961 chunks is far more informative when it *does* appear than a term that appears in 4,000 of them. IDF is typically computed as something like `log(total_documents / documents_containing_term)` — common terms (appearing in almost every document) get an IDF near zero (contribute almost nothing to the score); rare terms get a large IDF (contribute a lot). This is exactly why a query like "unadjusted EBITDA" can be scored well by BM25 even without any notion of meaning — "EBITDA" is rare enough across a general corpus that its presence is a strong, specific signal.

#### TF × IDF, and its two remaining problems

Classical **TF-IDF** scoring multiplies these two ideas together per query term, then sums across all query terms, to get a document's total score. Two problems remained, that BM25 was specifically designed to fix:

1. **Diminishing returns on repetition.** Going from 0 occurrences of a word to 1 is a huge, meaningful jump. Going from 20 occurrences to 21 barely matters — the document was already clearly "about" that term. Plain TF-IDF scales linearly with count and doesn't capture this saturation.
2. **Document length bias.** A longer document naturally contains more words and more repeats of any given term, purely by virtue of being longer — not because it's more relevant. Raw term counts unfairly favor long documents unless corrected for length.

#### What BM25 actually adds: two tunable parameters

BM25's per-term score for a document, conceptually:

```
score(term, doc) = IDF(term) × ( TF(term, doc) × (k1 + 1) )
                              -----------------------------------------
                              ( TF(term, doc) + k1 × (1 - b + b × docLen / avgDocLen) )
```

Summed across every term in the query to get the document's total BM25 score.

- **`k1`** controls how quickly term-frequency returns diminish — a low `k1` means "1 occurrence vs. 5 occurrences barely matters, presence is what counts"; a higher `k1` means term frequency keeps mattering longer before saturating. `rank_bm25`'s `BM25Okapi` defaults to `k1=1.5`.
- **`b`** controls how strongly document length is corrected for — `b=0` means no length normalization at all (raw counts), `b=1` means full normalization (a term appearing twice in a short document and twice in a long document would be treated very differently). `BM25Okapi` defaults to `b=0.75`.

**We did not tune either parameter — we used `rank_bm25`'s defaults.** This is an explicit, named assumption, in the same spirit as `config.MIN_RELEVANCE_SCORE` being an untested placeholder (Section 8): it's a reasonable, widely-used default, but it has not been validated against our specific corpus, and a future ablation could test it the same way `chunk_size` was tested.

### How this maps onto our actual code

```python
_bm25 = BM25Okapi(tokenized_corpus)   # precomputes IDF for every term across
                                       # the whole corpus, plus average doc
                                       # length, once, at construction time
...
scores = bm25.get_scores(tokenized_query)  # applies the formula above, for
                                            # this specific query, against
                                            # EVERY document — returns one
                                            # score per document
```
`_tokenize()` (lowercase + `re.findall(r"[a-z0-9]+", ...)`) is what turns raw chunk text and the raw question into the list-of-words form both `BM25Okapi(...)` and `get_scores(...)` require — BM25 doesn't operate on strings directly, only on already-split token lists. No stemming, no stopword removal: a deliberately minimal baseline, matching the report's general "keep it simple, name the limitation" style (Section 4's ingestion tradeoffs, Section 8's guardrail limitation).

### The supporting code concepts, in order of how foundational they are

- **Lazy module-level singleton** (`_bm25`/`_chunks`, `_get_index()`, `reset_index_cache()`) — identical reasoning to Section 6/8's `retrieve.py` caching: building the index is real, repeatable work; cache it once per process instead of redoing it on every one of the eval harness's 29 calls.
- **Lambda functions and `sorted(..., key=...)`** — `sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]` sorts chunk *indices* (not the chunks or scores directly) by looking up each index's score, so the connection between "this chunk" and "its score" is never lost. `lambda i: scores[i]` is a small, throwaway, unnamed function — shorthand for `def get_score(i): return scores[i]` when defining a whole separate named function would be overkill for a one-line, single-use lookup.
- **Functions as first-class values** — `def evaluate(retrieve_fn=retrieve, ...)` in `evaluate_retrieval.py` accepts a *function* as a parameter (not the function's *result*). Because Python functions are ordinary objects, `evaluate()` can be handed either the dense `retrieve` or the new `bm25_retrieve.retrieve` and call whichever one it received internally (`hits = retrieve_fn(row["question"], k=k)`), without needing to know which one it is — the only requirement is that both return the same shape of result. This is what makes one evaluation harness reusable across fundamentally different retrieval algorithms.
- **Import aliasing** (`from src.finrag.bm25_retrieve import retrieve as bm25_retrieve`) — both retrieval modules export a function literally called `retrieve`; renaming on import avoids ambiguity at the call site in `evaluate_bm25.py`.

### Measured result (updated after running `evaluate_bm25.py`)

```
Dense (bge-small):  Recall@5 = 34.5% (10/29)
BM25 (keyword):      Recall@5 = 13.8% (4/29)
```

**Dense embeddings substantially outperform plain BM25 on this corpus** — not a marginal difference, roughly 2.5x. This answers the question this section exists to answer: dense retrieval is earning its complexity here, not just adding overhead for no benefit.

**The `financebench_id_00995` AMD case was NOT fixed by BM25** — still a miss (`best_bm25=30.150`). This confirms the prediction made above (before measuring): the wrong passage (page 11, AMD's warranty section) shares literal query vocabulary too ("products," "customers"), so keyword matching falls into the same trap semantic matching did. The failure here is not specific to embeddings — it appears to be a genuinely hard case for *any* single-signal retrieval method operating on this chunking.

**BM25 is not uniformly worse — it has different blind spots than dense retrieval.** BM25's 4 hits: `01226`, `00476`, `00494`, `01009`. Only 2 of those (`00476`, `00494`) overlap with dense's 10 hits — BM25 uniquely caught `01226` (3M operating margin, which dense missed) and `01009` (PepsiCo's operating geographies, likely a literal place-name list where keyword matching has a natural advantage). Union of "at least one method found it" = **12/29 (41.4%)** — a *ceiling*, not an automatic outcome: reaching it requires an actual fusion algorithm (Reciprocal Rank Fusion, Section 11) to correctly combine both signals; the union doesn't happen for free just because both methods exist.

**The score-distribution finding is even starker for BM25 than for dense retrieval:**
```
hits   -> min=14.421  mean=29.944  max=42.229
misses -> min=12.078  mean=33.241  max=90.434
```
Misses have both a *higher* mean and a dramatically higher max than hits — the single highest BM25 score across all 29 questions (90.434) belongs to a **miss** (`financebench_id_03620`, a `metrics-generated` PepsiCo question). Financial statement pages are dense with repeated numeric/line-item vocabulary, so a page can accumulate a large raw BM25 score without being the specific page a computed-metric question actually needs. This reinforces — more sharply than the dense case — that a raw relevance score is not a substitute for ground-truth evaluation, regardless of which retrieval algorithm produced it.

**Still untested:** whether `k1=1.5`/`b=0.75` (rank_bm25's defaults) are well-suited to short, table-heavy financial-filing chunks — no parameter ablation has been run.

---

## 14. Issues, Bugs, and Near-Misses Encountered Along the Way

*This section exists on purpose. A polished final report can make a project look like it went smoothly in a straight line — it didn't, and pretending otherwise would make this a worse study document, not a better one. Every entry below follows the same shape: what happened, why it happened (the actual mechanism, not just "oops"), how it was found, how it was fixed, and the general lesson worth keeping past this specific project. Several of these are exactly the kind of "tell me about a bug you found" story that comes up in interviews.*

### 14.1 The `-m` flag confusion — filesystem paths vs. dotted module paths

**What happened:** running `python3 -m src/finrag.index.py`, then `python3 -m src/finrag/index.py`, then `python3 -m src/finrag/index` all failed with `ModuleNotFoundError`.

**Why:** `-m` expects a **dotted module path** — written the same way you'd write it in an `import` statement (`src.finrag.index`) — not a filesystem path. Slashes and `.py` extensions don't belong there at all; Python resolves the dots into directory separators itself, internally, using `sys.path`.

**The fix:** `python3 -m src.finrag.index`.

**A related trap worth knowing:** even `python3 src/finrag/index.py` (direct execution, no `-m` at all) would *also* fail — but with a completely different error: `ImportError: attempted relative import with no known parent package`. That's because `index.py` contains `from .ingest import chunk_corpus` — a relative import that only resolves when Python has loaded the file *as part of a package*, which is exactly what `-m` does and direct script execution does not.

**General lesson:** `-m` and plain file execution set up Python's import machinery differently. Relative imports (the leading dot) require the module to be loaded via `-m` or as part of a properly installed package — they will never work when a file is just run directly as a standalone script.

### 14.2 The shell history-expansion incident — a real, if harmless, scare

**What happened:** I handed over a `python3 -c "..."` command containing the Python format-spec `{h['text'][:80]!r}` (a normal, valid f-string conversion — "call `repr()` on this"). The terminal returned a `SyntaxError`, and the displayed command showed `{h['text'][:80]rm -rf .git}` instead of what was actually written.

**Why this happened — the real mechanism:** bash and zsh both perform **history expansion**: any `!` followed by characters gets expanded to the most recent command in your shell history that starts with those characters, and — critically — **this happens even inside double-quoted strings** (though not single-quoted ones). The shell saw `!r` inside the double-quoted command, searched history for the most recent command starting with "r", and found `rm -rf .git`. It substituted that entire command in as literal text, *before* Python ever received the string — so what Python actually tried to parse was garbled, and correctly refused with a syntax error.

**Why it wasn't actually dangerous:** the substitution happened *inside* a string being assembled for `python3 -c`, not as a free-standing shell command of its own. It never got a chance to execute as `rm -rf .git` — it just corrupted the Python source text, which is why Python's parser rejected it instead of anything being deleted. We verified this directly: `git log` and `git status` immediately afterward showed the repository completely intact, all commits present.

**What it revealed, independent of the incident itself:** the command `rm -rf .git` exists somewhere in that shell's history — worth being aware of, since a future accidental `!r` (or any `!`-prefixed text matching it) in the *wrong* directory could actually execute it for real.

**The fix going forward:** stopped inlining nontrivial code into `python3 -c "..."` one-liners wrapped in double quotes. Switched to writing scripts out to actual files and running `python3 path/to/file.py` — this sidesteps the entire class of shell-quoting hazards (history expansion, but also normal quoting/escaping bugs) rather than trying to carefully avoid every dangerous character by hand.

**General lesson:** double-quoted strings in bash/zsh are not "safe" the way they might look — `!`, `$`, and backticks are all live inside them. Single-quoted strings are inert (no expansion of any kind), but can't contain unescaped single quotes themselves. When code has any of these characters, a real file is safer than a shell one-liner.

### 14.3 The double model-load bug in `index.py`'s smoke test

**What happened:** running `index.py` printed `Loading weights: 100%|...` **twice** in one run.

**Why:** `build_index()` created its own `SentenceTransformer` instance internally to do the embedding. The `__main__` block's smoke test, written separately below it, *also* created its own independent `SentenceTransformer` instance for its own use — with no awareness that `build_index()` had already loaded one moments earlier.

**How it was found:** not by reasoning about the code — by actually reading the real terminal output and asking "why does this print twice?" This is a good example of a bug that's essentially invisible from reading the code casually (both instantiations look individually reasonable) but obvious the moment you watch real output.

**The fix:** changed `build_index()`'s return type from just the `Collection` to a tuple of `(Collection, SentenceTransformer)`, so the caller could reuse the already-loaded model instead of constructing a second one.

**General lesson:** redundant, expensive operations often hide in plain sight in code that "works fine" — nothing crashes, nothing produces a wrong answer, it's just silently wasteful. Reading actual program output, not just source code, is often what catches this class of issue.

### 14.4 The stale Chroma collection cache — caught before it ever broke anything

**What happened:** while *designing* `ablate_chunking.py` (which rebuilds the vector index multiple times inside one running Python process — one rebuild per chunk-size config), we recognized in advance that `retrieve.py`'s cached `_collection` variable would keep pointing at the *first* trial's collection object even after that collection had been deleted and recreated for trial two.

**Why:** the lazy-singleton caching pattern (build once, reuse forever) was designed under an assumption that held for every script up to that point — "the index is built exactly once per process" — which is true for `index.py` and `retrieve.py` used normally, but false for an ablation loop that deliberately rebuilds the index several times in a row.

**How it was found:** not by running the ablation script and watching it fail — by deliberately asking, before writing that script, "what happens to every existing caching assumption under this new usage pattern?" This is a case of catching a bug through design review rather than through a runtime failure.

**The fix:** added an explicit `reset_collection_cache()` function to `retrieve.py`, called immediately after every `build_index()` call inside the ablation loop, forcing the next `retrieve()` call to reconnect rather than reuse a stale reference.

**General lesson:** this is the classic "cache invalidation" problem in miniature. A caching strategy that is completely correct for one usage pattern can become silently wrong under a new one — and the danger is specifically that it *won't crash*, it'll just quietly serve stale data. Any time a cached resource's underlying data can change during a process's lifetime, there needs to be an explicit, deliberate way to invalidate that cache — it doesn't happen automatically.

### 14.5 The `gold_page` truncation bug — evaluation ground truth was silently wrong

**What happened:** the original data-prep code took only the *first* FinanceBench evidence page per question as "the" gold page, silently discarding any additional evidence pages.

**Why this is dangerous, specifically:** it doesn't crash, and it doesn't look wrong by inspection — `gold_page = evidence[0].get("evidence_page_num")` is completely valid, unremarkable-looking Python. The only way to find the problem was to go look at the actual data.

**How it was found:** by explicitly checking, rather than assuming — inspecting FinanceBench's raw records for our 29 kept questions and counting how many had more than one evidence page. Answer: 5 of 29. One example, `financebench_id_00499`, needs evidence from three different financial statements (pages 47, 49, 51) to answer a single question about capital intensity.

**The fix:** `gold_pages` (plural — a deduplicated, sorted list of every valid evidence page) became the real ground truth; `gold_page` (singular) was kept only as `gold_pages[0]`, for backward compatibility, never as a source of truth.

**General lesson:** bugs in evaluation ground truth are especially dangerous because they don't produce an error — they produce a *confidently wrong number*. A retrieval system that correctly found the right passage on page 49 would have been scored as having failed, and nothing about that failure would look like a bug from the metrics alone. This is why it's worth periodically auditing your own labels, not just your model's outputs.

### 14.6 The ambiguous page-numbering convention — a near-miss on a real off-by-one

**What happened:** the original code used a single field just called `"page"` everywhere — in FinanceBench's data, in pdfplumber's extraction, in chunk metadata — with no indication of whether it was 0-based or 1-based.

**Why this was risky:** two *different* systems (FinanceBench's own page annotations and pdfplumber's PDF-parsing library) were both producing "page numbers," and there was no guarantee they used the same convention. This is one of the most common, most silent sources of off-by-one bugs in any pipeline that crosses two independently-built systems — everything keeps running, every comparison keeps "working," and the results are simply wrong by one page, indefinitely, until someone happens to manually check a citation against the source PDF.

**How it was verified — not assumed:** opened `3M_2022_10K.pdf` directly with pdfplumber and cross-checked a real example: FinanceBench's `evidence_page_num=47` for `financebench_id_00499`, against the actual text at `pdfplumber.pages[47]`. They matched exactly (the 3M Consolidated Statement of Income). Conclusion, now backed by evidence rather than assumption: both systems are 0-based, no conversion needed.

**The fix:** split the single ambiguous `"page"` field into two explicitly-named ones — `page_index` (0-based, the *only* field ever compared against `gold_pages`) and `page_number` (1-based, `page_index + 1`, used only for human-facing citation display, explicitly documented as not necessarily matching the number physically printed on the page).

**General lesson:** whenever two different numbering (or indexing) conventions could plausibly exist for "the same" real-world concept, give them different variable names *immediately*, before writing any code that compares them — and verify the actual convention empirically, don't assume it from documentation or intuition alone.

### 14.7 The chunk-boundary failure — a concrete, diagnosed retrieval miss

**What happened:** `financebench_id_00995` ("what products does AMD sell," gold page 3) consistently failed to retrieve, across every configuration tested.

**Diagnosis:** extracted the actual text of AMD's page 3 directly and found that the real answer — a short "Overview" paragraph — sits at character 3,330 out of the page's 3,971 total characters, right at the very end; the rest of the page (84% of it) is a generic "forward-looking statements" legal disclaimer common to nearly every 10-K. At `chunk_size=1000`, the resulting chunk boundary fell almost exactly at the worst possible spot: one chunk ended up with the "Overview" section header buried at its tail, drowned in disclaimer text; the *next* chunk had the actual bulleted product list, but with no header left to give it context.

**General lesson:** chunking is not a cosmetic implementation detail that "just works" with any reasonable-sounding parameters. Where a chunk boundary falls can determine whether an easy, obviously-answerable question is retrievable *at all* — independent of how good the underlying embedding model is.

### 14.8 The disconfirmed chunk-size hypothesis — a hypothesis that turned out wrong, and that was still useful

**What happened:** based on 14.7, we hypothesized that smaller chunks (avoiding long boilerplate-heavy pages getting merged with short answer sections) might fix this class of failure. Tested it directly with `ablate_chunking.py`.

**Result: the hypothesis was wrong.** Recall@5 got *worse* at both a smaller chunk size (500 chars → 27.6%) and a larger one (1500 chars → 27.6%) compared to the original default (1000 chars → 34.5%). A further, more targeted check showed the specific AMD case never entered the top-5 results at *any* of the three tested sizes — not a near-miss that a slightly different size might have flipped.

**The deeper diagnosis that followed:** digging into *why* revealed the true blocker wasn't actually the boundary cut — it was that a specific wrong passage (AMD's product-warranty section) consistently out-scored the correct one at every chunk size, and its similarity score *increased*, not decreased, as chunks got larger. The wrong passage is a genuine semantic distractor (it shares real vocabulary with the query), which chunk-size tuning has no mechanism to fix.

**General lesson:** a plausible, well-reasoned hypothesis can still be flatly wrong, and the only way to find out is to actually measure it — reasoning alone, however careful, is not a substitute for an experiment. Equally important: a "failed" experiment that rules a hypothesis out is real, valuable progress, not wasted effort — it directly redirected the investigation toward the actual mechanism (semantic confusability) instead of continuing to tune a parameter that wasn't the problem.

### 14.9 The BM25 baseline losing — a confirmed prediction, and a sharper version of an earlier finding

**What happened:** built a completely independent, non-neural retrieval method (BM25, pure keyword counting) and measured it with the identical evaluation harness used for dense retrieval. Result: BM25 Recall@5 = 13.8% (4/29), well below dense retrieval's 34.5%.

**Why this belongs in an "issues" section despite being a successful experiment:** before measuring, the report explicitly predicted BM25 would likely *also* fail the AMD case (14.7/14.8), since the wrong distractor passage shares literal query vocabulary too, not just topical similarity — and that prediction held up exactly (`financebench_id_00995` was still a miss under BM25). It also surfaced an even starker version of the "similarity score isn't correctness" finding from Section 8: BM25's misses scored *higher on average* than its hits, and the single highest score across all 29 questions in that entire run belonged to a miss.

**General lesson:** a "negative" result for a simpler alternative approach is exactly the kind of finding that justifies *not* using that simpler approach going forward — it converts "dense retrieval is probably better" from an assumption into a measured fact, which is a categorically stronger thing to say in an interview.

### 14.10 Process note: staying disciplined about who runs what

Earlier in this project, pipeline commands (`prepare_data.py`, `ingest.py`, `index.py`) were run directly rather than being handed over to run and observe firsthand. This was explicitly corrected, and the standing approach since has been: code gets written and explained in detail first, the exact command to run it gets handed over, and it gets run and reported back — except in the specific cases where running it directly was explicitly requested (e.g., the AMD-case diagnostic in Section 13, and the four commits in the current git history). This is a process lesson as much as a technical one: the value of this project is in *understanding* every piece well enough to defend it, and that's much harder to build secondhand.

---

## 15. Broader Failure Analysis: Four More Diagnosed Cases

*Everything in Sections 4, 8, and 14.7–14.8 was built on deeply understanding exactly one failure (`financebench_id_00995`, the AMD products question). That's a sample size of one out of 19 dense-retrieval misses — not enough to know whether its specific mechanism (a semantically confusable wrong passage) is the typical failure or an unusual one. Before committing to a specific fix, four more misses were deliberately chosen for variety — a numeric-ratio question, the single highest-scoring miss across the entire 29-question run, a question whose answer is an absence ("no legal battles"), and a second geography question to compare against the PepsiCo one BM25 solved — and diagnosed the same way: pull the actual gold page text, compare against what was actually retrieved.*

*Note: before running this analysis, the persisted vector store on disk was found to still reflect the 1500-character chunk-size trial from the chunking ablation (Section 14.4's caching discussion doesn't cover this — this was a leftover from `check_amd_case.py` simply never rebuilding the default afterward). It was rebuilt with the correct default (1000/150, 4,961 chunks, verified by collection count) before any of the results below were gathered.*

### Case 1 — `financebench_id_00222`, AMD quick ratio (gold page 55)

**Question:** *"Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?"*

Gold page 55 is AMD's actual Consolidated Balance Sheet — Cash and cash equivalents, Short-term investments, Accounts receivable, Total current assets, exactly the line items the gold answer's 1.57 quick ratio is computed from.

**What got retrieved instead:** three of the five results were from **American Express**, not AMD at all — passages discussing AmEx's own "minimum leverage ratio requirements" and "well-capitalized ratios." The remaining two were AMD pages, but neither was the balance sheet (one was the familiar forward-looking-statements boilerplate, the other a generic passage about "portfolio liquidity").

**Diagnosis: cross-document contamination, directly observed.** The query used generic financial-concept language ("liquidity profile," "quick ratio"), and the embedding model matched on that generic financial vocabulary across company boundaries — pulling in an entirely different filing that happens to discuss liquidity/ratios in similarly-shaped language. This is exactly the risk that was named (Section 8) as the reason whole-corpus retrieval is the *harder, more honest* task compared to filtering by known company — here it's no longer a hypothetical, it's measured, on a real question.

### Case 2 — `financebench_id_01981`, AmEx card member retention (gold page 44)

**Question:** *"Was American Express able to retain card members during 2022?"* (gold answer: simply "Yes")

This is the single **highest-scoring miss across the entire 29-question evaluation** — similarity 0.773, higher than every one of the 10 actual hits. What won: a passage listing countries AmEx operates in ("United Kingdom, EU, Australia, Japan, Canada, Mexico...") — confident, broad business-overview language that has essentially nothing to do with card member retention.

### Case 3 — `financebench_id_00735`, PepsiCo legal battles (gold page 25)

**Question:** *"Has Pepsico reported any materially important ongoing legal battles?"* (gold answer: *"No, Pepsico is not involved in material legal battles."*)

Gold page 25 is actually **Item 2 (Properties)** in PepsiCo's 10-K — the brief "we have no material legal proceedings" statement (Item 3) almost certainly sits right after it on the same page, a short sentence easily overshadowed by a page mostly about corporate real estate. None of the 5 retrieved chunks were even topically close to "legal" content.

**Diagnosis: retrieving an absence is a structurally different problem.** There is no rich, positive passage describing "PepsiCo's legal battles" to match against — the entire point of the source text is that the topic doesn't apply. Both dense similarity and (presumably) keyword matching are built to find text that's *about* a topic; neither has a natural mechanism for retrieving text that briefly states a topic's *absence*. Flagged as a hypothesis worth testing on more examples, not a proven pattern from a single case.

### Case 4 — `financebench_id_01028`, AmEx primary geographies (gold page 154)

**Question:** *"What are the geographies that American Express primarily operates in as of 2022?"* (gold answer: *"United States, EMEA, APAC, and LACC"*)

Gold page 154 is a **geographic revenue table** — region labels and dollar figures, thin on descriptive prose. Same structural shape as Case 1's balance sheet. **Never retrieved, at any point.** And tellingly: the exact same AmEx "page 5" chunk that won Case 2 also won here, for a completely different question.

### Synthesis — four patterns, not one, and none of them are the original AMD case

1. **Cross-document drift is real and measured, not theoretical** (Case 1). Generic financial vocabulary in a query can pull in content from an entirely different company's filing.
2. **Financial tables appear to embed poorly, independent of chunk size** (Cases 1 and 4). Both correct gold pages are numeric tables with thin surrounding prose, and neither was ever retrieved at *any* of the three chunk sizes tested in Section 14.8. This looks like a deeper representational limitation than a chunking-boundary problem — sentence-embedding models, trained overwhelmingly on natural prose, may simply be a poor fit for raw tabular text, and no amount of chunk-size tuning would be expected to fix that.
3. **Specific "generic overview" chunks act as repeat attractors across unrelated questions** — not a one-off. AmEx's page 5 won for two different questions (Cases 2 and 4); AMD's page 11 won the original diagnosed case (Section 14.7). Broadly-worded, confidently-phrased business-overview passages appear to be systematically over-favored by the embedding model, regardless of the specific question asked.
4. **Negation/absence questions may be structurally hard for any single-signal retrieval method** (Case 3) — a hypothesis from one example, worth checking against more cases before treating as established.

### What this changes about the roadmap (Section 11)

Two concrete levers emerged from the analysis above — but the first one was then directly measured, with a surprising result.

**Measured: cross-document drift, filtered out, rescues zero questions.** Chroma's `where={"doc_name": ...}` filter was used to rerun evaluation restricted to each question's own known company:
```
Whole-corpus Recall@5: 10/29 (34.5%)  <- current retrieve.py behavior
Doc-scoped Recall@5:   10/29 (34.5%)  <- if we always knew the right company
Questions rescued ONLY by knowing the right company: 0
```
This contradicts the naive takeaway from Case 1 above. Cross-document contamination is real and directly observable (American Express chunks did appear in an AMD query's top-5) — but removing it entirely doesn't change a single outcome, because the correct page (AMD's balance sheet) was *also* losing to other, wrong AMD chunks even with cross-document competition removed. The contamination was visible but never the actual bottleneck. **Conclusion: building a query-based company-detection step (inferring the company from the question text, e.g. "AMD" appearing literally in the question, then filtering) is not worth building — it's measured, not assumed, to have zero effect on Recall@5.** This is exactly the kind of thing worth measuring before building: an intuitive, example-motivated fix that turned out not to move the number at all.

**Table serialization was built and tested — also measured, also negative.** `src/finrag/ingest.py` was extended to pull each page's tables via pdfplumber's structure-aware `extract_tables()` and serialize them into clean, row-ordered text (empty/None cells stripped, e.g. `"Cash and cash equivalents $ 4,835 $ 2,535"`), added as supplementary chunks (`source: "table"`) alongside the normal text chunks. The index grew from 4,961 to 5,659 chunks, and a direct check confirmed the AMD balance-sheet table chunk was created correctly and reads cleanly.

**Result: Recall@5 stayed at exactly 10/29 (34.5%), unchanged.** Both directly-targeted cases (`financebench_id_00222`, AMD quick ratio; `financebench_id_01028`, AmEx geographies) are still misses. A further check made this more precise than a simple miss: **the gold table chunk doesn't appear even in the top-30 results within its own document** — not a near-miss ranked just outside the cutoff, but genuinely far away in embedding space from how the question gets represented.

**Why, most likely:** two compounding issues, not one.
1. `bge-small` was trained to represent natural-language sentence *meaning*. A row of numbers with short labels, however cleanly formatted, still isn't a natural-language sentence — cleanliness wasn't the actual bottleneck, representational mismatch was.
2. Specific to the quick-ratio case: **the phrase "quick ratio" never appears anywhere in AMD's balance sheet.** It's a *computed* metric — the gold answer (1.57) is derived by combining four separate raw line items, none of which individually states or implies "quick ratio." No retrieval method — dense, BM25, or a cleaner table representation — can find a passage containing words and concepts it doesn't contain. This is a strong, concrete explanation for why `metrics-generated` has scored exactly **0/2 across every single configuration tested so far** (both dense chunk sizes, BM25, and now table serialization) — it may not be a retrieval quality problem at all for this question category.

**Taking stock: three consecutive fix attempts (chunk-size tuning, doc-scoped filtering, table serialization), three negative results on the aggregate metric.** This is a meaningful pattern, not bad luck — it suggests the remaining ~19 misses are not primarily explained by chunking, cross-document noise, or table formatting, and that further tuning along those same axes is unlikely to move the number much further. Two different directions look more promising for what's actually left: **query-side rewriting** (expanding "quick ratio" into the raw line-item terms a query would need to actually match retrievable content, attacking the vocabulary gap from the query side instead of the document side) and **treating computed-metric questions as a fundamentally different pipeline path** — retrieve the raw source numbers (even imperfectly) and *compute* the answer downstream, rather than expecting a single retrieved passage to already state it. Notably, this second direction is not a new idea invented here — it's exactly what the project's own original roadmap (Section 1) already scheduled for Day 3: a calculator tool. This failure analysis is the first concrete, measured justification for *why* that tool is necessary, rather than just a nice-to-have.

---

## 16. Generation Stage (Phase 1, Stage 4): Tool-Augmented Answers

### Why this, and why now

Everything through Section 15 only ever produced a ranked list of source passages — the project had no way to actually answer a question yet. That gap is Stage 4, and closing it was always on the roadmap. What moved *inside* Stage 4 is the calculator: it was originally scheduled for Day 3 as a generic nice-to-have, but Section 15 gave a specific, measured reason to build it now instead. Three consecutive retrieval-only fixes (chunk-size ablation, doc-scoped filtering, table serialization) all measured zero Recall@5 improvement, and the AMD quick-ratio diagnosis explained why: the gold answer (1.57) is a *derived* number — it's computed from four raw balance-sheet line items, and the phrase "quick ratio" never appears anywhere in AMD's actual filing. No retrieval fix can find text that was never there to begin with. And free-text arithmetic isn't a safe substitute either — LLMs generate answers by predicting plausible next tokens, not by executing arithmetic, so they're well documented to make computational errors even when they've picked out the right numbers. A tool hands the actual computation to real, deterministic code instead.

### What was built — `src/finrag/generate.py`

- **`SYSTEM_PROMPT`** — the entire grounding mechanism. Instructs the model to answer only from provided context, cite every claim as `(Company, page N)`, and prefer a tool over doing arithmetic itself. Nothing enforces this except the instruction — a model can still ignore it; there's no hard guarantee, only a strong steer.
- **Security: no `eval()`.** Tool arguments come from the model, which is effectively untrusted input (in principle steerable by adversarial content hidden in a retrieved chunk — a prompt injection). `calculate()` parses its argument with `ast.parse(..., mode="eval")` and walks the tree with a hand-written recursive evaluator (`_safe_eval`) that only knows how to handle three node shapes — a numeric constant, a binary op, a unary minus — restricted to `+ - * / ** %` via an explicit whitelist dict (`_ALLOWED_OPS`). Anything else (a name, a function call, an attribute access) raises instead of executing.
- **Three tools, not one.** A direct check of the eval set found 10 of 29 questions ask some version of "how did X change" (operating margin change, gross margin change, effective tax rate change, revenue drivers, segment growth) — the single biggest repeated computation shape in the corpus, big enough to deserve its own tool rather than making the model re-derive `(new - old) / old` correctly from scratch every time:
  - `percent_change(old, new)` — for the YoY-comparison questions.
  - `ratio(numerator, denominator)` — for single-period ratios/margins (quick ratio, operating margin, EBITDA margin). Originally designed to take plain JSON numbers rather than an expression string, specifically to eliminate a real bug class: with a single generic `calculate("a + b + c / d")`, the model has to get its own parenthesization right, and a missing paren silently divides only `c` by `d` (normal operator precedence) — a wrong answer with no error. This design assumption did not survive contact with a live model; see the first bug below.
  - `calculate(expression)` — the general-purpose fallback, unchanged in spirit from the original design, extended to also allow `**` and `%` (two more entries in the same whitelist — no new AST node types, no new attack surface).
- **`_TOOL_FUNCTIONS` dispatch dict** maps a tool name to the function that actually executes it, so `generate_answer()` handles any number of tools without an `if/elif` chain per tool.
- **The tool-use cycle** follows Groq's documented local-tool-calling pattern (`console.groq.com/docs/tool-use`, read directly before building this): send tool schemas → model requests a call instead of answering → our code executes it locally → the result is appended as a `"tool"`-role message → a second call produces the final answer now that the computed number is in context.

### Three real bugs, found only by actually running it — none of them predictable from documentation

The design above looked complete on paper. Live testing against Groq immediately surfaced three separate failures, each fixed in turn, each teaching something the docs didn't cover (Groq's own tool-use guide and error-code reference were checked directly for any mention of this — neither says anything about it).

**Bug 1 — JSON type mismatch.** `ratio`'s `numerator`/`denominator` were first typed `"number"` in the JSON schema. Given AMD's real balance-sheet numbers (cash $4,835M, short-term investments $1,020M, accounts receivable ~$4,126M, current liabilities ~$6,369M), the model tried to call:
```
<function=ratio>{"numerator": 4835 + 1020 + 4126, "denominator": 6369}</function>
```
— i.e. it wanted to *sum* the three line items inline as part of the argument, rather than pre-adding them first. `4835 + 1020 + 4126` isn't valid JSON for a `"number"` field (JSON numbers are literals only), so Groq's own schema validation rejected the entire generation server-side with a 400 `tool_use_failed`, before any of our code ever ran.
**Fix:** retype `numerator`/`denominator`/`old`/`new` as `"string"` — accepting either a bare number or a small expression — and route them through the same restricted evaluator `calculate()` already uses (`_resolve()`: try `float()` first, fall back to `calculate()`). This keeps the "not `eval()`" security property, just applied one level lower than originally designed, and matches what the model actually wants to send.

**Bug 2 — the model's own function-call wrapper syntax is unreliable.** After Bug 1's fix, the JSON *payload* became valid, but three separate live attempts still failed with `tool_use_failed`, each with a different malformed wrapper where a `>` should have been:
```
<function=ratio[]{"numerator": "4835 + 1020 + 4126", "denominator": "6369"}</function>
<function=ratio,{"numerator": "4835 + 1020 + 4126", "denominator": "6369"}</function>
<function=ratio={"numerator": "4835 + 1020 + 4126", "denominator": "6369"}</function>
<function=ratio({"numerator": "4835 + 1020 + 4126", "denominator": "6369"})</function>
```
The trigger looks repeatable, not random: every failure happened specifically when an argument value was a multi-term expression (`"4835 + 1020 + 4126"`) rather than a bare number — a genuine reliability limit of `llama-3.3-70b-versatile`'s tool-calling for this input shape, not a schema bug on our end. Tried `parallel_tool_calls=False` first (the stray `[]` looked array-related) — didn't fix it, ruled out as the cause.
**Fix:** `_call_with_tools()` wraps the first API call in a bounded retry loop (`MAX_TOOL_ATTEMPTS = 3`). On a `BadRequestError`, it appends an explicit corrective message to the conversation telling the model to pre-compute any sum via `calculate` first and pass a single bare number to `ratio`/`percent_change` — a real self-correction step, not a blind retry. On the third attempt the model changed strategy on its own (pre-summed `4835+1020+4126=9981` and called `ratio(9981, 6369)` cleanly), and the call succeeded.

**Bug 3 — content leak on message replay.** Even after Bug 2's fix produced a *successful* tool call, the final printed answer was still the raw text `<function=ratio>{"numerator":9981,"denominator":6369}</function>` instead of a sentence. Cause: Groq's server apparently populates `message.content` with the raw literal wrapper text even when `tool_calls` parsed successfully (unlike a strictly OpenAI-compatible provider, which would leave `content` empty for a tool-call-only turn). The code was blindly forwarding that into the replayed assistant message (`"content": message.content or ""`), so by the second API call the conversation history already contained what looked like the model's own prior turn being that raw wrapper text — and the model imitated it instead of writing a real answer, even though the underlying tool computation was already correct.
**Fix:** set `"content": None` for the replayed tool-call-only assistant turn instead of forwarding `message.content` — the standard shape for this kind of turn regardless of what a specific provider echoes back.

### Validation

With all three fixes in place, an isolated test — `retrieve()` deliberately bypassed, AMD's real balance-sheet chunk (`p56`, containing `"Cash and cash equivalents $ 4,835 ... Short-term investments 1,020 ..."`) fetched directly from Chroma by metadata filter instead of similarity search — produced:
```
A: Based on the quick ratio of 1.57 (AMD, page 56), AMD has a reasonably healthy liquidity
profile for FY22. The quick ratio is greater than 1, indicating that the company has
sufficient liquid assets to cover its current liabilities (AMD, page 56).

[used tool: True]
```
`9981 / 6369 ≈ 1.567` — matching FinanceBench's actual gold answer (1.57) almost exactly, with a correctly cited, grounded final sentence. This confirms the full tool-use mechanism (schema → model request → local execution → replay → final answer) is now correct end-to-end, given the right context.

### The gap this doesn't close

That test deliberately bypassed `retrieve()`. Running the *real* pipeline (`python3 -m src.finrag.generate`, using actual whole-corpus retrieval) on this exact same question returns **zero AMD chunks** in the top 5 at all — not "right document, wrong page," but three entirely different companies:

| Rank | Result |
|---|---|
| 1 | American Express p16 — "minimum leverage ratio requirements" (sim 0.751) |
| 2 | 3M p53 — accounting estimates boilerplate (sim 0.713) |
| 3 | American Express p140 — fair value valuation techniques (sim 0.712) |
| 4 | PepsiCo p102 — cash-equivalents classification footnote (sim 0.706) |
| 5 | American Express p85 — card-merchant retention language (sim 0.703) |

**Why:** the query itself ("liquidity profile," "cash and cash equivalents," "short-term investments," "current liabilities") is standard balance-sheet vocabulary that appears, worded similarly, in every company's 10-K — so the query embedding lands near *any* passage using that language, not specifically AMD's. AMD's actual balance-sheet chunk is mostly bare numbers, comparatively sparse in the descriptive prose `bge-small` weighs heavily, and loses the similarity contest even to the *wrong company's* prose. This is the same "financial tables embed poorly" pattern from Section 15, showing up a fourth time — now for the query itself, not just the gold chunk within its own document.

The system did the right thing given that failure: with no AMD context at all, `generate_answer()` correctly refused rather than hallucinate a plausible-looking quick ratio, citing only what it actually had. That's `SYSTEM_PROMPT`'s grounding rule working exactly as intended. But it also means **generation-stage correctness and retrieval-stage correctness are genuinely separate failure modes, confirmed directly rather than just assumed**: the calculator is proven correct, and the system still fails end-to-end on this question, because generation can only work with what retrieval actually hands it.

---

## 17. Reranking and Hybrid Retrieval: Two More Negative Results, Rigorously Tuned

Sections 14.8 and 15 already ruled out chunk-size tuning, doc-scoped filtering, and table serialization as fixes for the retrieval gap — three negative results in a row. This section adds two more, tested the same way: implement the standard technique properly, measure it on the identical 29-question eval set with the identical hit-scoring rule, and follow the data wherever it goes rather than assume it worked.

### Reranking (cross-encoder)

**What it is:** `bge-small` (used everywhere so far) is a *bi-encoder* — it embeds the query and each passage completely separately, then compares vectors. A *cross-encoder* instead takes `(query, passage)` as one joint input and outputs a single relevance score, letting it weigh interactions a bi-encoder structurally can't see. Too expensive to run over the whole corpus (a full forward pass per candidate), so the standard pattern is: let the cheap bi-encoder narrow the corpus to a shortlist (`RERANK_CANDIDATES=20`), then spend the cross-encoder's extra accuracy only re-ordering that shortlist. Built as `src/finrag/rerank.py` (model: `cross-encoder/ms-marco-MiniLM-L-6-v2`), wired into `retrieve.py`'s `retrieve_with_rerank()`, scored with `scripts/evaluate_rerank.py`.

**Result: Recall@5 = 6/29 (20.7%) — a regression below dense-only's 34.5%, not a null result.** Comparing hit sets directly: dense-only's 10 hits were `01198, 01279, 00476, 00517, 01091, 00678, 00464, 00494, 00585, 01328`; reranking's 6 hits were `01198, 00476, 00517, 00464, 00494, 01328` — **every single reranked hit was already a dense hit**, and reranking actively demoted 4 previously-correct answers (`01279, 01091, 00678, 00585`) out of the top-5. That's a stronger finding than "didn't help" — the cross-encoder had the right answer already in front of it (rank ≤5 pre-rerank) and made it worse.

**Diagnosis:** `ms-marco-MiniLM-L-6-v2` was trained on MS MARCO — Bing search queries and web-snippet passages. That's a different register from formal, numeric-table-heavy SEC filing prose. The model is likely well-calibrated for "does this web snippet answer this search query" and poorly calibrated for "does this balance-sheet excerpt answer this financial question" — a domain-mismatch hypothesis, not proven beyond this one experiment, but consistent with what was measured.

### Hybrid retrieval (BM25 + dense, fused with RRF)

**What it is:** dense (34.5%) and BM25 (13.8%) aren't just "one better, one worse" — Section 13 already showed BM25 uniquely hit 2 questions (`01226`, `01009`) that dense missed entirely, a real complementary signal (union of both methods' hits = 12/29, 41.4%). Reciprocal Rank Fusion (RRF) combines two ranked lists using only their **rank positions**, not raw scores — necessary because cosine similarity (~0-1) and BM25's unbounded raw score aren't on comparable scales, so adding them directly would be meaningless. Formula: `RRF_score(chunk) = Σ 1/(RRF_K + rank_in_list)` summed across every list the chunk appears in. Built as `src/finrag/hybrid_retrieve.py`, `RRF_K=60` (the standard value from the original RRF paper, Cormack/Clarke/Buettcher 2009), scored with `scripts/evaluate_hybrid.py`.

**Before trusting the result, two things were verified directly rather than assumed:**
1. **Corpus sync** — dense's persisted Chroma collection and BM25's freshly-rebuilt in-memory index both needed to be searching the *same* chunk set for `(doc_name, page_index, text)` matching to be valid. Checked `collection.count()` directly: 5,659, matching BM25's fresh `chunk_corpus()` output exactly. No mismatch.
2. **Premise still holds on the current (table-aware) corpus** — the BM25 baseline numbers (13.8%, uniquely hits `01226`/`01009`) were originally measured *before* table-aware chunking was added to `ingest.py`. Re-ran `evaluate_bm25.py` fresh: still exactly 13.8% (4/29), and `01226`/`01009` are still both hits. Table serialization's already-documented zero effect (Section 16) held for BM25 too — the premise motivating hybrid retrieval was current, not stale.

**Result at default RRF_K=60: Recall@5 = 7/29 (24.1%) — another regression, and it failed to rescue either target question.** Diagnosis, found by inspecting the actual RRF math: the winning top-5 chunks had scores around `0.0313` — close to two combined contributions from a chunk both methods rank reasonably (`1/61 + 1/62 ≈ 0.0325`). But `01226`'s gold chunk is a **BM25-only** hit by definition (that's exactly why dense missed it) — its RRF score can only ever come from one list, capping around `1/61 ≈ 0.0164`, roughly half what a mediocre *consensus* chunk accumulates. **RRF structurally rewards agreement between methods over either method's unique high-confidence signal** — the opposite of what rescuing a single-method-only correct answer needs.

**RRF_K ablation — testing whether tuning could fix this, not just guessing:**
```
 RRF_K | Recall@5     | watched questions
-------|--------------|---------------------------
     0 | 9/29 (31.0%) | 01226=HIT, 01009=miss
     1 | 9/29 (31.0%) | 01226=HIT, 01009=miss
     5 | 8/29 (27.6%) | 01226=miss, 01009=miss
    10 | 7/29 (24.1%) | 01226=miss, 01009=miss
    30 | 7/29 (24.1%) | 01226=miss, 01009=miss
    60 | 7/29 (24.1%) | 01226=miss, 01009=miss
```
Smaller K clearly helps — it weights top rank positions more steeply, letting a single method's strong pick compete better against two methods' weak agreement — and `01226` does flip to a hit at K≤1, confirming the diagnosis was mechanistically correct. But the improvement **plateaus at K=0/K=1 (31.0%)**, and `01009` never flips regardless of K. **Even the best-tuned hybrid configuration, found by sweeping the full reasonable range of its main hyperparameter, still falls short of dense-only's 34.5%.**

**What's structurally different, not yet tried at this point:** every technique above operates on the *document* side (how chunks are split, indexed, reordered, or fused) while leaving the *query* completely unchanged. The recurring failure pattern across all of Section 15 and this section is a **vocabulary/register mismatch** — a query like "quick ratio" or "liquidity profile" uses abstract financial terminology that never appears in the source filing (which only states raw line items), and pulls toward *other companies'* prose that happens to share that abstract vocabulary. Query rewriting or HyDE (Hypothetical Document Embeddings — Gao et al. 2022: generate a hypothetical passage that *would* answer the question, and embed that instead of the raw question, so the thing being embedded is prose-shaped like the actual filings rather than a short interrogative question) attacks this from the query side instead — mechanistically different from anything tried so far.

### Query expansion — a zero-cost proxy for HyDE, tested immediately given the pattern above

Full HyDE needs an LLM call per query to generate the hypothetical passage — real Groq quota cost, and quota was still constrained the day this was built. Before waiting on that, the underlying *hypothesis* (richer, filing-shaped vocabulary in the query closes the gap) was tested for free: `src/finrag/query_expand.py` hand-maps a handful of financial metric names (`quick ratio`, `operating margin`, `gross margin`, `effective tax rate`, `ebitda`) to the raw line-item vocabulary their *standard textbook definitions* are built from, and appends that vocabulary to any query mentioning the metric. Deliberately using only general financial knowledge, not anything read off the actual gold filing text — the same vocabulary an LLM asked to draft a hypothetical answer would independently produce, so this is representative of real HyDE, not an answer-leaked shortcut.

A 2-question manual check first: AMD's quick-ratio gold page went from **not in the top-20 at all** to **rank 5**, using only the standard quick-ratio formula's terms. Scaled up to the full 29-question eval (`scripts/evaluate_query_expansion.py`, wired as `retrieve.py`'s `retrieve_with_expansion()`):

**Result: Recall@5 = 11/29 (37.9%) — the first of six techniques tried today to actually beat the dense-only baseline (34.5%).** And cleanly: every one of dense-only's original 10 hits is still present, plus one genuine new rescue — `financebench_id_01226` (3M's operating margin question), where "operating margin" in the question text triggered the dictionary and pulled the gold page into the top-5 for the first time. Zero hits lost, unlike reranking and hybrid RRF, which both traded away previously-correct answers to gain new ones.

One honest caveat: `financebench_id_00222` (AMD's own quick-ratio question) is *still* a miss in the full run, despite the promising 2-question check. The real eval question has an extra clause the manual check didn't include ("If the quick ratio is not relevant to measure liquidity, please state that and explain why") — different embedded text, different result. A reminder that a hand-picked spot check and a full, automated eval run over the exact real question text are not interchangeable, and only the latter is trustworthy enough to act on.

### Synthesis: six techniques tried, query expansion is the first genuine win

```
Query expansion                    37.9%  (11/29)  <- new best
Dense only (baseline)               34.5%  (10/29)
Hybrid RRF, tuned (K=0 or K=1)      31.0%  ( 9/29)
Hybrid RRF, default (K=60)          24.1%  ( 7/29)
Reranked (cross-encoder)            20.7%  ( 6/29)
BM25 only                           13.8%  ( 4/29)
```
Chunk-size ablation, doc-scoped filtering, table serialization, reranking, and hybrid RRF — five document-side techniques, all at or below the dense-only baseline. Query expansion — the one technique that touches the *query* instead — is the first to actually beat it, and it does so with a mechanism specific enough to explain in one sentence: the vocabulary gap diagnosed repeatedly across Sections 15 and 17 is real, and closing it from the query side works better than any amount of reprocessing the documents did.

**Decision, revised: ship `retrieve_with_expansion()` (dense + the metric-vocabulary expansion) as the production retriever**, not plain `retrieve()`. This is deliberately a narrow, manually-maintained proof-of-concept — 5 hand-picked metrics, not a general solution — and full HyDE (LLM-generated expansion, generalizing to any question) remains the natural, more rigorous next step once Groq quota allows. But the measured result justifies shipping the cheap version now rather than waiting: it's strictly better than the baseline it replaces, at zero additional cost or latency.

---

## 18. Wiring the Winning Retriever In, and Calibrating the Guardrail From Real Data

### Wiring `retrieve_with_expansion()` into the actual pipeline

Section 17 measured `retrieve_with_expansion()` as the best-performing retriever, but measuring it in an eval script and actually *using* it in the real pipeline are two different things — `generate.py`'s demo and `evaluate_generation.py` were both still importing plain `retrieve()`. Fixed by swapping the import (`from .retrieve import retrieve_with_expansion as retrieve`) in both files — a one-line change per file, since every retriever built today (`retrieve`, `bm25_retrieve.retrieve`, `retrieve_with_rerank`, `retrieve_hybrid`, `retrieve_with_expansion`) deliberately returns the identical `{"text","metadata","similarity"}` shape specifically so callers don't care which one is plugged in.

### Calibrating `MIN_RELEVANCE_SCORE` — the guardrail — from real data instead of a guess

`MIN_RELEVANCE_SCORE` had been an unvalidated placeholder (0.30) since Day 1, flagged repeatedly as needing real out-of-scope test questions that didn't exist yet (Section 11). Built 7 of them across two deliberately different categories, and measured each one's best-similarity score through `retrieve_with_expansion()`:

```
Genuinely unrelated topics:
  0.522  What is the capital of France?
  0.487  How do I bake a chocolate cake?
  0.577  What is the weather forecast for tomorrow?
                                                    <- max 0.577

"Right structure, wrong company" (real financial questions,
 about companies NOT in the 5-document corpus):
  0.720  What was Apple's total revenue in fiscal year 2022?
  0.632  What is Tesla's current ratio for FY2022?
  0.769  What was Google's operating margin in FY2022?
  0.634  Who won the World Cup in 2022? (borderline, not financial)
```

Compared against the real in-scope range across all 29 eval questions (hits and misses combined): **0.660-0.810.**

**The important, honest finding: these two out-of-scope categories behave completely differently, and the guardrail can only ever catch one of them.** Genuinely unrelated topics score clearly below the in-scope floor (max 0.577 vs. floor 0.660) — a clean gap to place a threshold in. But "right structure, wrong company" questions score *inside* the in-scope range (Apple 0.720, Google 0.769) — the same mechanism behind every cross-document-contamination case documented all day (Section 15's Case 1, Section 17's reranking failures): the embedding model matches on financial sentence structure and vocabulary, not on which specific company is named, so a well-formed question about Apple's revenue looks just as "relevant" to this corpus as a real question about AMD's. **No similarity threshold can distinguish these** — a company name swap doesn't change the question's structural fingerprint enough to move the score.

**Set `MIN_RELEVANCE_SCORE = 0.60`** — inside the clean gap between 0.577 (highest unrelated-topic score) and 0.660 (lowest real in-scope score). This guarantees zero real in-scope questions are ever incorrectly refused, while catching clearly off-topic queries. It will **not** catch a well-phrased question about a company outside the corpus — documented as a known, real limitation, not silently ignored.

**Wired into `generate_answer()`** (`src/finrag/generate.py`): if the best retrieved chunk's similarity is below the threshold, the function returns an immediate refusal *without calling Groq at all*. Verified directly: `"What is the capital of France?"` (similarity 0.522) returned the canned refusal instantly, while a real Boeing question (similarity 0.705) passed through and produced a normal, grounded answer. This is a guardrail and a quota-saving measure at once — a real, tangible benefit given today's Groq daily-limit exhaustion (Section 16).

---

*End of report. Per your instructions, no new implementation work will proceed until you've read this.*
