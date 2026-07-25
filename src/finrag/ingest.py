"""
Stage 1 — Ingestion: PDF -> page text -> chunks with metadata.

Pipeline: for each PDF in config.PDF_DIR, extract text per page (so we never
lose the page number), then split each page's text into overlapping chunks.
Every chunk carries (doc_name, company, page_index, page_number, source)
metadata so later stages can cite exactly where an answer came from.

Tables get a second, separate representation. pdfplumber's normal
extract_text() flattens a page's tables into linear reading-order text
mixed in with everything else on the page — fine for prose, but financial
statement tables (thin on words, dense with numbers) embed poorly this
way (see docs/finrag_learning_report.md, Section 15: AMD's balance sheet
and AmEx's geographic revenue table were never retrieved at any chunk
size). extract_pages() also pulls each page's tables via pdfplumber's
structure-aware extract_tables(), and chunk_document() serializes them
into their own, separate chunks (metadata source="table") IN ADDITION TO
the normal text chunks (source="text") — not a replacement. Having two
representations of the same content is a deliberate redundancy, not a
bug: the normal flattened text still exists for prose-style questions,
and the isolated table text gives a cleaner, more focused embedding for
questions the table itself answers.
"""

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def extract_pages(pdf_path) -> list[dict]:
    """Return one dict per page: {"page_index": int, "text": str, "tables": list}.

    page_index is 0-based — pdfplumber's native indexing, which also happens
    to match FinanceBench's `evidence_page_num` (verified against the gold
    data in scripts/prepare_data.py). This is the convention used everywhere
    internally: chunk metadata, gold labels, retrieval hit-checking. It is
    NOT the same as `page_number` (1-based), which exists only for
    human-facing citations and is not guaranteed to equal the page number
    printed on the physical page (10-Ks often have unnumbered cover pages,
    roman-numeral TOCs, etc.) — it's the PDF's page_index + 1.

    "tables" is whatever pdfplumber's extract_tables() finds on this page:
    a list of tables, each table a list of rows, each row a list of cell
    strings (or None for empty cells). Most pages have none (empty list).
    Table detection isn't perfect — borderless or irregularly-formatted
    tables may not be found — this is a best-effort structural extraction,
    not a guarantee.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""  # extract_text() returns None for blank/image-only pages
            tables = page.extract_tables()
            pages.append({"page_index": i, "text": text, "tables": tables})
    return pages


def serialize_table(table: list[list]) -> str:
    """Turn one pdfplumber-extracted table into plain text.

    pdfplumber's raw cells are messier than a clean "label: value" table
    would suggest — currency symbols often land in their own cell, and
    empty/merged cells come through as None or "" (verified directly
    against AMD's balance sheet and AmEx's geographic revenue table — see
    docs/finrag_learning_report.md, Section 15's "What this changes about
    the roadmap"). Reconstructing proper column headers isn't reliable
    enough to do here, so this deliberately does something simpler: drop
    the empty cells and join what's left, one line per row. E.g. a row
    like ['Cash and cash equivalents', '$', '4,835', '', '$', '2,535']
    becomes "Cash and cash equivalents $ 4,835 $ 2,535" — not a fully
    labeled sentence, but a clean, row-ordered line instead of the
    scrambled, page-wide flattening extract_text() would have produced.
    """
    lines = []
    for row in table:
        cells = [str(cell).strip() for cell in row if cell not in (None, "")]
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)


def chunk_document(doc_name: str, company: str) -> list[dict]:
    """Extract + chunk one document. Returns a list of chunk dicts, each with
    'text' and a 'metadata' dict (doc_name, company, page)."""
    pdf_path = config.PDF_DIR / f"{doc_name}.pdf"
    pages = extract_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = []
    for page in pages:
        base_metadata = {
            "doc_name": doc_name,
            "company": company,
            "page_index": page["page_index"],           # 0-based, compare against gold_pages
            "page_number": page["page_index"] + 1,       # 1-based, for citations shown to users
        }

        if page["text"].strip():
            for piece in splitter.split_text(page["text"]):
                chunks.append({"text": piece, "metadata": {**base_metadata, "source": "text"}})
        # else: skip blank pages (e.g. intentional page breaks, cover pages)

        # Tables get their own, ADDITIONAL chunks — see module docstring and
        # serialize_table() for why this is deliberate duplication, not a
        # replacement of the normal text chunks above.
        for table in page["tables"]:
            table_text = serialize_table(table)
            if not table_text.strip():
                continue
            for piece in splitter.split_text(table_text):
                chunks.append({"text": piece, "metadata": {**base_metadata, "source": "table"}})
    return chunks


def chunk_corpus() -> list[dict]:
    """Chunk every document in config.CORPUS_DOCS. Returns one flat list."""
    all_chunks = []
    for doc_name, label in config.CORPUS_DOCS.items():
        doc_chunks = chunk_document(doc_name, label)
        print(f"[ok] {doc_name}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_corpus()
    print(f"\nTotal chunks: {len(chunks)}")
    print("\n--- sample chunk ---")
    print(chunks[0]["metadata"])
    print(chunks[0]["text"][:300])
