"""
Stage 1 — Ingestion: PDF -> page text -> chunks with metadata.

Pipeline: for each PDF in config.PDF_DIR, extract text per page (so we never
lose the page number), then split each page's text into overlapping chunks.
Every chunk carries (doc_name, company, page_index, page_number) metadata so
later stages can cite exactly where an answer came from.
"""

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def extract_pages(pdf_path) -> list[dict]:
    """Return one dict per page: {"page_index": int, "text": str}.

    page_index is 0-based — pdfplumber's native indexing, which also happens
    to match FinanceBench's `evidence_page_num` (verified against the gold
    data in scripts/prepare_data.py). This is the convention used everywhere
    internally: chunk metadata, gold labels, retrieval hit-checking. It is
    NOT the same as `page_number` (1-based), which exists only for
    human-facing citations and is not guaranteed to equal the page number
    printed on the physical page (10-Ks often have unnumbered cover pages,
    roman-numeral TOCs, etc.) — it's the PDF's page_index + 1.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""  # extract_text() returns None for blank/image-only pages
            pages.append({"page_index": i, "text": text})
    return pages


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
        if not page["text"].strip():
            continue  # skip blank pages (e.g. intentional page breaks, cover pages)
        for piece in splitter.split_text(page["text"]):
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        "doc_name": doc_name,
                        "company": company,
                        "page_index": page["page_index"],           # 0-based, compare against gold_pages
                        "page_number": page["page_index"] + 1,      # 1-based, for citations shown to users
                    },
                }
            )
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
