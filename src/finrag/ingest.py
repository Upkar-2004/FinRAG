"""
Stage 1 — Ingestion: PDF -> page text -> chunks with metadata.
"""

from pathlib import Path
from typing import Literal, TypedDict 

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config



ExtractedTable = list[list[str | None]]

class ExtractedPage(TypedDict):
    page_index: int
    text: str
    tables: list[ExtractedTable]

class ChunkMetadata(TypedDict):
    doc_name: str
    company: str
    page_index: int
    page_number: int
    source: Literal["text", "table"]


class DocumentChunk(TypedDict):
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    
def extract_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Return one dict per page: {"page_index": int, "text": str, "tables": list}.
    """
    
    pages: list[ExtractedPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = ( page.extract_text() or "" )  # extract_text() returns None for blank/image-only pages
            tables = page.extract_tables()
            pages.append(
                {
                    "page_index": page_index, 
                    "text": text, 
                    "tables": tables
                    }
            )

    return pages



def serialize_table(table: ExtractedTable) -> str:
    """Turn one pdfplumber-extracted table into plain text.
    """
    lines = []
    for row in table:
        cells = [str(cell).strip() for cell in row if cell not in (None, "")]
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)




def chunk_document(doc_name: str, company: str) -> list[DocumentChunk]:
    """Extract + chunk one document. Returns a list of chunk dicts, each with
    'text' and a 'metadata' dict (doc_name, company, page)."""

    pdf_path = config.PDF_DIR / f"{doc_name}.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}"
                                "Run 'python scripts/prepare_data.py' first")

    
    pages = extract_pages(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[DocumentChunk] = []

    for page in pages:
        base_metadata = {
            "doc_name": doc_name,
            "company": company,
            "page_index": page["page_index"],  # 0-based, compare against gold_pages
            "page_number": page["page_index"] + 1,  # 1-based, for citations shown to users
        }

        if page["text"].strip():
            for piece_index, piece in enumerate(splitter.split_text(page["text"])):
                chunk_id = (
                    f"{doc_name}_p{page['page_index']}_text_c{piece_index}"
                )
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": piece,
                        "metadata": {**base_metadata, "source": "text"},
                    }
                )

      
        for table_index, table in enumerate(page["tables"]):
            table_text = serialize_table(table)
            if not table_text.strip():
                continue
            for piece_index, piece in enumerate(splitter.split_text(table_text)):
                chunk_id = (
                    f"{doc_name}_p{page['page_index']}_table_t{table_index}_c{piece_index}"
                )
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": piece,
                        "metadata": {**base_metadata, "source": "table"},
                    }
                )

    return chunks




def chunk_corpus() -> list[DocumentChunk]:
    """Chunk every document in config.CORPUS_DOCS. Returns one flat list."""

    all_chunks: list[DocumentChunk] = []
    for doc_name, label in config.CORPUS_DOCS.items():
        doc_chunks = chunk_document(doc_name, label)
        print(f"[ok] {doc_name}: {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)
    return all_chunks


def main() -> None:
    """Build and save the canonical corpus artifacts."""
    from .corpus_store import save_corpus

    chunks = chunk_corpus()
    manifest = save_corpus(chunks)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Saved chunks: {config.CHUNKS_PATH}")
    print(f"Saved manifest: {config.MANIFEST_PATH}")
    print(f"Corpus fingerprint: {manifest['corpus_fingerprint']}")
    print("\n--- sample chunk ---")
    print(chunks[0]["chunk_id"])
    print(chunks[0]["metadata"])
    print(chunks[0]["text"][:300])


if __name__ == "__main__":
    main()
