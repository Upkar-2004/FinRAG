"""
Stage 1 — Ingestion: PDF -> page text -> chunks with metadata.
"""

import re
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfplumber.page import Page

import config

TableRow = list[str | None]

TABLE_CONTEXT_HEIGHT = 130
TABLE_BLOCK_GAP = 6
TABLE_UNIT_PATTERN = re.compile(
    r"\([^)]*\b(?:thousand|million|billion)s?\b[^)]*\)",
    re.IGNORECASE,
)


class TableData(TypedDict):
    table_id: str
    doc_name: str
    company: str
    page_index: int
    page_number: int
    table_index: int
    bbox: list[float]
    title: str
    headers: str
    units: list[str]
    rows: list[TableRow]


class TablePiece(TypedDict):
    text: str
    row_start: int
    row_end: int
    context_length: int


class ExtractedPage(TypedDict):
    page_index: int
    text: str
    tables: list[TableData]


class ChunkMetadata(TypedDict):
    doc_name: str
    company: str
    page_index: int
    page_number: int
    source: Literal["text", "table"]
    table_id: NotRequired[str]
    table_index: NotRequired[int]
    row_start: NotRequired[int]
    row_end: NotRequired[int]
    context_length: NotRequired[int]


class DocumentChunk(TypedDict):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


def extract_table_context(
    page: Page,
    bbox: tuple[float, float, float, float],
) -> tuple[str, str, list[str]]:
    """Extract a nearby title, column headers, and units from above a table."""
    x0, table_top, x1, _ = bbox
    context_top = max(0, table_top - TABLE_CONTEXT_HEIGHT)
    context_area = page.crop((x0, context_top, x1, table_top))
    lines = context_area.extract_text_lines(strip=True, return_chars=False)

    if not lines:
        return "", "", []

    blocks = []
    current_block = []

    for line in lines:
        if current_block and line["top"] - current_block[-1]["bottom"] >= TABLE_BLOCK_GAP:
            blocks.append(current_block)
            current_block = []
        current_block.append(line)

    if current_block:
        blocks.append(current_block)

    header_block = blocks[-1]
    headers = "\n".join(line["text"].strip() for line in header_block)

    title = ""
    for block in reversed(blocks[:-1]):
        block_text = "\n".join(line["text"].strip() for line in block)
        if TABLE_UNIT_PATTERN.sub("", block_text).strip():
            title = block_text
            break

    context_text = "\n".join(line["text"] for line in lines)
    units = list(dict.fromkeys(TABLE_UNIT_PATTERN.findall(context_text)))

    return title, headers, units


def extract_pages(pdf_path: Path, doc_name: str, company: str) -> list[ExtractedPage]:
    """Return one dict per page: {"page_index": int, "text": str, "tables": list}."""

    pages: list[ExtractedPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = (
                page.extract_text() or ""
            )  # extract_text() returns None for blank/image-only pages
            tables: list[TableData] = []

            for table_index, table in enumerate(page.find_tables()):
                title, headers, units = extract_table_context(page, table.bbox)
                tables.append(
                    {
                        "table_id": f"{doc_name}_p{page_index}_t{table_index}",
                        "doc_name": doc_name,
                        "company": company,
                        "page_index": page_index,
                        "page_number": page_index + 1,
                        "table_index": table_index,
                        "bbox": list(table.bbox),
                        "title": title,
                        "headers": headers,
                        "units": units,
                        "rows": table.extract(),
                    }
                )

            pages.append({"page_index": page_index, "text": text, "tables": tables})

    return pages


def serialize_table_context(table: TableData) -> str:
    """Combine the title, headers, and units shared by every table chunk."""
    lines = []

    if table["title"]:
        lines.append(table["title"])

    if table["headers"]:
        lines.append(table["headers"])

    context_text = "\n".join(lines)

    for unit in table["units"]:
        if unit not in context_text:
            lines.append(unit)

    return "\n".join(lines)


def serialize_table_row(row: TableRow) -> str:
    """Convert one extracted table row into searchable text."""
    cells = [str(cell).strip() for cell in row if cell not in (None, "")]
    return " ".join(cells)


def build_table_piece(
    context: str,
    rows: list[tuple[int, str]],
) -> TablePiece:
    """Combine shared table context with a group of complete rows."""
    text_parts = [context]
    text_parts.extend(row_text for _, row_text in rows)

    return {
        "text": "\n".join(part for part in text_parts if part),
        "row_start": rows[0][0],
        "row_end": rows[-1][0],
        "context_length": len(context),
    }


def chunk_table(table: TableData) -> list[TablePiece]:
    """Group complete table rows without exceeding the target size when possible."""
    context = serialize_table_context(table)
    pieces: list[TablePiece] = []
    current_rows: list[tuple[int, str]] = []

    for row_index, row in enumerate(table["rows"]):
        row_text = serialize_table_row(row)

        if not row_text:
            continue

        candidate_rows = [*current_rows, (row_index, row_text)]
        candidate_piece = build_table_piece(context, candidate_rows)

        if current_rows and len(candidate_piece["text"]) > config.CHUNK_SIZE:
            pieces.append(build_table_piece(context, current_rows))
            current_rows = [(row_index, row_text)]
        else:
            current_rows = candidate_rows

    if current_rows:
        pieces.append(build_table_piece(context, current_rows))

    return pieces


def chunk_document(doc_name: str, company: str) -> list[DocumentChunk]:
    """Extract + chunk one document. Returns a list of chunk dicts, each with
    'text' and a 'metadata' dict (doc_name, company, page)."""

    pdf_path = config.PDF_DIR / f"{doc_name}.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}Run 'python scripts/prepare_data.py' first"
        )

    pages = extract_pages(pdf_path, doc_name, company)

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
                chunk_id = f"{doc_name}_p{page['page_index']}_text_c{piece_index}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": piece,
                        "metadata": {**base_metadata, "source": "text"},
                    }
                )

        for table in page["tables"]:
            for table_piece in chunk_table(table):
                row_start = table_piece["row_start"]
                row_end = table_piece["row_end"]

                chunk_id = f"{table['table_id']}_r{row_start}-{row_end}"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": table_piece["text"],
                        "metadata": {
                            **base_metadata,
                            "source": "table",
                            "table_id": table["table_id"],
                            "table_index": table["table_index"],
                            "row_start": row_start,
                            "row_end": row_end,
                            "context_length": table_piece["context_length"],
                        },
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
