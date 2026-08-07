"""Save and load the canonical chunks shared by every retriever."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import config

from .ingest import DocumentChunk

SCHEMA_VERSION = 1
PIPELINE_VERSION = 1


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _financebench_revision() -> str | None:
    """Return the checked-out FinanceBench revision when available."""
    if not (config.FINANCEBENCH_SRC / ".git").exists():
        return None

    result = subprocess.run(
        ["git", "-C", str(config.FINANCEBENCH_SRC), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_chunks(chunks: list[DocumentChunk]) -> None:
    """Reject empty, blank, or duplicate canonical chunks."""
    if not chunks:
        raise ValueError("Cannot save an empty corpus.")

    seen_ids: set[str] = set()
    for position, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        if not chunk_id:
            raise ValueError(f"Chunk at position {position} has an empty chunk_id.")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        if not chunk["text"].strip():
            raise ValueError(f"Chunk {chunk_id} has blank text.")
        seen_ids.add(chunk_id)


def _document_manifest() -> list[dict]:
    """Describe the configured PDF files and their content hashes."""
    documents = []
    for doc_name, company in config.CORPUS_DOCS.items():
        pdf_path = config.PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        documents.append(
            {
                "doc_name": doc_name,
                "company": company,
                "pdf_sha256": file_sha256(pdf_path),
            }
        )
    return documents


def save_corpus(chunks: list[DocumentChunk]) -> dict:
    """Write chunks.jsonl and its manifest, returning the manifest."""
    _validate_chunks(chunks)
    documents = _document_manifest()
    financebench_revision = _financebench_revision()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chunks_temp = Path(f"{config.CHUNKS_PATH}.tmp")
    with chunks_temp.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    chunks_temp.replace(config.CHUNKS_PATH)

    chunks_hash = file_sha256(config.CHUNKS_PATH)
    fingerprint_data = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "chunks_sha256": chunks_hash,
        "documents": documents,
    }
    fingerprint_json = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
    corpus_fingerprint = hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_fingerprint": corpus_fingerprint,
        "chunk_count": len(chunks),
        "artifacts": {
            "chunks": {
                "path": config.CHUNKS_PATH.name,
                "sha256": chunks_hash,
            }
        },
        "ingestion": {
            "pipeline_version": PIPELINE_VERSION,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "source": {
            "financebench_revision": financebench_revision,
            "documents": documents,
        },
    }

    manifest_temp = Path(f"{config.MANIFEST_PATH}.tmp")
    manifest_temp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_temp.replace(config.MANIFEST_PATH)
    return manifest


def load_manifest() -> dict:
    """Load the manifest and verify its version and chunking settings."""
    if not config.MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Corpus manifest not found: {config.MANIFEST_PATH}. "
            "Run `python -m src.finrag.ingest` first."
        )

    manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("The corpus manifest uses an unsupported schema version.")

    ingestion = manifest.get("ingestion", {})
    expected_settings = {
        "pipeline_version": PIPELINE_VERSION,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }
    if ingestion != expected_settings:
        raise RuntimeError(
            "The saved corpus does not match the current ingestion settings. "
            "Rebuild it with `python -m src.finrag.ingest`."
        )

    saved_documents = [
        (document["doc_name"], document["company"])
        for document in manifest.get("source", {}).get("documents", [])
    ]
    if saved_documents != list(config.CORPUS_DOCS.items()):
        raise RuntimeError(
            "The saved corpus does not match config.CORPUS_DOCS. "
            "Rebuild it with `python -m src.finrag.ingest`."
        )
    return manifest


def load_corpus() -> tuple[dict, list[DocumentChunk]]:
    """Load and validate the canonical manifest and chunks."""
    manifest = load_manifest()

    if manifest["source"]["documents"] != _document_manifest():
        raise RuntimeError(
            "The configured PDF files do not match the corpus manifest. Rebuild the corpus."
        )

    if not config.CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Canonical chunks not found: {config.CHUNKS_PATH}")

    expected_hash = manifest["artifacts"]["chunks"]["sha256"]
    actual_hash = file_sha256(config.CHUNKS_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError("chunks.jsonl does not match its manifest. Rebuild the corpus.")

    chunks: list[DocumentChunk] = []
    with config.CHUNKS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in chunks.jsonl at line {line_number}."
                ) from error
            chunks.append(cast(DocumentChunk, chunk))

    _validate_chunks(chunks)
    if len(chunks) != manifest["chunk_count"]:
        raise RuntimeError(
            f"Chunk count mismatch: manifest={manifest['chunk_count']}, actual={len(chunks)}."
        )
    return manifest, chunks
