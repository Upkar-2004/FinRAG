"""Expand retrieved table rows with neighboring pieces from the same table."""

from .ingest import DocumentChunk


def _group_table_chunks(
    corpus: list[DocumentChunk],
) -> dict[str, list[DocumentChunk]]:
    """Group canonical table chunks by table_id and sort them by row position."""
    groups: dict[str, list[DocumentChunk]] = {}

    for chunk in corpus:
        metadata = chunk["metadata"]

        if metadata["source"] != "table":
            continue

        table_id = metadata.get("table_id")
        row_start = metadata.get("row_start")

        if table_id is None or row_start is None:
            raise RuntimeError(f"Table chunk {chunk['chunk_id']} is missing table metadata.")

        groups.setdefault(table_id, []).append(chunk)

    for table_chunks in groups.values():
        table_chunks.sort(key=lambda chunk: chunk["metadata"]["row_start"])

    return groups


def _get_context_length(chunk: DocumentChunk) -> int:
    """Return and validate a table chunk's context boundary."""
    context_length = chunk["metadata"].get("context_length")

    if context_length is None:
        raise RuntimeError(f"Table chunk {chunk['chunk_id']} has no context_length.")

    if not 0 <= context_length <= len(chunk["text"]):
        raise RuntimeError(f"Table chunk {chunk['chunk_id']} has an invalid context_length.")

    return context_length


def _rows_only(chunk: DocumentChunk) -> str:
    """Return only the row-specific portion of a table chunk."""
    context_length = _get_context_length(chunk)
    return chunk["text"][context_length:].lstrip()


def expand_table_hits(
    hits: list[dict],
    corpus: list[DocumentChunk],
    neighbor_count: int | None = 1,
) -> list[dict]:
    """Expand table hits with neighboring pieces while preserving retrieval scores."""
    if neighbor_count is not None and neighbor_count < 0:
        raise ValueError("neighbor_count must be zero, positive, or None.")

    groups = _group_table_chunks(corpus)
    expanded_hits = []

    for hit in hits:
        expanded_hit = hit.copy()
        metadata = hit["metadata"]

        if metadata["source"] != "table":
            expanded_hits.append(expanded_hit)
            continue

        table_id = metadata.get("table_id")

        if table_id is None:
            raise RuntimeError(f"Retrieved table chunk {hit['chunk_id']} has no table_id.")

        sibling_chunks = groups.get(table_id)

        if not sibling_chunks:
            raise RuntimeError(f"No canonical table chunks were found for {table_id}.")

        selected_index = next(
            (
                index
                for index, chunk in enumerate(sibling_chunks)
                if chunk["chunk_id"] == hit["chunk_id"]
            ),
            None,
        )

        if selected_index is None:
            raise RuntimeError(f"Retrieved chunk {hit['chunk_id']} is not in the canonical corpus.")

        if neighbor_count is None:
            selected_chunks = sibling_chunks
        else:
            start = max(0, selected_index - neighbor_count)
            end = min(
                len(sibling_chunks),
                selected_index + neighbor_count + 1,
            )
            selected_chunks = sibling_chunks[start:end]

        selected_chunk = sibling_chunks[selected_index]
        context_length = _get_context_length(selected_chunk)
        context = selected_chunk["text"][:context_length].strip()

        row_sections = [_rows_only(chunk) for chunk in selected_chunks]

        expanded_hit["text"] = "\n".join(part for part in [context, *row_sections] if part)
        expanded_hits.append(expanded_hit)

    return expanded_hits
