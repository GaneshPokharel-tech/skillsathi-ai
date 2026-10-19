"""SkillSathi AI: document chunking and RAG utilities."""

import re
import uuid


def _split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split long text with overlap, preferring word boundaries."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            lower_bound = start + max_chars // 2

            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(
                    separator, lower_bound, end + 1
                )
                if boundary >= 0:
                    end = boundary + len(separator)
                    break

        part = text[start:end].strip()

        if part:
            chunks.append(part)

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap_chars)

        # Avoid beginning the next chunk halfway through a word.
        if (
            next_start > 0
            and not text[next_start - 1].isspace()
        ):
            while (
                next_start < end
                and not text[next_start].isspace()
            ):
                next_start += 1

            while (
                next_start < end
                and text[next_start].isspace()
            ):
                next_start += 1

        start = min(next_start, end)

    return chunks


def chunk_markdown(
    markdown: str,
    *,
    source_url: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[dict]:
    """Create source-grounded chunks using Markdown headings.

    Each chunk contains a stable UUID, source URL, section,
    document title, and text suitable for embedding.
    """
    if not isinstance(markdown, str):
        raise TypeError("Markdown must be a string.")

    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A source URL is required.")

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80.")

    if not 0 <= overlap_chars < max_chars:
        raise ValueError(
            "overlap_chars must be nonnegative and smaller than max_chars."
        )

    if not markdown.strip():
        return []

    document_title = ""
    current_section = "Overview"
    current_lines = []
    sections = []

    inside_code_fence = False

    def save_section():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_section, body))

    for line in markdown.splitlines():
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            inside_code_fence = not inside_code_fence

        heading = None

        if not inside_code_fence:
            heading = re.match(
                r"^\s{0,3}(#{1,6})\s+(.+?)\s*$",
                line,
            )

        if heading:
            save_section()
            current_lines = []

            title = heading.group(2).strip().rstrip("#").strip()
            current_section = title or "Overview"

            if len(heading.group(1)) == 1 and not document_title:
                document_title = current_section
        else:
            current_lines.append(line)

    save_section()

    chunks = []

    for section_index, (section, body) in enumerate(sections):
        headings = []

        if document_title:
            headings.append(document_title)

        if section != document_title:
            headings.append(section)

        prefix = "\n".join(headings) + "\n\n"

        # Keep most of the chunk budget for actual content.
        if len(prefix) > max_chars // 3:
            prefix = ""

        body_limit = max_chars - len(prefix)
        overlap = min(overlap_chars, body_limit - 1)

        parts = _split_text(body, body_limit, overlap)

        for part_index, part in enumerate(parts):
            text = (prefix + part).strip()

            identifier = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"{source_url}|{section_index}|"
                        f"{part_index}|{text}"
                    ),
                )
            )

            chunks.append(
                {
                    "chunk_id": identifier,
                    "source_url": source_url,
                    "document_title": document_title,
                    "section": section,
                    "text": text,
                }
            )

    return chunks


def index_chunks(
    client,
    *,
    collection_name: str,
    chunks: list[dict],
    vectors: list[list[float]],
) -> int:
    """Store source-grounded chunks in Qdrant using stable point IDs."""
    import math
    from qdrant_client import models

    if len(chunks) != len(vectors):
        raise ValueError("Chunks and vectors must have equal lengths.")

    if not chunks:
        return 0

    dimension = len(vectors[0])

    if dimension < 1:
        raise ValueError("Embedding vectors cannot be empty.")

    for vector in vectors:
        if (
            len(vector) != dimension
            or not all(math.isfinite(float(value)) for value in vector)
        ):
            raise ValueError("Invalid embedding vector.")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        collection = client.get_collection(collection_name)
        config = collection.config.params.vectors

        if (
            isinstance(config, dict)
            or config.size != dimension
            or config.distance != models.Distance.COSINE
        ):
            raise ValueError(
                "Existing collection has incompatible vector settings."
            )

    points = []

    for chunk, vector in zip(chunks, vectors):
        payload = {
            "source_url": chunk["source_url"],
            "document_title": chunk["document_title"],
            "section": chunk["section"],
            "text": chunk["text"],
        }

        points.append(
            models.PointStruct(
                id=chunk["chunk_id"],
                vector=[float(value) for value in vector],
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

    return len(points)


def search_chunks(
    client,
    *,
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[dict]:
    """Retrieve similar chunks with their source metadata."""
    import math

    if limit < 1:
        raise ValueError("Search limit must be positive.")

    if (
        not query_vector
        or not all(math.isfinite(float(v)) for v in query_vector)
    ):
        raise ValueError("Invalid query embedding.")

    response = client.query_points(
        collection_name=collection_name,
        query=[float(value) for value in query_vector],
        limit=limit,
        with_payload=True,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            {
                **payload,
                "chunk_id": str(point.id),
                "score": float(point.score),
            }
        )

    return results
