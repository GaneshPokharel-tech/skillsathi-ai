"""Plan safe page-level changes before mutating the vector database."""


def plan_page_refresh(
    *,
    source_url: str,
    existing_chunk_ids: list[str],
    fresh_chunks: list[dict],
) -> dict:
    """Compare one successfully processed page against its indexed chunks.

    This function never modifies Qdrant. The caller must validate that
    the fresh page was crawled and processed successfully.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A source URL is required.")

    if not isinstance(existing_chunk_ids, list):
        raise TypeError("Existing chunk IDs must be a list.")

    if not isinstance(fresh_chunks, list) or not fresh_chunks:
        raise ValueError(
            "Refusing refresh without nonempty validated page chunks."
        )

    for chunk_id in existing_chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Invalid existing chunk ID.")

    existing = set(existing_chunk_ids)

    if len(existing) != len(existing_chunk_ids):
        raise ValueError("Duplicate existing chunk IDs.")

    fresh_ids = set()

    for chunk in fresh_chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Fresh chunks must be dictionaries.")

        if chunk.get("source_url") != source_url:
            raise ValueError("Fresh chunk belongs to another source URL.")

        chunk_id = chunk.get("chunk_id")
        content = chunk.get("text")

        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError("Invalid fresh chunk.")

        if chunk_id in fresh_ids:
            raise ValueError("Duplicate fresh chunk ID.")

        fresh_ids.add(chunk_id)

    return {
        "source_url": source_url,
        "upsert_chunks": [
            chunk
            for chunk in fresh_chunks
            if chunk["chunk_id"] not in existing
        ],
        "delete_ids": sorted(existing - fresh_ids),
        "unchanged_ids": sorted(existing & fresh_ids),
    }



def get_indexed_page_chunk_ids(
    client,
    *,
    collection_name: str,
    source_url: str,
    batch_size: int = 128,
) -> list[str]:
    """Read indexed chunk IDs for exactly one source page.

    No Qdrant writes are performed.
    """
    from qdrant_client import models

    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("A collection name is required.")

    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A source URL is required.")

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > 1000
    ):
        raise ValueError("batch_size must be between 1 and 1000.")

    if not client.collection_exists(collection_name):
        return []

    source_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source_url",
                match=models.MatchValue(value=source_url),
            )
        ]
    )

    chunk_ids = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=source_filter,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        chunk_ids.extend(str(record.id) for record in records)

        if offset is None:
            break

    return sorted(chunk_ids)



def stage_page_refresh(
    client,
    *,
    collection_name: str,
    source_url: str,
    fresh_chunks: list[dict],
    embed_texts,
    vector_size: int,
) -> dict:
    """Prepare and validate one page refresh without writing to Qdrant.

    The embedding callback must only generate vectors. This function
    performs no database mutation. Applying changes is a separate step.
    """
    import math
    from qdrant_client import models

    if (
        isinstance(vector_size, bool)
        or not isinstance(vector_size, int)
        or not 1 <= vector_size <= 4096
    ):
        raise ValueError("Invalid expected vector dimension.")

    if not callable(embed_texts):
        raise TypeError("embed_texts must be callable.")

    if client.collection_exists(collection_name):
        collection = client.get_collection(collection_name)
        config = collection.config.params.vectors

        if (
            isinstance(config, dict)
            or config.size != vector_size
            or config.distance != models.Distance.COSINE
        ):
            raise ValueError(
                "Collection vector settings are incompatible."
            )

    existing_ids = get_indexed_page_chunk_ids(
        client,
        collection_name=collection_name,
        source_url=source_url,
    )

    plan = plan_page_refresh(
        source_url=source_url,
        existing_chunk_ids=existing_ids,
        fresh_chunks=fresh_chunks,
    )

    changes = plan["upsert_chunks"]

    if not changes:
        return {
            **plan,
            "upsert_vectors": [],
        }

    texts = [chunk["text"] for chunk in changes]
    vectors = embed_texts(texts)

    if not isinstance(vectors, list):
        raise ValueError("Embedding result must be a list.")

    if len(vectors) != len(changes):
        raise ValueError("Embedding count does not match chunk count.")

    validated = []

    for vector in vectors:
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Invalid embedding vector structure.")

        if len(vector) != vector_size:
            raise ValueError("Invalid embedding vector dimension.")

        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Non-numeric embedding vector.") from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite embedding vector.")

        validated.append(values)

    return {
        **plan,
        "upsert_vectors": validated,
    }
