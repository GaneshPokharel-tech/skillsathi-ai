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



ACTIVE_COLLECTION_ALIAS = "skillshikshya_active"


def activate_collection_snapshot(
    client,
    *,
    alias_name: str,
    expected_active_collection: str,
    staged_collection: str,
    expected_chunk_ids: list[str],
) -> dict:
    """Validate a complete staged snapshot before switching its alias.

    The caller must establish that expected_chunk_ids represents an
    approved, complete snapshot. This function never deletes collections.
    """
    from qdrant_client import models

    names = (
        alias_name,
        expected_active_collection,
        staged_collection,
    )

    if any(
        not isinstance(name, str) or not name.strip()
        for name in names
    ):
        raise ValueError("Collection and alias names are required.")

    if len(set(names)) != 3:
        raise ValueError("Alias and collection names must be distinct.")

    if (
        not isinstance(expected_chunk_ids, list)
        or not expected_chunk_ids
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in expected_chunk_ids
        )
        or len(set(expected_chunk_ids)) != len(expected_chunk_ids)
    ):
        raise ValueError("A unique, nonempty chunk manifest is required.")

    aliases = {
        item.alias_name: item.collection_name
        for item in client.get_aliases().aliases
    }

    if aliases.get(alias_name) != expected_active_collection:
        raise ValueError(
            "Active alias changed or was not initialized."
        )

    if (
        not client.collection_exists(expected_active_collection)
        or not client.collection_exists(staged_collection)
    ):
        raise ValueError("Required collection is missing.")

    old_config = client.get_collection(
        expected_active_collection
    ).config.params.vectors

    new_config = client.get_collection(
        staged_collection
    ).config.params.vectors

    if (
        isinstance(old_config, dict)
        or isinstance(new_config, dict)
        or old_config.size != new_config.size
        or old_config.distance != new_config.distance
    ):
        raise ValueError("Incompatible snapshot vector configuration.")

    actual_ids = set()
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=staged_collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        actual_ids.update(str(record.id) for record in records)

        if offset is None:
            break

    expected_ids = set(expected_chunk_ids)

    if actual_ids != expected_ids:
        raise ValueError(
            "Staged snapshot does not match approved chunk manifest."
        )

    client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(
                    alias_name=alias_name,
                )
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=staged_collection,
                    alias_name=alias_name,
                )
            ),
        ]
    )

    return {
        "alias_name": alias_name,
        "previous_collection": expected_active_collection,
        "active_collection": staged_collection,
        "verified_chunks": len(actual_ids),
    }
