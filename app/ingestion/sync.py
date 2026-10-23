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
    manifest_path=None,
    processed_pages_dir=None,
    report_path=None,
) -> dict:
    """Validate a complete staged snapshot before switching its alias.

    The caller must establish that expected_chunk_ids represents an
    approved, complete snapshot. This function never deletes collections.
    """
    verified = require_verified_crawl_run(
        manifest_path=manifest_path,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )

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

    verified_chunks = {}
    verified_urls = set(verified["validated_urls"])

    for source_url in sorted(verified_urls):
        chunks = _load_verified_page_chunks(
            verified=verified,
            source_url=source_url,
            processed_pages_dir=processed_pages_dir,
            report_path=report_path,
        )

        for chunk in chunks:
            identifier = chunk["chunk_id"]

            if identifier in verified_chunks:
                raise ValueError(
                    "Duplicate chunk ID in verified crawl."
                )

            verified_chunks[identifier] = {
                "source_url": chunk["source_url"],
                "document_title": chunk["document_title"],
                "section": chunk["section"],
                "text": chunk["text"],
            }

    actual_ids = set()
    seen_verified_ids = set()
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=staged_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            identifier = str(record.id)
            actual_ids.add(identifier)

            payload = record.payload

            if identifier in verified_chunks:
                if payload != verified_chunks[identifier]:
                    raise ValueError(
                        "Staged payload does not match verified Markdown."
                    )

                seen_verified_ids.add(identifier)

            elif (
                isinstance(payload, dict)
                and payload.get("source_url") in verified_urls
            ):
                raise ValueError(
                    "Unexpected chunk for a verified crawl page."
                )

        if offset is None:
            break

    if seen_verified_ids != set(verified_chunks):
        raise ValueError(
            "Staged collection is missing verified crawl chunks."
        )

    expected_ids = set(expected_chunk_ids)

    if actual_ids != expected_ids:
        raise ValueError(
            "Staged snapshot does not match approved chunk manifest."
        )

    def read_unverified_points(collection_name):
        """Read points belonging to pages outside the verified batch."""
        copied = {}
        offset = None

        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for record in records:
                payload = record.payload or {}

                if payload.get("source_url") in verified_urls:
                    continue

                identifier = str(record.id)
                copied[identifier] = {
                    "payload": payload,
                    "vector": record.vector,
                }

            if offset is None:
                break

        return copied

    original_copied = read_unverified_points(
        expected_active_collection
    )
    staged_copied = read_unverified_points(
        staged_collection
    )

    if original_copied != staged_copied:
        raise ValueError(
            "Copied course points differ from the active collection."
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



def build_staged_page_snapshot(
    client,
    *,
    source_collection: str,
    staged_collection: str,
    plan: dict,
) -> dict:
    """Build a separate snapshot without changing the active collection.

    The plan must come from a validated page refresh. This function
    handles one page; complete website refresh validation is separate.
    """
    import math
    from qdrant_client import models

    if (
        not isinstance(source_collection, str)
        or not source_collection.strip()
        or not isinstance(staged_collection, str)
        or not staged_collection.strip()
        or source_collection == staged_collection
    ):
        raise ValueError("Distinct collection names are required.")

    if not isinstance(plan, dict):
        raise ValueError("A validated refresh plan is required.")

    try:
        source_url = plan["source_url"]
        additions = plan["upsert_chunks"]
        vectors = plan["upsert_vectors"]
        removed = plan["delete_ids"]
        unchanged = plan["unchanged_ids"]
    except KeyError as exc:
        raise ValueError("Incomplete refresh plan.") from exc

    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("Invalid source URL.")

    if any(not isinstance(group, list) for group in (
        additions, vectors, removed, unchanged
    )):
        raise ValueError("Invalid refresh plan structure.")

    if len(additions) != len(vectors):
        raise ValueError("Missing staged embeddings.")

    if not additions and not unchanged:
        raise ValueError("Cannot replace a page with empty content.")

    old_ids = removed + unchanged

    if (
        any(not isinstance(item, str) or not item for item in old_ids)
        or len(old_ids) != len(set(old_ids))
    ):
        raise ValueError("Invalid existing chunk IDs.")

    if not client.collection_exists(source_collection):
        raise ValueError("Source collection does not exist.")

    if client.collection_exists(staged_collection):
        raise ValueError("Staged collection already exists.")

    configuration = client.get_collection(
        source_collection
    ).config.params.vectors

    if (
        isinstance(configuration, dict)
        or configuration.size < 1
        or configuration.distance != models.Distance.COSINE
    ):
        raise ValueError("Unsupported source vector configuration.")

    prepared = []
    new_ids = set()

    for chunk, vector in zip(additions, vectors):
        if (
            not isinstance(chunk, dict)
            or chunk.get("source_url") != source_url
            or not isinstance(chunk.get("text"), str)
            or not chunk["text"].strip()
        ):
            raise ValueError("Invalid updated chunk.")

        identifier = chunk.get("chunk_id")

        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in new_ids
        ):
            raise ValueError("Duplicate or invalid updated chunk ID.")

        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != configuration.size
        ):
            raise ValueError("Invalid staged vector dimension.")

        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid staged vector values.") from exc

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite staged vector.")

        new_ids.add(identifier)

        prepared.append(models.PointStruct(
            id=identifier,
            vector=values,
            payload={
                "source_url": source_url,
                "document_title": chunk["document_title"],
                "section": chunk["section"],
                "text": chunk["text"],
            },
        ))

    retained = []
    source_ids = set()
    all_old_ids = set()
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=source_collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        for record in records:
            identifier = str(record.id)
            all_old_ids.add(identifier)

            if (record.payload or {}).get("source_url") == source_url:
                source_ids.add(identifier)

            if identifier not in removed:
                retained.append(models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload or {},
                ))

        if offset is None:
            break

    if source_ids != set(old_ids):
        raise ValueError("Refresh plan does not match indexed page.")

    if new_ids & all_old_ids:
        raise ValueError("Updated IDs collide with existing points.")

    expected_ids = sorted(
        {str(point.id) for point in retained} | new_ids
    )

    client.create_collection(
        collection_name=staged_collection,
        vectors_config=models.VectorParams(
            size=configuration.size,
            distance=configuration.distance,
        ),
    )

    try:
        points = retained + prepared

        for position in range(0, len(points), 128):
            client.upsert(
                collection_name=staged_collection,
                points=points[position:position + 128],
                wait=True,
            )

        actual_ids = set()
        offset = None

        while True:
            records, offset = client.scroll(
                collection_name=staged_collection,
                limit=128,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            actual_ids.update(str(record.id) for record in records)

            if offset is None:
                break

        if actual_ids != set(expected_ids):
            raise ValueError("Staged snapshot verification failed.")

    except Exception:
        try:
            client.delete_collection(
                collection_name=staged_collection,
            )
        except Exception:
            pass
        raise

    return {
        "staged_collection": staged_collection,
        "expected_chunk_ids": expected_ids,
        "copied_chunks": len(retained),
        "updated_chunks": len(prepared),
        "removed_chunks": len(removed),
    }



def refresh_page_snapshot(
    client,
    *,
    source_collection: str,
    staged_collection: str,
    alias_name: str,
    source_url: str,
    fresh_chunks: list[dict],
    embed_texts,
    vector_size: int,
    manifest_path=None,
    processed_pages_dir=None,
    report_path=None,
) -> dict:
    """Stage, verify and activate a single successfully crawled page.

    Requires a previously initialized alias, validated complete page
    content, and exclusive database-write coordination. Never deletes
    the previous active collection.
    """
    verified = require_verified_crawl_run(
        manifest_path=manifest_path,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )

    if source_url not in verified["validated_urls"]:
        raise ValueError(
            "Refresh source URL is absent from verified crawl."
        )

    expected_chunks = _load_verified_page_chunks(
        verified=verified,
        source_url=source_url,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )

    if fresh_chunks != expected_chunks:
        raise ValueError(
            "Refresh chunks do not match verified crawl content."
        )

    if (
        not isinstance(alias_name, str)
        or not alias_name.strip()
    ):
        raise ValueError("An active alias name is required.")

    aliases = {
        alias.alias_name: alias.collection_name
        for alias in client.get_aliases().aliases
    }

    if aliases.get(alias_name) != source_collection:
        raise ValueError(
            "Active alias is missing or points to another collection."
        )

    plan = stage_page_refresh(
        client,
        collection_name=source_collection,
        source_url=source_url,
        fresh_chunks=fresh_chunks,
        embed_texts=embed_texts,
        vector_size=vector_size,
    )

    if not plan["upsert_chunks"] and not plan["delete_ids"]:
        return {
            "status": "unchanged",
            "active_collection": source_collection,
            "updated_chunks": 0,
            "removed_chunks": 0,
        }

    built = build_staged_page_snapshot(
        client,
        source_collection=source_collection,
        staged_collection=staged_collection,
        plan=plan,
    )

    activated = activate_collection_snapshot(
        client,
        alias_name=alias_name,
        expected_active_collection=source_collection,
        staged_collection=staged_collection,
        expected_chunk_ids=built["expected_chunk_ids"],
        manifest_path=manifest_path,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )

    return {
        "status": "activated",
        **activated,
        "copied_chunks": built["copied_chunks"],
        "updated_chunks": built["updated_chunks"],
        "removed_chunks": built["removed_chunks"],
    }



def validate_crawl_snapshot(
    *,
    run_id: str,
    expected_urls: list[str],
    crawled_pages: list[dict],
    processed_pages: list[dict],
) -> dict:
    """Validate coverage of a single approved crawl batch.

    Expected URLs are the selected crawl targets, not necessarily every
    URL discovered on the website.

    This function does not read files or modify Qdrant. Each record
    must belong to the same uniquely identified crawl run.
    """
    import re

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("A crawl run ID is required.")

    if not isinstance(expected_urls, list) or not expected_urls:
        raise ValueError("Expected crawl URLs cannot be empty.")

    if any(
        not isinstance(url, str) or not url.strip()
        for url in expected_urls
    ):
        raise ValueError("Invalid expected URL.")

    expected = set(expected_urls)

    if len(expected) != len(expected_urls):
        raise ValueError("Duplicate expected URLs.")

    def validate_records(records, *, stage, success_status):
        if not isinstance(records, list):
            raise ValueError(f"Invalid {stage} records.")

        seen = set()

        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid {stage} record.")

            url = record.get("source_url")

            if (
                not isinstance(url, str)
                or url not in expected
                or url in seen
            ):
                raise ValueError(
                    f"Unexpected or duplicate {stage} URL."
                )

            seen.add(url)

            if record.get("run_id") != run_id:
                raise ValueError(
                    f"{stage} record belongs to another run."
                )

            if record.get("status") != success_status:
                raise ValueError(
                    f"{stage} did not succeed for {url}."
                )

            if stage == "processing":
                digest = record.get("content_hash")
                characters = record.get("cleaned_character_count")

                if (
                    not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or isinstance(characters, bool)
                    or not isinstance(characters, int)
                    or characters <= 0
                ):
                    raise ValueError(
                        f"Invalid cleaned content for {url}."
                    )

        if seen != expected:
            raise ValueError(
                f"Incomplete {stage} coverage: "
                f"{len(seen)}/{len(expected)} pages."
            )

    validate_records(
        crawled_pages,
        stage="crawl",
        success_status="success",
    )

    validate_records(
        processed_pages,
        stage="processing",
        success_status="processed",
    )

    return {
        "status": "complete",
        "run_id": run_id,
        "validated_pages": len(expected),
        "validated_urls": sorted(expected),
    }



def process_crawl_run(
    *,
    manifest_path,
    raw_pages_dir,
    processed_pages_dir,
    report_path,
    scope=None,
) -> dict:
    """Process only verified pages belonging to one complete crawl run.

    All selected pages are validated before processing. Processed
    files are staged before publication. Qdrant is never modified.
    """
    import hashlib
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from app.ingestion.crawler import (
        normalize_content_url,
        page_id,
    )
    from app.ingestion.processor import _process_page

    manifest_path = Path(manifest_path)
    raw_pages_dir = Path(raw_pages_dir)
    processed_pages_dir = Path(processed_pages_dir)
    report_path = Path(report_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if not isinstance(manifest, dict):
        raise ValueError("Crawl manifest must be an object.")

    run_id = manifest.get("run_id")
    expected_urls = manifest.get("expected_urls")
    crawl_records = manifest.get("pages")

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Missing crawl run ID.")

    if (
        not isinstance(expected_urls, list)
        or not expected_urls
        or any(
            not isinstance(url, str) or not url.strip()
            for url in expected_urls
        )
        or len(set(expected_urls)) != len(expected_urls)
    ):
        raise ValueError("Invalid expected crawl URLs.")

    if (
        manifest.get("status") != "complete"
        or manifest.get("attempted") != len(expected_urls)
        or manifest.get("successful") != len(expected_urls)
        or manifest.get("failed") != 0
        or not isinstance(crawl_records, list)
        or len(crawl_records) != len(expected_urls)
    ):
        raise ValueError("Crawl run is not complete.")

    records_by_url = {}

    for record in crawl_records:
        if not isinstance(record, dict):
            raise ValueError("Invalid crawl record.")

        url = record.get("source_url")

        if (
            url not in expected_urls
            or url in records_by_url
            or record.get("run_id") != run_id
            or record.get("status") != "success"
            or record.get("final_url") != url
        ):
            raise ValueError("Invalid or stale crawl record.")

        records_by_url[url] = record

    if set(records_by_url) != set(expected_urls):
        raise ValueError("Crawl page coverage is incomplete.")

    selected_pages = []

    # Preflight every selected raw page before writing anything.
    for url in expected_urls:
        approved = normalize_content_url(url, scope=scope)

        if approved != url:
            raise ValueError("Unapproved or noncanonical page URL.")

        identifier = page_id(url)
        markdown_path = raw_pages_dir / f"{identifier}.md"
        metadata_path = raw_pages_dir / f"{identifier}.json"

        if not markdown_path.is_file() or not metadata_path.is_file():
            raise ValueError(
                f"Missing selected raw page or metadata: {url}"
            )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if (
            not isinstance(metadata, dict)
            or metadata.get("run_id") != run_id
            or metadata.get("source_url") != url
        ):
            raise ValueError("Raw page metadata belongs to another run.")

        raw = markdown_path.read_bytes()

        if not raw.strip():
            raise ValueError("Empty raw page content.")

        if (
            "content_hash" in metadata
            and metadata["content_hash"]
            != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("Raw page content hash mismatch.")

        selected_pages.append((url, markdown_path))

    with TemporaryDirectory(prefix="skillsathi-processing-") as directory:
        staging_dir = Path(directory)
        processed_records = []

        for url, markdown_path in selected_pages:
            result = _process_page(
                markdown_path,
                scope=scope,
                output_dir=staging_dir,
            )

            if result["status"] != "processed":
                raise ValueError(
                    f"Page processing failed for {url}."
                )

            identifier = page_id(url)

            cleaned_path = staging_dir / f"{identifier}.md"
            cleaned_metadata_path = staging_dir / f"{identifier}.json"

            cleaned = cleaned_path.read_bytes()

            cleaned_metadata = json.loads(
                cleaned_metadata_path.read_text(encoding="utf-8")
            )

            digest = hashlib.sha256(cleaned).hexdigest()

            if (
                cleaned_metadata.get("run_id") != run_id
                or cleaned_metadata.get("source_url") != url
                or cleaned_metadata.get("content_hash") != digest
                or not cleaned.strip()
            ):
                raise ValueError("Invalid processed page metadata.")

            processed_records.append({
                "run_id": run_id,
                "source_url": url,
                "status": "processed",
                "file": markdown_path.name,
                "content_hash": digest,
                "cleaned_character_count": len(
                    cleaned.decode("utf-8")
                ),
            })

        # Final validation before publishing processed files.
        validate_crawl_snapshot(
            run_id=run_id,
            expected_urls=expected_urls,
            crawled_pages=crawl_records,
            processed_pages=processed_records,
        )

        report = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "processed": len(processed_records),
            "skipped": 0,
            "failed": 0,
            "pages": processed_records,
        }

        processed_pages_dir.mkdir(parents=True, exist_ok=True)

        for url, _ in selected_pages:
            identifier = page_id(url)

            for extension in (".md", ".json"):
                filename = identifier + extension

                destination = processed_pages_dir / filename
                temporary = processed_pages_dir / (
                    filename + "." + run_id + ".tmp"
                )

                temporary.write_bytes(
                    (staging_dir / filename).read_bytes()
                )
                temporary.replace(destination)

        report_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_report = report_path.with_name(
            report_path.name + "." + run_id + ".tmp"
        )

        temporary_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        temporary_report.replace(report_path)

    return report



def verify_published_crawl_run(
    *,
    manifest_path,
    processed_pages_dir,
    report_path,
) -> dict:
    """Verify a published crawl run against its actual cleaned files.

    Read-only validation. This does not access Qdrant or activate an
    index snapshot. Call it again immediately before a future publish.
    """
    import hashlib
    import json
    from pathlib import Path

    from app.ingestion.crawler import normalize_content_url, page_id

    def read_json(path, description):
        if not path.is_file():
            raise ValueError(f"Missing {description}: {path}")

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid {description}: {path}") from exc

        if not isinstance(value, dict):
            raise ValueError(f"{description} must be a JSON object.")

        return value

    manifest = read_json(Path(manifest_path), "crawl manifest")
    report = read_json(Path(report_path), "cleaning report")
    directory = Path(processed_pages_dir)

    run_id = manifest.get("run_id")
    urls = manifest.get("expected_urls")
    crawled = manifest.get("pages")
    processed = report.get("pages")

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Missing crawl run ID.")

    if (
        not isinstance(urls, list)
        or not urls
        or any(not isinstance(url, str) or not url for url in urls)
        or len(set(urls)) != len(urls)
    ):
        raise ValueError("Invalid expected crawl URLs.")

    count = len(urls)

    if (
        manifest.get("status") != "complete"
        or manifest.get("attempted") != count
        or manifest.get("successful") != count
        or manifest.get("failed") != 0
        or report.get("run_id") != run_id
        or report.get("processed") != count
        or report.get("skipped") != 0
        or report.get("failed") != 0
    ):
        raise ValueError("Crawl or processing report is incomplete.")

    result = validate_crawl_snapshot(
        run_id=run_id,
        expected_urls=urls,
        crawled_pages=crawled,
        processed_pages=processed,
    )

    if any(
        record.get("final_url") != record["source_url"]
        for record in crawled
    ):
        raise ValueError("Unexpected crawl redirect.")

    records_by_url = {
        record["source_url"]: record
        for record in processed
    }

    for url in urls:
        if normalize_content_url(url) != url:
            raise ValueError(f"Unapproved or noncanonical URL: {url}")

        identifier = page_id(url)
        filename = identifier + ".md"

        markdown_path = directory / filename
        metadata_path = directory / (identifier + ".json")

        if not markdown_path.is_file():
            raise ValueError(f"Missing processed Markdown: {url}")

        metadata = read_json(
            metadata_path,
            "processed page metadata",
        )
        record = records_by_url[url]

        if (
            metadata.get("run_id") != run_id
            or metadata.get("source_url") != url
            or record.get("file") != filename
        ):
            raise ValueError(f"Stale or mismatched page metadata: {url}")

        try:
            markdown_bytes = markdown_path.read_bytes()
            markdown = markdown_bytes.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"Unreadable processed Markdown: {url}"
            ) from exc

        if not markdown.strip():
            raise ValueError(f"Empty processed Markdown: {url}")

        digest = hashlib.sha256(markdown_bytes).hexdigest()
        characters = len(markdown)

        if (
            metadata.get("content_hash") != digest
            or record.get("content_hash") != digest
            or metadata.get("cleaned_character_count") != characters
            or record.get("cleaned_character_count") != characters
        ):
            raise ValueError(
                f"Published content does not match its report: {url}"
            )

    return result



def require_verified_crawl_run(
    *,
    manifest_path=None,
    processed_pages_dir=None,
    report_path=None,
) -> dict:
    """Fail closed unless an actual published crawl run verifies."""
    if any(
        value is None
        for value in (
            manifest_path,
            processed_pages_dir,
            report_path,
        )
    ):
        raise ValueError(
            "Missing verified crawl proof for snapshot activation."
        )

    return verify_published_crawl_run(
        manifest_path=manifest_path,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )



def _load_verified_page_chunks(
    *,
    verified: dict,
    source_url: str,
    processed_pages_dir,
    report_path,
) -> list[dict]:
    """Recreate canonical chunks from a verified published page."""
    import hashlib
    import json
    from pathlib import Path

    from app.ingestion.crawler import page_id
    from app.rag.engine import chunk_markdown

    if source_url not in verified["validated_urls"]:
        raise ValueError("Page is absent from verified crawl.")

    identifier = page_id(source_url)
    directory = Path(processed_pages_dir)

    try:
        report = json.loads(
            Path(report_path).read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (directory / (identifier + ".json")).read_text(
                encoding="utf-8"
            )
        )
        content_bytes = (
            directory / (identifier + ".md")
        ).read_bytes()
        markdown = content_bytes.decode("utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Verified page files cannot be read."
        ) from exc

    if (
        not isinstance(report, dict)
        or not isinstance(metadata, dict)
        or report.get("run_id") != verified["run_id"]
        or not isinstance(report.get("pages"), list)
    ):
        raise ValueError("Verified run metadata changed.")

    matching_records = [
        record
        for record in report["pages"]
        if isinstance(record, dict)
        and record.get("source_url") == source_url
    ]

    if len(matching_records) != 1:
        raise ValueError("Verified page report is inconsistent.")

    record = matching_records[0]
    digest = hashlib.sha256(content_bytes).hexdigest()

    if (
        metadata.get("run_id") != verified["run_id"]
        or metadata.get("source_url") != source_url
        or metadata.get("content_hash") != digest
        or record.get("run_id") != verified["run_id"]
        or record.get("status") != "processed"
        or record.get("content_hash") != digest
        or record.get("cleaned_character_count") != len(markdown)
        or metadata.get("cleaned_character_count") != len(markdown)
        or not markdown.strip()
    ):
        raise ValueError(
            "Published page no longer matches verified crawl."
        )

    chunks = chunk_markdown(
        markdown,
        source_url=source_url,
    )

    if not chunks:
        raise ValueError("Verified page produced no indexable chunks.")

    return chunks
