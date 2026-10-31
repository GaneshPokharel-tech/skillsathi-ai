
"""Verified, incremental Qdrant activation.

Call under the API's shared chat lock when using embedded Qdrant.
A completed refresh preparation is required.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qdrant_client import models

from app.ingestion import sync
from app.ingestion.auto_sync import _verified_snapshot
from app.rag.engine import index_chunks


def _target(db, alias):
    return {
        item.alias_name: item.collection_name
        for item in db.get_aliases().aliases
    }.get(alias)


def _payload(chunk):
    return {
        "source_url": chunk["source_url"],
        "document_title": chunk["document_title"],
        "section": chunk["section"],
        "text": chunk["text"],
    }


def _switch(db, alias, current, destination):
    db.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(
                    alias_name=alias
                )
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=destination,
                    alias_name=alias,
                )
            ),
        ]
    )


def activate_prepared_refresh(
    prepared: dict,
    *,
    db,
    receipt_path: Path = Path("data/knowledge_activation.json"),
    embedder=None,
) -> dict:
    """Reuse identical chunks, verify all vectors, then switch alias.

    db must be an exclusively owned or serialized Qdrant client.
    No automatic removal of previously verified URLs is permitted.
    """
    receipt_path = Path(receipt_path)
    baseline = receipt_path.read_bytes()
    receipt = json.loads(baseline)

    alias = receipt["active_alias"]
    old_collection = receipt["active_collection"]

    if (
        alias != "skillshikshya_active"
        or prepared.get("previous_collection") != old_collection
        or _target(db, alias) != old_collection
    ):
        raise ValueError("Activation baseline changed.")

    if not db.collection_exists(old_collection):
        raise ValueError("Previous collection is missing.")

    _, old_snapshot = _verified_snapshot(
        Path(receipt["snapshot_pointer"])
    )
    new_proof, new_snapshot = _verified_snapshot(
        Path(prepared["snapshot_pointer"])
    )

    old_urls = set(old_snapshot["validated_urls"])
    new_urls = set(new_snapshot["validated_urls"])

    if (
        prepared["run_id"] != new_snapshot["run_id"]
        or prepared["verified_pages"] != len(new_urls)
        or receipt["verified_pages"] != len(old_urls)
        or not old_urls.issubset(new_urls)
    ):
        raise ValueError("Snapshot mismatch or unverified deletion.")

    old_points = {}
    offset = None

    while True:
        records, offset = db.scroll(
            collection_name=old_collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in records:
            identifier = str(point.id)
            if identifier in old_points:
                raise ValueError("Duplicate existing vector ID.")
            old_points[identifier] = point

            if (point.payload or {}).get("source_url") not in old_urls:
                raise ValueError("Old vector source is unverified.")

        if offset is None:
            break

    if (
        len(old_points) != receipt["active_vectors"]
        or db.count(
            collection_name=old_collection,
            exact=True,
        ).count != len(old_points)
    ):
        raise ValueError("Previous vector baseline mismatch.")

    chunks = []
    expected = {}

    for url in sorted(new_urls):
        page = sync._load_verified_page_chunks(
            verified=new_snapshot,
            source_url=url,
            processed_pages_dir=new_proof["processed_pages_dir"],
            report_path=new_proof["report_path"],
        )

        if not page:
            raise ValueError("Verified page has no chunks.")

        for chunk in page:
            identifier = str(chunk["chunk_id"])
            if identifier in expected:
                raise ValueError("Duplicate fresh chunk ID.")
            expected[identifier] = _payload(chunk)
            chunks.append(chunk)

    reusable = []
    changed = []

    for chunk in chunks:
        previous = old_points.get(str(chunk["chunk_id"]))
        if (
            previous is not None
            and previous.payload == _payload(chunk)
        ):
            vector = previous.vector
            if (
                not isinstance(vector, list)
                or len(vector) != 384
                or not all(
                    math.isfinite(float(value))
                    for value in vector
                )
            ):
                raise ValueError("Invalid existing embedding.")
            reusable.append((chunk, vector))
        else:
            changed.append(chunk)

    config = db.get_collection(
        old_collection
    ).config.params.vectors

    if (
        isinstance(config, dict)
        or config.size != 384
        or config.distance != models.Distance.COSINE
    ):
        raise ValueError("Unexpected embedding configuration.")

    staged = (
        "skillshikshya_gen_"
        + new_snapshot["run_id"][:12]
        + "_"
        + uuid4().hex[:8]
    )

    temporary_receipt = receipt_path.with_name(
        "." + receipt_path.name + "." + uuid4().hex + ".tmp"
    )

    switched = False

    try:
        db.create_collection(
            collection_name=staged,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

        for start in range(0, len(reusable), 64):
            batch = reusable[start:start + 64]
            index_chunks(
                db,
                collection_name=staged,
                chunks=[item[0] for item in batch],
                vectors=[item[1] for item in batch],
            )

        if changed:
            if embedder is None:
                from fastembed import TextEmbedding
                embedder = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5"
                )

            for start in range(0, len(changed), 48):
                batch = changed[start:start + 48]
                vectors = [
                    vector.tolist()
                    for vector in embedder.embed(
                        [item["text"] for item in batch],
                        batch_size=32,
                    )
                ]
                index_chunks(
                    db,
                    collection_name=staged,
                    chunks=batch,
                    vectors=vectors,
                )

        found = set()
        offset = None

        while True:
            records, offset = db.scroll(
                collection_name=staged,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for point in records:
                identifier = str(point.id)
                vector = point.vector

                if (
                    identifier in found
                    or identifier not in expected
                    or point.payload != expected[identifier]
                    or not isinstance(vector, list)
                    or len(vector) != 384
                    or not all(
                        math.isfinite(float(value))
                        for value in vector
                    )
                ):
                    raise ValueError(
                        "Staged point verification failed."
                    )

                found.add(identifier)

            if offset is None:
                break

        if (
            found != set(expected)
            or db.count(
                collection_name=staged,
                exact=True,
            ).count != len(expected)
        ):
            raise ValueError("Staged collection incomplete.")

        # Recheck source, receipt and alias immediately before switch.
        _, final_snapshot = _verified_snapshot(
            Path(prepared["snapshot_pointer"])
        )

        if (
            final_snapshot != new_snapshot
            or receipt_path.read_bytes() != baseline
            or _target(db, alias) != old_collection
        ):
            raise ValueError(
                "Activation baseline changed during indexing."
            )

        updated_receipt = {
            **receipt,
            "active_collection": staged,
            "previous_collection": old_collection,
            "snapshot_pointer": str(
                Path(prepared["snapshot_pointer"]).resolve()
            ),
            "run_id": new_snapshot["run_id"],
            "verified_pages": len(new_urls),
            "active_vectors": len(expected),
            "activated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        temporary_receipt.write_text(
            json.dumps(updated_receipt, indent=2),
            encoding="utf-8",
        )

        _switch(db, alias, old_collection, staged)
        switched = True

        if _target(db, alias) != staged:
            raise RuntimeError("Alias switch verification failed.")

        os.replace(temporary_receipt, receipt_path)

        return {
            "status": "activated",
            "active_collection": staged,
            "verified_pages": len(new_urls),
            "active_vectors": len(expected),
            "reused_vectors": len(reusable),
            "new_embeddings": len(changed),
            "previous_collection": old_collection,
        }

    except Exception:
        # Roll back a switched alias if receipt publication fails.
        if switched and _target(db, alias) == staged:
            _switch(db, alias, staged, old_collection)

        # Never delete a collection still used by the active alias.
        if (
            _target(db, alias) != staged
            and db.collection_exists(staged)
        ):
            db.delete_collection(staged)
        raise

    finally:
        temporary_receipt.unlink(missing_ok=True)
