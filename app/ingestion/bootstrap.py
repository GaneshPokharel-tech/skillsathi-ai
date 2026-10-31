"""Safe first-time SkillShikshya knowledge initialization."""

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ingestion import crawler, publication, sync
from app.rag.engine import QDRANT_PATH, index_chunks


ALIAS = "skillshikshya_active"
MODEL = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


def _preflight(receipt_path: Path, db_path: Path) -> None:
    """Never overwrite an existing initialization."""
    receipt_path = Path(receipt_path)
    db_path = Path(db_path)

    if receipt_path.is_symlink() or receipt_path.exists():
        raise RuntimeError(
            "Knowledge already initialized. Bootstrap refused."
        )

    if db_path.is_symlink():
        raise RuntimeError("Qdrant path is a symlink.")

    if db_path.exists():
        if not db_path.is_dir() or any(db_path.iterdir()):
            raise RuntimeError(
                "Existing Qdrant data detected. Bootstrap refused."
            )


def _payload(chunk: dict) -> dict:
    return {
        "source_url": chunk["source_url"],
        "document_title": chunk["document_title"],
        "section": chunk["section"],
        "text": chunk["text"],
    }


def bootstrap_verified_generation(
    *,
    db,
    pointer: Path,
    receipt_path: Path,
    embedder=None,
) -> dict:
    """Initialize an empty Qdrant DB from a verified crawl generation."""
    from qdrant_client import models

    pointer = Path(pointer).resolve()
    receipt_path = Path(receipt_path)

    if receipt_path.is_symlink() or receipt_path.exists():
        raise RuntimeError("Activation receipt already exists.")

    if db.get_collections().collections or db.get_aliases().aliases:
        raise RuntimeError(
            "Qdrant is not empty. Initial activation refused."
        )

    proof = publication.resolve_active_generation(
        active_pointer_path=pointer
    )
    verified = sync.verify_published_crawl_run(**proof)

    urls = verified.get("validated_urls")
    if (
        verified.get("status") != "complete"
        or not isinstance(urls, list)
        or not urls
        or len(urls) != len(set(urls))
        or verified.get("validated_pages") != len(urls)
    ):
        raise ValueError("Incomplete verified bootstrap snapshot.")

    chunks = []
    expected = {}

    for url in sorted(urls):
        page = sync._load_verified_page_chunks(
            verified=verified,
            source_url=url,
            processed_pages_dir=proof["processed_pages_dir"],
            report_path=proof["report_path"],
        )

        if not page:
            raise ValueError("Verified page contains no chunks.")

        for chunk in page:
            identifier = str(chunk["chunk_id"])
            if identifier in expected:
                raise ValueError("Duplicate bootstrap chunk ID.")

            expected[identifier] = _payload(chunk)
            chunks.append(chunk)

    if embedder is None:
        from fastembed import TextEmbedding
        embedder = TextEmbedding(model_name=MODEL)

    vectors = []

    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        produced = list(
            embedder.embed([chunk["text"] for chunk in batch])
        )

        if len(produced) != len(batch):
            raise ValueError("Embedding count mismatch.")

        for vector in produced:
            values = [float(value) for value in vector]
            if (
                len(values) != DIMENSION
                or not all(math.isfinite(x) for x in values)
            ):
                raise ValueError("Invalid bootstrap embedding.")

            vectors.append(values)

    staged = (
        "skillshikshya_gen_"
        + verified["run_id"][:12]
        + "_"
        + uuid4().hex[:8]
    )

    temporary_receipt = receipt_path.with_name(
        "." + receipt_path.name + "." + uuid4().hex + ".tmp"
    )

    alias_created = False

    try:
        indexed = index_chunks(
            db,
            collection_name=staged,
            chunks=chunks,
            vectors=vectors,
        )

        if indexed != len(chunks):
            raise ValueError("Incomplete bootstrap indexing.")

        if db.count(
            collection_name=staged,
            exact=True,
        ).count != len(expected):
            raise ValueError("Bootstrap vector count mismatch.")

        observed = {}
        offset = None

        while True:
            points, offset = db.scroll(
                collection_name=staged,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                identifier = str(point.id)
                if identifier in observed:
                    raise ValueError("Duplicate indexed point.")
                observed[identifier] = point.payload or {}

            if offset is None:
                break

        if observed != expected:
            raise ValueError(
                "Bootstrap indexed payload verification failed."
            )

        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError("Activation receipt changed.")

        receipt = {
            "active_alias": ALIAS,
            "active_collection": staged,
            "previous_collection": None,
            "snapshot_pointer": str(pointer),
            "run_id": verified["run_id"],
            "verified_pages": len(urls),
            "active_vectors": len(expected),
            "activated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        receipt_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        temporary_receipt.write_text(
            json.dumps(receipt, indent=2),
            encoding="utf-8",
        )

        db.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=staged,
                        alias_name=ALIAS,
                    )
                )
            ]
        )
        alias_created = True

        aliases = {
            item.alias_name: item.collection_name
            for item in db.get_aliases().aliases
        }

        if aliases.get(ALIAS) != staged:
            raise RuntimeError("Bootstrap alias verification failed.")

        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError("Activation receipt changed.")

        os.replace(temporary_receipt, receipt_path)

        return {
            "status": "activated",
            "verified_pages": len(urls),
            "active_vectors": len(expected),
            "active_collection": staged,
        }

    except Exception:
        if alias_created:
            db.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(
                            alias_name=ALIAS
                        )
                    )
                ]
            )

        if db.collection_exists(staged):
            db.delete_collection(staged)

        raise

    finally:
        temporary_receipt.unlink(missing_ok=True)


def run_initial_bootstrap(
    *,
    receipt_path: Path = Path("data/knowledge_activation.json"),
    db_path: Path = QDRANT_PATH,
    runs_root: Path = Path("data/bootstrap_runs"),
) -> dict:
    """Crawl and initialize a fresh local knowledge database once."""
    from qdrant_client import QdrantClient
    from app.ingestion.locking import (
        DEFAULT_REFRESH_LOCK_PATH,
        refresh_lock,
    )

    receipt_path = Path(receipt_path)
    db_path = Path(db_path)
    runs_root = Path(runs_root)

    _preflight(receipt_path, db_path)
    runs_root.mkdir(parents=True, exist_ok=True)

    with refresh_lock(Path("data/auto-sync-run.lock")):
        _preflight(receipt_path, db_path)

        scope = crawler.build_crawl_scope(crawler.BASE_URL)
        run_root = (
            runs_root / ("initial-" + uuid4().hex[:12])
        ).resolve()

        run_root.mkdir(parents=True, exist_ok=False)

        original_output = crawler.OUTPUT_DIR
        original_max = crawler.MAX_PAGES

        try:
            crawler.OUTPUT_DIR = run_root
            crawler.MAX_PAGES = max(
                int(original_max), 250
            )

            async def crawl():
                await crawler.crawl_homepage(
                    scope=scope,
                    lock_path=DEFAULT_REFRESH_LOCK_PATH,
                )
                await crawler.crawl_pages(
                    scope=scope,
                    lock_path=DEFAULT_REFRESH_LOCK_PATH,
                )

            asyncio.run(crawl())

        finally:
            crawler.OUTPUT_DIR = original_output
            crawler.MAX_PAGES = original_max

        storage = crawler.site_storage_paths(
            scope, root=run_root
        )
        site = storage.output_dir.resolve()
        pointer = site / "active-generation.json"

        publication.process_and_publish_generation(
            manifest_path=site / "crawl_manifest.json",
            raw_pages_dir=storage.pages_dir,
            active_pointer_path=pointer,
            lock_path=site / "refresh.lock",
            scope=scope,
        )

        _preflight(receipt_path, db_path)

        db = QdrantClient(path=str(db_path))
        try:
            return bootstrap_verified_generation(
                db=db,
                pointer=pointer,
                receipt_path=receipt_path,
            )
        finally:
            db.close()


def main() -> None:
    result = run_initial_bootstrap()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
