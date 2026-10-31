
"""Prepare a verified SkillShikshya refresh without touching Qdrant.

The activation stage and scheduler are implemented separately.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ingestion import crawler, publication, sync
from app.ingestion.change_detection import (
    merge_discovered_urls,
    plan_content_changes,
)
from app.ingestion.locking import (
    DEFAULT_REFRESH_LOCK_PATH,
    refresh_lock,
)


def _read_hashes(site: Path, urls: list[str]) -> dict[str, str]:
    """Read page hashes from a verified crawl's original metadata."""
    hashes = {}

    for url in urls:
        metadata_path = (
            site / "pages" /
            (crawler.page_id(url) + ".json")
        )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if metadata.get("source_url") != url:
            raise ValueError("Metadata source URL mismatch.")

        digest = metadata.get("content_hash")

        if (
            not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Invalid source content hash.")

        hashes[url] = digest

    return hashes


def _verified_snapshot(pointer: Path) -> tuple[dict, dict]:
    proof = publication.resolve_active_generation(
        active_pointer_path=pointer
    )

    verified = sync.verify_published_crawl_run(**proof)

    urls = verified.get("validated_urls", [])

    if (
        verified.get("status") != "complete"
        or not isinstance(urls, list)
        or not urls
        or len(urls) != len(set(urls))
        or verified.get("validated_pages") != len(urls)
    ):
        raise ValueError("Incomplete verified website snapshot.")

    return proof, verified


def prepare_refresh_once(
    *,
    receipt_path: Path = Path("data/knowledge_activation.json"),
    runs_root: Path = Path("data/refresh_runs"),
    lock_path: Path = Path("data/auto-sync.lock"),
) -> dict:
    """Discover, crawl, verify and compare one complete website snapshot.

    Does not modify the Qdrant database or the current activation receipt.
    Must run in a worker thread, not inside an active asyncio event loop.
    """
    receipt_path = Path(receipt_path)
    runs_root = Path(runs_root)
    lock_path = Path(lock_path)

    with refresh_lock(lock_path):
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8")
        )

        previous_pointer = Path(
            receipt["snapshot_pointer"]
        )

        _, previous_verified = _verified_snapshot(
            previous_pointer
        )

        previous_urls = sorted(
            previous_verified["validated_urls"]
        )

        old_hashes = _read_hashes(
            previous_pointer.parent,
            previous_urls,
        )

        scope = crawler.CrawlScope(
            base_url=crawler.BASE_URL,
            allowed_hosts=frozenset(crawler.ALLOWED_HOSTS),
            content_paths=crawler.CONTENT_PATHS,
        )

        run_name = (
            datetime.now(timezone.utc).strftime(
                "%Y%m%d-%H%M%S"
            )
            + "-"
            + uuid4().hex[:8]
        )

        run_root = (runs_root / run_name).resolve()

        storage = crawler.site_storage_paths(
            scope,
            root=run_root,
        )

        if storage.output_dir.exists():
            raise FileExistsError("Refresh run already exists.")

        original_output = crawler.OUTPUT_DIR
        original_max = crawler.MAX_PAGES

        try:
            crawler.OUTPUT_DIR = run_root

            asyncio.run(
                crawler.crawl_homepage(
                    scope=scope,
                    lock_path=DEFAULT_REFRESH_LOCK_PATH,
                )
            )

            discovered = json.loads(
                storage.urls_file.read_text(encoding="utf-8")
            )

            if not isinstance(discovered, list):
                raise ValueError("Invalid discovered URL inventory.")

            merged = merge_discovered_urls(
                previous_urls,
                discovered,
            )

            approved = crawler.prepare_urls(
                merged,
                limit=len(merged),
                scope=scope,
            )

            if (
                not approved
                or len(approved) > 250
                or set(approved) != set(merged)
                or not set(previous_urls).issubset(
                    set(approved)
                )
            ):
                raise ValueError(
                    "Discovered URL inventory is incomplete "
                    "or exceeds the safe page limit."
                )

            storage.urls_file.write_text(
                json.dumps(
                    approved, indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )

            crawler.MAX_PAGES = len(approved)

            asyncio.run(
                crawler.crawl_pages(
                    scope=scope,
                    lock_path=DEFAULT_REFRESH_LOCK_PATH,
                )
            )

        finally:
            crawler.OUTPUT_DIR = original_output
            crawler.MAX_PAGES = original_max

        manifest = json.loads(
            (storage.output_dir / "crawl_manifest.json")
            .read_text(encoding="utf-8")
        )

        if (
            manifest.get("status") != "complete"
            or manifest.get("successful") != len(approved)
            or manifest.get("failed") != 0
        ):
            raise RuntimeError(
                "Website crawl incomplete. "
                "Previous knowledge remains active."
            )

        pointer = (
            storage.output_dir / "active-generation.json"
        )

        publication.process_and_publish_generation(
            manifest_path=(
                storage.output_dir / "crawl_manifest.json"
            ),
            raw_pages_dir=storage.pages_dir,
            active_pointer_path=pointer,
            lock_path=(
                storage.output_dir / "refresh.lock"
            ),
            scope=scope,
        )

        proof, verified = _verified_snapshot(pointer)

        if set(verified["validated_urls"]) != set(approved):
            raise RuntimeError(
                "Published generation does not cover "
                "all discovered and preserved URLs."
            )

        fresh_hashes = _read_hashes(
            storage.output_dir,
            verified["validated_urls"],
        )

        changes = plan_content_changes(
            old_hashes,
            fresh_hashes,
        )

        report_path = (
            storage.output_dir / "content_changes.json"
        )

        report_path.write_text(
            json.dumps(
                changes, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )

        return {
            "status": "prepared",
            "run_id": verified["run_id"],
            "verified_pages": verified["validated_pages"],
            "approved_urls": len(approved),
            "changes": changes,
            "snapshot_pointer": str(pointer.resolve()),
            "inventory_path": str(
                storage.urls_file.resolve()
            ),
            "change_report": str(
                report_path.resolve()
            ),
            "previous_collection": receipt["active_collection"],
            "proof": proof,
        }
