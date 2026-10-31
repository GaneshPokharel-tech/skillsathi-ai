import os

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion import crawler
from app.ingestion.auto_sync import prepare_refresh_once


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class AutoSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        self.receipt = self.root / "activation.json"
        self.runs = self.root / "runs"
        self.lock = self.root / "sync.lock"

        self.old_url = (
            "https://skillshikshya.com/blogs/accounting-career-path"
        )
        self.about = "https://skillshikshya.com/about"
        self.new_url = (
            "https://skillshikshya.com/blogs/npl-sponsors-marketing-lessons"
        )

        self.original_receipt = {
            "snapshot_pointer": str(
                self.root / "old" / "active-generation.json"
            ),
            "active_collection": "existing_collection",
            "active_alias": "skillshikshya_active",
        }

        self.receipt.write_text(
            json.dumps(self.original_receipt),
            encoding="utf-8",
        )

    def test_discovery_preserves_old_pages_and_detects_new(self):
        old_verified = {
            "status": "complete",
            "validated_pages": 1,
            "validated_urls": [self.old_url],
        }
        new_verified = {
            "status": "complete",
            "validated_pages": 3,
            "validated_urls": [
                self.old_url, self.about, self.new_url
            ],
            "run_id": "verified-new-run",
        }

        async def fake_homepage(*, scope, lock_path):
            storage = crawler.site_storage_paths(
                scope, root=crawler.OUTPUT_DIR
            )
            storage.output_dir.mkdir(
                parents=True, exist_ok=True
            )
            storage.urls_file.write_text(
                json.dumps([self.about, self.new_url]),
                encoding="utf-8",
            )

        async def fake_pages(*, scope, lock_path):
            storage = crawler.site_storage_paths(
                scope, root=crawler.OUTPUT_DIR
            )
            inventory = json.loads(
                storage.urls_file.read_text(encoding="utf-8")
            )
            (storage.output_dir / "crawl_manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "successful": len(inventory),
                    "failed": 0,
                }),
                encoding="utf-8",
            )

        hashes = [
            {self.old_url: digest("old")},
            {
                self.old_url: digest("old"),
                self.about: digest("about"),
                self.new_url: digest("new"),
            },
        ]

        with patch(
            "app.ingestion.auto_sync.publication.resolve_active_generation",
            side_effect=[{"phase": "old"}, {"phase": "new"}],
        ), patch(
            "app.ingestion.auto_sync.sync.verify_published_crawl_run",
            side_effect=[old_verified, new_verified],
        ), patch(
            "app.ingestion.auto_sync._read_hashes",
            side_effect=hashes,
        ), patch(
            "app.ingestion.auto_sync.crawler.crawl_homepage",
            side_effect=fake_homepage,
        ), patch(
            "app.ingestion.auto_sync.crawler.crawl_pages",
            side_effect=fake_pages,
        ), patch(
            "app.ingestion.auto_sync.publication.process_and_publish_generation"
        ) as publish:
            result = prepare_refresh_once(
                receipt_path=self.receipt,
                runs_root=Path(os.path.relpath(self.runs)),
                lock_path=self.lock,
            )

        self.assertEqual(
            result["changes"]["added"],
            sorted([self.about, self.new_url]),
        )
        self.assertEqual(result["changes"]["removed"], [])
        self.assertEqual(result["verified_pages"], 3)
        self.assertEqual(result["approved_urls"], 3)
        self.assertEqual(publish.call_count, 1)

        arguments = publish.call_args.kwargs
        pointer = arguments["active_pointer_path"]
        lock = arguments["lock_path"]

        self.assertTrue(pointer.is_absolute())
        self.assertTrue(lock.is_absolute())
        self.assertEqual(lock, pointer.parent / "refresh.lock")

        inventory = json.loads(
            Path(result["inventory_path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(self.old_url, inventory)
        self.assertEqual(
            json.loads(self.receipt.read_text()),
            self.original_receipt,
        )

    def test_failed_crawl_preserves_existing_activation(self):
        old_verified = {
            "status": "complete",
            "validated_pages": 1,
            "validated_urls": [self.old_url],
        }

        async def fail_crawl(*, scope, lock_path):
            raise RuntimeError("Website temporarily unavailable")

        old_output = crawler.OUTPUT_DIR
        old_max = crawler.MAX_PAGES

        with patch(
            "app.ingestion.auto_sync.publication.resolve_active_generation",
            return_value={"phase": "old"},
        ), patch(
            "app.ingestion.auto_sync.sync.verify_published_crawl_run",
            return_value=old_verified,
        ), patch(
            "app.ingestion.auto_sync._read_hashes",
            return_value={self.old_url: digest("old")},
        ), patch(
            "app.ingestion.auto_sync.crawler.crawl_homepage",
            side_effect=fail_crawl,
        ):
            with self.assertRaises(RuntimeError):
                prepare_refresh_once(
                    receipt_path=self.receipt,
                    runs_root=self.runs,
                    lock_path=self.lock,
                )

        self.assertEqual(crawler.OUTPUT_DIR, old_output)
        self.assertEqual(crawler.MAX_PAGES, old_max)
        self.assertEqual(
            json.loads(self.receipt.read_text()),
            self.original_receipt,
        )


if __name__ == "__main__":
    unittest.main()
