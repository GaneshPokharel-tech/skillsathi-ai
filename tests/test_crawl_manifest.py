"""Crawl-run manifest integration tests using a mocked browser."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.ingestion import crawler


class TestCrawlRunManifest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pages_dir = self.root / "pages"
        self.urls_file = self.root / "discovered_urls.json"

        self.urls = [
            "https://skillshikshya.com/courses/test-course-a",
            "https://skillshikshya.com/courses/test-course-b",
        ]

        self.urls_file.write_text(
            json.dumps(self.urls),
            encoding="utf-8",
        )

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def crawl(self, failures=None):
        failures = set(failures or [])

        class FakeBrowser:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def arun(self, *, url, config):
                return SimpleNamespace(
                    success=url not in failures,
                    url=url,
                    error_message="Simulated failure",
                )

        with (
            patch.object(crawler, "PAGES_DIR", self.pages_dir),
            patch.object(crawler, "URLS_FILE", self.urls_file),
            patch.object(
                crawler,
                "prepare_urls",
                return_value=self.urls,
            ),
            patch.object(crawler, "validate_public_url"),
            patch.object(crawler, "validate_dns_scope"),
            patch.object(crawler, "configure_browser_request_guard"),
            patch.object(crawler, "BrowserConfig"),
            patch.object(crawler, "_run_config"),
            patch.object(
                crawler,
                "_extract_markdown",
                return_value="# Course\n## Curriculum\nLearn SQL.",
            ),
            patch.object(
                crawler,
                "AsyncWebCrawler",
                return_value=FakeBrowser(),
            ),
            patch.object(crawler, "REQUEST_DELAY", 0),
        ):
            await crawler.crawl_pages()

        return json.loads(
            (self.root / "crawl_manifest.json").read_text(
                encoding="utf-8"
            )
        )

    async def test_successful_run_records_all_selected_pages(self):
        manifest = await self.crawl()

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["expected_urls"], self.urls)
        self.assertEqual(manifest["attempted"], 2)
        self.assertEqual(manifest["successful"], 2)
        self.assertEqual(manifest["failed"], 0)
        self.assertEqual(len(manifest["pages"]), 2)
        self.assertEqual(
            {p["status"] for p in manifest["pages"]},
            {"success"},
        )

        for url in self.urls:
            metadata = json.loads(
                (
                    self.pages_dir
                    / f"{crawler.page_id(url)}.json"
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(
                metadata["run_id"],
                manifest["run_id"],
            )

    async def test_failed_run_cannot_reuse_previous_page_metadata(self):
        first = await self.crawl()

        second = await self.crawl(
            failures={self.urls[1]},
        )

        self.assertNotEqual(
            first["run_id"],
            second["run_id"],
        )

        self.assertEqual(second["status"], "incomplete")
        self.assertEqual(second["successful"], 1)
        self.assertEqual(second["failed"], 1)

        statuses = {
            item["source_url"]: item["status"]
            for item in second["pages"]
        }

        self.assertEqual(
            statuses[self.urls[1]],
            "failed",
        )

        # An old file can still exist. Its run_id exposes that it
        # does not belong to the current crawl.
        old_metadata = json.loads(
            (
                self.pages_dir
                / f"{crawler.page_id(self.urls[1])}.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            old_metadata["run_id"],
            first["run_id"],
        )
        self.assertNotEqual(
            old_metadata["run_id"],
            second["run_id"],
        )


if __name__ == "__main__":
    unittest.main()
