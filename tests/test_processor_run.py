"""Tests for manifest-bound processing without stale crawl files."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.ingestion.crawler import page_id
from app.ingestion import sync


class TestManifestBoundProcessing(unittest.TestCase):

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        root = Path(self.temp.name)

        self.raw_dir = root / "raw"
        self.processed_dir = root / "processed"
        self.manifest_path = root / "crawl_manifest.json"
        self.report_path = root / "cleaning_report.json"

        self.raw_dir.mkdir()

        self.run_id = "crawl-run-2026-a"

        self.urls = [
            "https://skillshikshya.com/courses/course-a",
            "https://skillshikshya.com/courses/course-b",
        ]

        for url in self.urls:
            self.write_page(url, self.run_id)

        self.manifest = {
            "run_id": self.run_id,
            "expected_urls": self.urls,
            "status": "complete",
            "attempted": 2,
            "successful": 2,
            "failed": 0,
            "pages": [
                {
                    "run_id": self.run_id,
                    "source_url": url,
                    "final_url": url,
                    "status": "success",
                }
                for url in self.urls
            ],
        }

        self.save_manifest()

    def write_page(self, url, run_id):
        identifier = page_id(url)

        markdown = (
            "# Professional Training Course\n\n"
            "## Overview\n"
            "This practical training program covers industry "
            "skills and relevant professional techniques.\n\n"
            "## Curriculum\n"
            "Students develop practical abilities through "
            "guided exercises, projects, and structured learning.\n"
        )

        (self.raw_dir / f"{identifier}.md").write_text(
            markdown,
            encoding="utf-8",
        )

        (self.raw_dir / f"{identifier}.json").write_text(
            json.dumps({
                "source_url": url,
                "run_id": run_id,
                "crawled_at": "2026-10-21T10:00:00+00:00",
            }),
            encoding="utf-8",
        )

    def save_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    def process(self):
        return sync.process_crawl_run(
            manifest_path=self.manifest_path,
            raw_pages_dir=self.raw_dir,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
        )

    def test_processes_only_selected_manifest_pages(self):
        old_url = (
            "https://skillshikshya.com/courses/old-course"
        )

        self.write_page(old_url, "previous-run")

        report = self.process()

        self.assertEqual(report["run_id"], self.run_id)
        self.assertEqual(report["processed"], 2)
        self.assertEqual(report["failed"], 0)

        self.assertEqual(
            {p["source_url"] for p in report["pages"]},
            set(self.urls),
        )

        self.assertFalse(
            (
                self.processed_dir
                / f"{page_id(old_url)}.md"
            ).exists()
        )

        for page in report["pages"]:
            self.assertEqual(page["run_id"], self.run_id)
            self.assertEqual(page["status"], "processed")
            self.assertEqual(len(page["content_hash"]), 64)
            self.assertGreater(
                page["cleaned_character_count"],
                0,
            )

    def test_incomplete_crawl_is_rejected(self):
        self.manifest["status"] = "incomplete"
        self.manifest["failed"] = 1
        self.save_manifest()

        with self.assertRaises(ValueError):
            self.process()

        self.assertFalse(self.report_path.exists())

    def test_previous_run_metadata_is_rejected(self):
        self.write_page(
            self.urls[1],
            "previous-run",
        )

        with self.assertRaises(ValueError):
            self.process()

        self.assertFalse(self.report_path.exists())

    def test_missing_selected_raw_page_is_rejected(self):
        missing = self.raw_dir / (
            page_id(self.urls[1]) + ".md"
        )
        missing.unlink()

        with self.assertRaises(ValueError):
            self.process()

        self.assertFalse(self.report_path.exists())

    def test_report_matches_completeness_validator(self):
        report = self.process()

        result = sync.validate_crawl_snapshot(
            run_id=self.run_id,
            expected_urls=self.urls,
            crawled_pages=self.manifest["pages"],
            processed_pages=report["pages"],
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["validated_pages"], 2)


    def verify_published(self):
        return sync.verify_published_crawl_run(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
        )

    def test_verifier_accepts_complete_published_run(self):
        self.process()

        result = self.verify_published()

        self.assertEqual(result["run_id"], self.run_id)
        self.assertEqual(result["validated_pages"], 2)
        self.assertEqual(
            set(result["validated_urls"]),
            set(self.urls),
        )

    def test_verifier_rejects_modified_cleaned_markdown(self):
        self.process()

        path = self.processed_dir / (
            page_id(self.urls[0]) + ".md"
        )

        path.write_text(
            path.read_text(encoding="utf-8")
            + " Unauthorized content change.",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            self.verify_published()

    def test_verifier_rejects_stale_processed_metadata(self):
        self.process()

        path = self.processed_dir / (
            page_id(self.urls[0]) + ".json"
        )

        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["run_id"] = "previous-run"

        path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            self.verify_published()

    def test_verifier_rejects_incomplete_manifest(self):
        self.process()

        self.manifest["status"] = "incomplete"
        self.save_manifest()

        with self.assertRaises(ValueError):
            self.verify_published()

    def test_verifier_rejects_stale_cleaning_report(self):
        self.process()

        report = json.loads(
            self.report_path.read_text(encoding="utf-8")
        )

        report["run_id"] = "previous-run"

        self.report_path.write_text(
            json.dumps(report),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            self.verify_published()

    def test_verifier_rejects_missing_processed_page(self):
        self.process()

        path = self.processed_dir / (
            page_id(self.urls[1]) + ".md"
        )

        path.unlink()

        with self.assertRaises(ValueError):
            self.verify_published()


if __name__ == "__main__":
    unittest.main()
