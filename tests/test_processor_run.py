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
            lock_path=self.manifest_path.parent / "refresh.lock",
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


    def test_processing_rejects_busy_refresh_lock(self):
        from app.ingestion.locking import refresh_lock

        lock_path = self.manifest_path.parent / "refresh.lock"

        with refresh_lock(lock_path):
            with self.assertRaisesRegex(
                RuntimeError,
                "(?i)lock|refresh|busy",
            ):
                sync.process_crawl_run(
                    manifest_path=self.manifest_path,
                    raw_pages_dir=self.raw_dir,
                    processed_pages_dir=self.processed_dir,
                    report_path=self.report_path,
                    lock_path=lock_path,
                )

        self.assertFalse(self.report_path.exists())
        self.assertFalse(self.processed_dir.exists())

    def test_processing_requires_shared_lock_path(self):
        with self.assertRaisesRegex(
            ValueError,
            "(?i)lock",
        ):
            sync.process_crawl_run(
                manifest_path=self.manifest_path,
                raw_pages_dir=self.raw_dir,
                processed_pages_dir=self.processed_dir,
                report_path=self.report_path,
            )

        self.assertFalse(self.report_path.exists())



    def test_legacy_blocks_simulated_partial_publication(self):
        """Legacy processing rejects overwrites before partial publication."""
        import hashlib
        import json
        from unittest.mock import patch

        from app.ingestion.crawler import page_id

        # First, publish a fully verified two-course snapshot.
        self.process()

        proof = {
            "manifest_path": self.manifest_path,
            "processed_pages_dir": self.processed_dir,
            "report_path": self.report_path,
        }

        verified_before = sync.verify_published_crawl_run(**proof)

        def published_files():
            return {
                item.name: item.read_bytes()
                for item in self.processed_dir.iterdir()
                if item.is_file()
                and item.suffix in (".md", ".json")
            }

        files_before = published_files()
        report_before = self.report_path.read_bytes()

        # Simulate updated website content for the first course.
        first_id = page_id(self.urls[0])
        second_id = page_id(self.urls[1])

        raw_markdown = self.raw_dir / f"{first_id}.md"
        raw_metadata = self.raw_dir / f"{first_id}.json"

        fresh_bytes = (
            b"# Updated Course A\\n\\n"
            b"Learn Python APIs, automated testing, "
            b"and practical deployment.\\n"
        )

        raw_markdown.write_bytes(fresh_bytes)

        metadata = json.loads(
            raw_metadata.read_text(encoding="utf-8")
        )
        metadata["content_hash"] = (
            hashlib.sha256(fresh_bytes).hexdigest()
        )
        raw_metadata.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        first_destination = (
            self.processed_dir / f"{first_id}.md"
        )
        second_destination = (
            self.processed_dir / f"{second_id}.md"
        )

        original_replace = Path.replace
        first_published = False

        def interrupt_after_first_page(current, destination):
            nonlocal first_published

            destination = Path(destination)

            if destination == second_destination:
                if not first_published:
                    raise AssertionError(
                        "Failure occurred before first publication."
                    )
                raise OSError(
                    "simulated publication interruption"
                )

            result = original_replace(current, destination)

            if destination == first_destination:
                first_published = True

            return result

        # Controlled failure in temporary test directories only.
        with patch.object(
            Path,
            "replace",
            new=interrupt_after_first_page,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Legacy processing cannot overwrite published content",
            ):
                self.process()

        self.assertFalse(first_published)

        # The previously verified snapshot must remain intact.
        self.assertEqual(
            published_files(),
            files_before,
            "Failed publication modified the previous snapshot.",
        )
        self.assertEqual(
            self.report_path.read_bytes(),
            report_before,
        )
        self.assertEqual(
            sync.verify_published_crawl_run(**proof),
            verified_before,
        )



    def test_legacy_blocks_overwrite_before_crash_trigger(self):
        """Legacy overwrite protection prevents the crash trigger."""
        import hashlib
        import json
        import subprocess
        import sys

        from app.ingestion.crawler import page_id

        # Establish the last known-good snapshot.
        self.process()

        proof = {
            "manifest_path": self.manifest_path,
            "processed_pages_dir": self.processed_dir,
            "report_path": self.report_path,
        }

        verified_before = sync.verify_published_crawl_run(
            **proof
        )

        files_before = {
            item.name: item.read_bytes()
            for item in self.processed_dir.iterdir()
            if item.is_file()
            and item.suffix in (".md", ".json")
        }

        report_before = self.report_path.read_bytes()

        # Prepare changed course content in the temporary fixture.
        identifier = page_id(self.urls[0])
        raw_markdown = self.raw_dir / f"{identifier}.md"
        raw_metadata = self.raw_dir / f"{identifier}.json"

        fresh_content = (
            b"# Updated Course A\\n\\n"
            b"Learn advanced Python, APIs, and testing.\\n"
        )

        raw_markdown.write_bytes(fresh_content)

        metadata = json.loads(
            raw_metadata.read_text(encoding="utf-8")
        )
        metadata["content_hash"] = hashlib.sha256(
            fresh_content
        ).hexdigest()

        raw_metadata.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        target = self.processed_dir / f"{identifier}.md"

        # Only this child process will terminate abruptly.
        child_code = """
import os
import sys
from pathlib import Path
from unittest.mock import patch

from app.ingestion.sync import process_crawl_run

target = Path(sys.argv[1])
original_replace = Path.replace

def terminate_after_publication(current, destination):
    result = original_replace(current, destination)

    if Path(destination) == target:
        os._exit(83)

    return result

with patch.object(
    Path,
    "replace",
    new=terminate_after_publication,
):
    process_crawl_run(
        manifest_path=Path(sys.argv[2]),
        raw_pages_dir=Path(sys.argv[3]),
        processed_pages_dir=Path(sys.argv[4]),
        report_path=Path(sys.argv[5]),
        lock_path=Path(sys.argv[6]),
    )

raise RuntimeError("Expected hard-crash point was not reached.")
"""

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                str(target),
                str(self.manifest_path),
                str(self.raw_dir),
                str(self.processed_dir),
                str(self.report_path),
                str(self.manifest_path.parent / "refresh.lock"),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertNotEqual(
            child.returncode,
            83,
            "Legacy processing unexpectedly reached the crash trigger.",
        )
        self.assertNotEqual(
            child.returncode,
            0,
            "Legacy processing unexpectedly succeeded.",
        )
        self.assertIn(
            "Legacy processing cannot overwrite published content.",
            child.stderr,
        )

        # The previous verified snapshot must survive the crash.
        files_after = {
            item.name: item.read_bytes()
            for item in self.processed_dir.iterdir()
            if item.is_file()
            and item.suffix in (".md", ".json")
        }

        self.assertEqual(
            files_after,
            files_before,
            "Hard crash corrupted the last published snapshot.",
        )

        self.assertEqual(
            self.report_path.read_bytes(),
            report_before,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**proof),
            verified_before,
        )



    def test_verified_generation_survives_source_changes(self):
        """A prepared generation must use independent verified files."""
        from app.ingestion import publication

        self.process()

        original_proof = {
            "manifest_path": self.manifest_path,
            "processed_pages_dir": self.processed_dir,
            "report_path": self.report_path,
        }

        verified_before = sync.verify_published_crawl_run(
            **original_proof
        )

        generations_dir = (
            self.manifest_path.parent / "generations"
        )

        snapshot = publication.prepare_verified_generation(
            **original_proof,
            generations_dir=generations_dir,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**snapshot),
            verified_before,
        )

        self.assertTrue(
            Path(snapshot["manifest_path"]).is_relative_to(
                generations_dir
            )
        )
        self.assertTrue(
            Path(snapshot["processed_pages_dir"]).is_relative_to(
                generations_dir
            )
        )
        self.assertTrue(
            Path(snapshot["report_path"]).is_relative_to(
                generations_dir
            )
        )

        # Simulate subsequently damaged mutable source files.
        from app.ingestion.crawler import page_id

        identifier = page_id(self.urls[0])

        (self.processed_dir / f"{identifier}.md").write_text(
            "CORRUPTED SOURCE",
            encoding="utf-8",
        )

        self.report_path.write_text(
            '{"status": "incomplete"}',
            encoding="utf-8",
        )

        self.manifest_path.write_text(
            '{"status": "incomplete"}',
            encoding="utf-8",
        )

        # The prepared generation must still verify independently.
        self.assertEqual(
            sync.verify_published_crawl_run(**snapshot),
            verified_before,
        )

    def test_existing_generation_cannot_be_overwritten(self):
        """A published generation must never be silently replaced."""
        from app.ingestion import publication

        self.process()

        args = {
            "manifest_path": self.manifest_path,
            "processed_pages_dir": self.processed_dir,
            "report_path": self.report_path,
            "generations_dir": (
                self.manifest_path.parent / "generations"
            ),
        }

        first = publication.prepare_verified_generation(**args)

        with self.assertRaises(FileExistsError):
            publication.prepare_verified_generation(**args)

        # The rejected operation must not damage the first copy.
        sync.verify_published_crawl_run(**first)



    def test_active_pointer_resolves_verified_generation(self):
        """An active pointer resolves to one pinned verified snapshot."""
        from app.ingestion import publication

        self.process()

        root = self.manifest_path.parent
        generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        pointer = root / "active-generation.json"

        publication.publish_active_generation(
            generation_proof=generation,
            active_pointer_path=pointer,
            lock_path=root / "refresh.lock",
        )

        pinned = publication.resolve_active_generation(
            active_pointer_path=pointer,
        )

        for key in (
            "manifest_path",
            "processed_pages_dir",
            "report_path",
        ):
            self.assertEqual(
                Path(pinned[key]).resolve(),
                Path(generation[key]).resolve(),
            )

        self.assertEqual(
            sync.verify_published_crawl_run(**pinned),
            sync.verify_published_crawl_run(**generation),
        )

    def test_invalid_generation_cannot_change_active_pointer(self):
        """An invalid candidate must leave the active snapshot intact."""
        from app.ingestion import publication

        self.process()

        root = self.manifest_path.parent
        generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        publication.publish_active_generation(
            generation_proof=generation,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        original_pointer_bytes = pointer.read_bytes()

        invalid = dict(generation)
        invalid["report_path"] = root / "missing-report.json"

        with self.assertRaises(ValueError):
            publication.publish_active_generation(
                generation_proof=invalid,
                active_pointer_path=pointer,
                lock_path=lock_path,
            )

        self.assertEqual(
            pointer.read_bytes(),
            original_pointer_bytes,
        )

        pinned = publication.resolve_active_generation(
            active_pointer_path=pointer,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**pinned),
            sync.verify_published_crawl_run(**generation),
        )



    def test_new_run_publishes_without_mutating_previous_generation(self):
        """A fresh run must switch generations without rewriting old files."""
        import hashlib
        import json

        from app.ingestion import publication
        from app.ingestion.crawler import page_id

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        # Establish the previously published content.
        self.process()

        old_generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        publication.publish_active_generation(
            generation_proof=old_generation,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        old_verified = sync.verify_published_crawl_run(
            **old_generation
        )

        old_files = {
            item.name: item.read_bytes()
            for item in self.processed_dir.iterdir()
            if item.is_file()
        }
        old_report = self.report_path.read_bytes()

        # Simulate a newly completed website crawl run.
        new_run_id = "crawl-run-2026-b"

        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        manifest["run_id"] = new_run_id

        for record in manifest["pages"]:
            record["run_id"] = new_run_id

        for index, url in enumerate(self.urls):
            identifier = page_id(url)
            markdown_path = self.raw_dir / f"{identifier}.md"
            metadata_path = self.raw_dir / f"{identifier}.json"

            content = (
                f"# Updated Course {index + 1}\\n\\n"
                "New lessons, practical projects, and assessments.\\n"
            ).encode("utf-8")

            markdown_path.write_bytes(content)

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            metadata["run_id"] = new_run_id
            metadata["content_hash"] = (
                hashlib.sha256(content).hexdigest()
            )
            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

        self.manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        # New API: process, verify and publish under one lock.
        new_generation = publication.process_and_publish_generation(
            manifest_path=self.manifest_path,
            raw_pages_dir=self.raw_dir,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        pinned = publication.resolve_active_generation(
            active_pointer_path=pointer,
        )

        self.assertEqual(
            Path(pinned["manifest_path"]),
            Path(new_generation["manifest_path"]),
        )

        new_verified = sync.verify_published_crawl_run(
            **pinned
        )

        self.assertEqual(new_verified["run_id"], new_run_id)

        # The previous generation must remain independently valid.
        self.assertEqual(
            sync.verify_published_crawl_run(**old_generation),
            old_verified,
        )

        # Legacy mutable publication must not be touched.
        self.assertEqual(
            {
                item.name: item.read_bytes()
                for item in self.processed_dir.iterdir()
                if item.is_file()
            },
            old_files,
        )
        self.assertEqual(
            self.report_path.read_bytes(),
            old_report,
        )



    def test_crash_before_pointer_switch_allows_safe_retry(self):
        """A verified generation left by a hard crash must be reusable."""
        import hashlib
        import json
        import subprocess
        import sys

        from app.ingestion import publication
        from app.ingestion.crawler import page_id

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        # Establish an existing, independently verified generation.
        self.process()

        old_generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        publication.publish_active_generation(
            generation_proof=old_generation,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        old_pointer = pointer.read_bytes()
        old_verified = sync.verify_published_crawl_run(
            **old_generation
        )

        # Prepare a different, complete crawl run.
        new_run_id = "crawl-run-2026-crash-retry"

        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        manifest["run_id"] = new_run_id

        for record in manifest["pages"]:
            record["run_id"] = new_run_id

        for index, url in enumerate(self.urls):
            identifier = page_id(url)

            markdown_path = self.raw_dir / f"{identifier}.md"
            metadata_path = self.raw_dir / f"{identifier}.json"

            content = (
                f"# Revised Course {index + 1}\\n\\n"
                "Updated course lessons and practical projects.\\n"
            ).encode("utf-8")

            markdown_path.write_bytes(content)

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            metadata["run_id"] = new_run_id
            metadata["content_hash"] = (
                hashlib.sha256(content).hexdigest()
            )

            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

        self.manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        # Crash exactly before the atomic pointer replacement.
        child_code = """
import os
import sys
from pathlib import Path
from unittest.mock import patch

from app.ingestion.publication import (
    process_and_publish_generation,
)

pointer = Path(sys.argv[1])
original_replace = os.replace

def crash_before_pointer_switch(source, destination):
    if Path(destination) == pointer:
        os._exit(83)
    return original_replace(source, destination)

with patch("os.replace", new=crash_before_pointer_switch):
    process_and_publish_generation(
        manifest_path=Path(sys.argv[2]),
        raw_pages_dir=Path(sys.argv[3]),
        active_pointer_path=pointer,
        lock_path=Path(sys.argv[4]),
    )

raise RuntimeError("Expected crash point was not reached.")
"""

        child = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                str(pointer),
                str(self.manifest_path),
                str(self.raw_dir),
                str(lock_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        self.assertEqual(
            child.returncode,
            83,
            child.stdout + child.stderr,
        )

        # The pointer and previous generation must survive.
        self.assertEqual(pointer.read_bytes(), old_pointer)

        self.assertEqual(
            sync.verify_published_crawl_run(**old_generation),
            old_verified,
        )

        generation_name = "run-" + hashlib.sha256(
            new_run_id.encode("utf-8")
        ).hexdigest()

        self.assertTrue(
            (root / "generations" / generation_name).is_dir()
        )

        # An abrupt process exit bypasses TemporaryDirectory cleanup.
        abandoned_before_retry = [
            item.name
            for item in root.iterdir()
            if item.name.startswith(".processing-generation-")
        ]

        self.assertTrue(
            abandoned_before_retry,
            "Crash simulation did not leave a processing directory.",
        )

        # A retry must verify and reuse the completed generation,
        # not overwrite it or leave the active pointer stuck.
        retried = publication.process_and_publish_generation(
            manifest_path=self.manifest_path,
            raw_pages_dir=self.raw_dir,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        pinned = publication.resolve_active_generation(
            active_pointer_path=pointer,
        )

        self.assertEqual(
            Path(pinned["manifest_path"]),
            Path(retried["manifest_path"]),
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**pinned)["run_id"],
            new_run_id,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**old_generation),
            old_verified,
        )

        abandoned_after_retry = [
            item.name
            for item in root.iterdir()
            if item.name.startswith(".processing-generation-")
        ]

        self.assertEqual(
            abandoned_after_retry,
            [],
            "Retry left abandoned processing directories.",
        )



    def test_same_run_id_changed_content_rejects_reuse(self):
        """A run ID cannot authorize different published content."""
        import hashlib
        import json

        from app.ingestion import publication
        from app.ingestion.crawler import page_id

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        original = publication.process_and_publish_generation(
            manifest_path=self.manifest_path,
            raw_pages_dir=self.raw_dir,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        pointer_before = pointer.read_bytes()
        verified_before = sync.verify_published_crawl_run(
            **original
        )

        identifier = page_id(self.urls[0])
        raw_page = self.raw_dir / f"{identifier}.md"
        raw_metadata = self.raw_dir / f"{identifier}.json"

        changed = (
            b"# Changed Course Curriculum\\n\\n"
            b"Updated lessons with different learning outcomes.\\n"
        )

        raw_page.write_bytes(changed)

        metadata = json.loads(
            raw_metadata.read_text(encoding="utf-8")
        )
        metadata["content_hash"] = hashlib.sha256(
            changed
        ).hexdigest()

        raw_metadata.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        # The run ID is deliberately unchanged.
        with self.assertRaisesRegex(
            ValueError,
            "(?i)generation|differs|match",
        ):
            publication.process_and_publish_generation(
                manifest_path=self.manifest_path,
                raw_pages_dir=self.raw_dir,
                active_pointer_path=pointer,
                lock_path=lock_path,
            )

        self.assertEqual(
            pointer.read_bytes(),
            pointer_before,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**original),
            verified_before,
        )

    def test_cleanup_preserves_unrelated_prefix_directory(self):
        """Cleanup must not delete arbitrary similarly named folders."""
        from app.ingestion import publication

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        # This is NOT a tempfile generated by our workflow.
        unrelated = root / ".processing-generation-user-notes"
        unrelated.mkdir()

        sentinel = unrelated / "keep.txt"
        sentinel.write_text(
            "Important unrelated file",
            encoding="utf-8",
        )

        generation = publication.process_and_publish_generation(
            manifest_path=self.manifest_path,
            raw_pages_dir=self.raw_dir,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        self.assertTrue(
            sentinel.is_file(),
            "Cleanup deleted an unrelated directory.",
        )

        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "Important unrelated file",
        )

        sync.verify_published_crawl_run(**generation)



    def test_legacy_processing_rejects_published_snapshot_overwrite(self):
        """Legacy processing must not overwrite a published snapshot."""
        import hashlib
        import json

        from app.ingestion.crawler import page_id

        # Establish a complete, verified legacy snapshot.
        self.process()

        proof = {
            "manifest_path": self.manifest_path,
            "processed_pages_dir": self.processed_dir,
            "report_path": self.report_path,
        }

        verified_before = sync.verify_published_crawl_run(**proof)

        files_before = {
            item.name: item.read_bytes()
            for item in self.processed_dir.iterdir()
            if item.is_file()
        }

        report_before = self.report_path.read_bytes()

        # Change source content without changing the run ID.
        identifier = page_id(self.urls[0])

        raw_page = self.raw_dir / f"{identifier}.md"
        raw_metadata = self.raw_dir / f"{identifier}.json"

        updated_content = (
            b"# Revised Course Content\n\n"
            b"New practical lessons and assignments.\n"
        )

        raw_page.write_bytes(updated_content)

        metadata = json.loads(
            raw_metadata.read_text(encoding="utf-8")
        )
        metadata["content_hash"] = hashlib.sha256(
            updated_content
        ).hexdigest()

        raw_metadata.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        # Existing published files must not be replaced.
        with self.assertRaisesRegex(
            ValueError,
            "(?i)published|overwrite|immutable|generation",
        ):
            self.process()

        self.assertEqual(
            {
                item.name: item.read_bytes()
                for item in self.processed_dir.iterdir()
                if item.is_file()
            },
            files_before,
        )

        self.assertEqual(
            self.report_path.read_bytes(),
            report_before,
        )

        self.assertEqual(
            sync.verify_published_crawl_run(**proof),
            verified_before,
        )



    def test_pinned_activation_rejects_wrong_run_before_qdrant(self):
        """A stale caller must not activate the wrong crawl run."""
        from unittest.mock import patch

        from app.ingestion import publication, sync

        self.process()

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        publication.publish_active_generation(
            generation_proof=generation,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        pointer_before = pointer.read_bytes()

        with patch.object(
            sync,
            "_activate_collection_snapshot_locked",
        ) as activate:
            with self.assertRaisesRegex(
                ValueError,
                "(?i)run|generation|mismatch",
            ):
                publication.activate_pinned_generation_snapshot(
                    object(),
                    active_pointer_path=pointer,
                    expected_run_id="different-crawl-run",
                    lock_path=lock_path,
                    alias_name="test-active",
                    expected_active_collection="test-v1",
                    staged_collection="test-v2",
                    expected_chunk_ids=[],
                )

            activate.assert_not_called()

        self.assertEqual(
            pointer.read_bytes(),
            pointer_before,
        )

    def test_pinned_activation_uses_immutable_generation_paths(self):
        """Qdrant must receive paths pinned by the active pointer."""
        from unittest.mock import patch

        from app.ingestion import publication, sync

        self.process()

        root = self.manifest_path.parent
        pointer = root / "active-generation.json"
        lock_path = root / "refresh.lock"

        generation = publication.prepare_verified_generation(
            manifest_path=self.manifest_path,
            processed_pages_dir=self.processed_dir,
            report_path=self.report_path,
            generations_dir=root / "generations",
        )

        publication.publish_active_generation(
            generation_proof=generation,
            active_pointer_path=pointer,
            lock_path=lock_path,
        )

        expected_result = {"activated": True}
        client = object()

        with patch.object(
            sync,
            "_activate_collection_snapshot_locked",
            return_value=expected_result,
        ) as activate:
            result = publication.activate_pinned_generation_snapshot(
                client,
                active_pointer_path=pointer,
                expected_run_id=self.run_id,
                lock_path=lock_path,
                alias_name="test-active",
                expected_active_collection="test-v1",
                staged_collection="test-v2",
                expected_chunk_ids=["expected-point"],
            )

            self.assertEqual(result, expected_result)
            activate.assert_called_once()

            args, kwargs = activate.call_args

            self.assertIs(args[0], client)
            self.assertEqual(kwargs["alias_name"], "test-active")
            self.assertEqual(
                kwargs["expected_active_collection"],
                "test-v1",
            )
            self.assertEqual(kwargs["staged_collection"], "test-v2")
            self.assertEqual(
                kwargs["expected_chunk_ids"],
                ["expected-point"],
            )

            for key in (
                "manifest_path",
                "processed_pages_dir",
                "report_path",
            ):
                self.assertEqual(
                    Path(kwargs[key]),
                    Path(generation[key]),
                )

        # No legacy mutable paths may be passed to Qdrant.
        self.assertNotEqual(
            Path(generation["processed_pages_dir"]),
            self.processed_dir,
        )


if __name__ == "__main__":
    unittest.main()
