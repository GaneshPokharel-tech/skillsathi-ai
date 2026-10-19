import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.processor import clean_markdown
from app.ingestion import processor as processor_module


class TestMarkdownCleaner(unittest.TestCase):

    def test_removes_images_and_repeated_navigation_noise(self):
        navigation = "[Home](https://skillshikshya.com/) | [Courses](https://skillshikshya.com/courses)"
        content = f"""# Python Training

![Course banner](https://example.com/course.png)

{navigation}
{navigation}



Learn Python programming through practical projects.
"""

        cleaned = clean_markdown(content)

        self.assertNotIn("https://example.com/course.png", cleaned)
        self.assertNotIn(navigation, cleaned)
        self.assertIn("Course banner", cleaned)
        self.assertIn("Learn Python programming through practical projects.", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_removes_leading_site_navigation_but_keeps_course_title(self):
        content = """×
S
Skill Shikshya
●Usually replies in a few minutes
👋 Looking for the right course?
[See all courses](https://skillshikshya.com/courses)[Corporate training](https://skillshikshya.com/corporate-training)[Counselling](https://skillshikshya.com/online-counselling)
For Corporates
For Students
Call us directly
Send us an email
[](https://skillshikshya.com/)
[Enroll Now](https://skillshikshya.com/enroll-form)MENU
Fee: NPR 25,000
[Detailed syllabus](https://skillshikshya.com/courses/python-training#syllabus)
Learn Complete Accounting Training with AI Tools
# Complete Accounting Training
The 48-hour program covers practical accounting, tax, and financial reporting.
"""

        cleaned = clean_markdown(content)

        self.assertNotIn("For Corporates", cleaned)
        self.assertNotIn("Enroll Now", cleaned)
        self.assertNotIn("online-counselling", cleaned)
        self.assertNotIn("[](https://skillshikshya.com/)", cleaned)
        self.assertIn("Fee: NPR 25,000", cleaned)
        self.assertIn(
            "[Detailed syllabus](https://skillshikshya.com/courses/python-training#syllabus)",
            cleaned,
        )
        self.assertIn("Learn Complete Accounting Training with AI Tools", cleaned)
        self.assertIn("# Complete Accounting Training", cleaned)
        self.assertIn("The 48-hour program covers practical accounting", cleaned)

    def test_preserves_course_information_formatting_and_source_links(self):
        content = """# Python Training

Duration: 8 weeks
Fee: NPR 25,000

## Curriculum

| Module | Topics |
| --- | --- |
| Python | Variables, loops, and functions |

- Prerequisite: Basic computer skills
- FAQ: Do I need prior experience?

Read the [official syllabus](https://skillshikshya.com/courses/python-training#syllabus).

Inline example: `print("hello")`
"""

        cleaned = clean_markdown(content)

        for expected in (
            "# Python Training",
            "Duration: 8 weeks",
            "Fee: NPR 25,000",
            "## Curriculum",
            "| Module | Topics |",
            "- Prerequisite: Basic computer skills",
            "- FAQ: Do I need prior experience?",
            "[official syllabus](https://skillshikshya.com/courses/python-training#syllabus)",
            '`print("hello")`',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, cleaned)

    def test_normalizes_entities_and_escaped_heading_without_flattening(self):
        content = "\\# Course &amp; Career\n\n\nDescription"

        cleaned = clean_markdown(content)

        self.assertEqual(cleaned, "# Course & Career\n\nDescription")

    def test_handles_empty_and_malformed_markdown_safely(self):
        self.assertEqual(clean_markdown(" \n\t "), "")

        malformed = "![unfinished image\n[unfinished link"
        self.assertEqual(clean_markdown(malformed), malformed)


class TestPageCleaningPipeline(unittest.TestCase):

    def test_processes_approved_pages_and_skips_external_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_pages = root / "raw"
            processed_pages = root / "processed" / "pages"
            report_path = root / "processed" / "cleaning_report.json"
            raw_pages.mkdir()

            raw_content = "# Python Training\n\nFee: NPR 25,000\n"
            approved_metadata = {
                "source_url": "https://skillshikshya.com/courses/python-training",
                "crawled_at": "2026-10-15T12:00:00+00:00",
            }
            external_metadata = {
                "source_url": "https://example.com/courses/python-training",
                "crawled_at": "2026-10-15T12:00:00+00:00",
            }

            (raw_pages / "approved.md").write_text(raw_content, encoding="utf-8")
            (raw_pages / "approved.json").write_text(
                json.dumps(approved_metadata), encoding="utf-8"
            )
            (raw_pages / "external.md").write_text(raw_content, encoding="utf-8")
            (raw_pages / "external.json").write_text(
                json.dumps(external_metadata), encoding="utf-8"
            )

            with (
                patch.object(processor_module, "RAW_PAGES_DIR", raw_pages),
                patch.object(
                    processor_module, "PROCESSED_PAGES_DIR", processed_pages
                ),
                patch.object(
                    processor_module, "CLEANING_REPORT_PATH", report_path
                ),
            ):
                report = processor_module.clean_pages()

            self.assertEqual(
                (report["processed"], report["skipped"], report["failed"]),
                (1, 1, 0),
            )
            self.assertEqual(
                (raw_pages / "approved.md").read_text(encoding="utf-8"),
                raw_content,
            )
            self.assertFalse((processed_pages / "external.md").exists())

            cleaned_metadata = json.loads(
                (processed_pages / "approved.json").read_text(encoding="utf-8")
            )
            cleaned_content = (processed_pages / "approved.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(cleaned_metadata["source_url"], approved_metadata["source_url"])
            self.assertEqual(
                cleaned_metadata["crawled_at"], approved_metadata["crawled_at"]
            )
            self.assertEqual(cleaned_metadata["raw_character_count"], len(raw_content))
            self.assertEqual(
                cleaned_metadata["cleaned_character_count"], len(cleaned_content)
            )
            self.assertEqual(
                cleaned_metadata["content_hash"],
                hashlib.sha256(cleaned_content.encode("utf-8")).hexdigest(),
            )
            self.assertIn("cleaned_at", cleaned_metadata)

            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["processed"], 1)
            self.assertEqual(saved_report["pages"][0]["raw_character_count"], len(raw_content))


if __name__ == "__main__":
    unittest.main()


from unittest import TestCase


class TestScopedProcessorIsolation(TestCase):

    def test_processes_only_selected_website(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from app.ingestion import processor
        from app.ingestion.crawler import build_crawl_scope

        scope = build_crawl_scope("https://example.org/")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            processed_root = root / "processed"

            site_pages = raw_root / "example.org" / "pages"
            site_pages.mkdir(parents=True)

            # Approved course belonging to the selected website.
            (site_pages / "python.md").write_text(
                "# Python Course\n\nLearn Python programming.",
                encoding="utf-8",
            )
            (site_pages / "python.json").write_text(
                json.dumps({
                    "source_url": "https://example.org/courses/python",
                    "crawled_at": "2026-10-19T10:00:00+00:00",
                }),
                encoding="utf-8",
            )

            # External-domain metadata must never be approved.
            (site_pages / "external.md").write_text(
                "# External Course\n\nUntrusted content.",
                encoding="utf-8",
            )
            (site_pages / "external.json").write_text(
                json.dumps({
                    "source_url": "https://other-site.com/courses/python",
                    "crawled_at": "2026-10-19T10:00:00+00:00",
                }),
                encoding="utf-8",
            )

            with (
                patch.object(
                    processor,
                    "RAW_PAGES_DIR",
                    raw_root / "pages",
                ),
                patch.object(
                    processor,
                    "PROCESSED_PAGES_DIR",
                    processed_root / "pages",
                ),
                patch.object(
                    processor,
                    "CLEANING_REPORT_PATH",
                    processed_root / "cleaning_report.json",
                ),
            ):
                report = processor.clean_pages(scope=scope)

            output = processed_root / "example.org"

            self.assertEqual(report["processed"], 1)
            self.assertEqual(report["skipped"], 1)

            self.assertTrue(
                (output / "pages" / "python.md").exists()
            )
            self.assertFalse(
                (output / "pages" / "external.md").exists()
            )
            self.assertTrue(
                (output / "cleaning_report.json").exists()
            )

            # No accidental writes to the legacy shared folder.
            self.assertFalse(
                (processed_root / "pages").exists()
            )


from unittest import IsolatedAsyncioTestCase


class TestScopedIngestionPipeline(IsolatedAsyncioTestCase):

    async def test_crawler_to_processor_storage_consistency(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.ingestion import crawler, processor

        scope = crawler.build_crawl_scope("https://example.org/")

        async def fake_crawl(*, url, config):
            if url == "https://example.org/":
                return SimpleNamespace(
                    success=True,
                    url=url,
                    markdown="# Example Academy\n\nLearn new skills.",
                    links={
                        "internal": [
                            {"href": "/courses/python"},
                            {"href": "https://evil.com/private"},
                        ]
                    },
                )

            if url == "https://example.org/courses/python":
                return SimpleNamespace(
                    success=True,
                    url=url,
                    markdown=(
                        "# Python Programming Course\n\n"
                        "Learn Python programming fundamentals."
                    ),
                )

            raise AssertionError(f"Unexpected crawl URL: {url}")

        worker = MagicMock()
        worker.arun = AsyncMock(side_effect=fake_crawl)

        browser = MagicMock()
        browser.__aenter__ = AsyncMock(return_value=worker)
        browser.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            processed_root = root / "processed"

            with (
                patch.object(crawler, "OUTPUT_DIR", raw_root),
                patch.object(
                    crawler, "URLS_FILE",
                    raw_root / "legacy_urls.json"
                ),
                patch.object(
                    crawler, "PAGES_DIR",
                    raw_root / "legacy_pages"
                ),
                patch.object(
                    crawler, "validate_dns_scope",
                    return_value=None
                ),
                patch.object(
                    crawler, "AsyncWebCrawler",
                    return_value=browser
                ),
                patch.object(
                    processor, "RAW_PAGES_DIR",
                    raw_root / "pages"
                ),
                patch.object(
                    processor, "PROCESSED_PAGES_DIR",
                    processed_root / "pages"
                ),
                patch.object(
                    processor, "CLEANING_REPORT_PATH",
                    processed_root / "legacy_report.json"
                ),
            ):
                await crawler.crawl_homepage(scope=scope)
                await crawler.crawl_pages(scope=scope)
                report = processor.clean_pages(scope=scope)

            page_name = crawler.page_id(
                "https://example.org/courses/python"
            )

            raw_page = (
                raw_root / "example.org" /
                "pages" / f"{page_name}.md"
            )
            cleaned_page = (
                processed_root / "example.org" /
                "pages" / f"{page_name}.md"
            )
            metadata_file = cleaned_page.with_suffix(".json")

            self.assertEqual(worker.arun.await_count, 2)
            self.assertTrue(raw_page.exists())
            self.assertTrue(cleaned_page.exists())
            self.assertTrue(metadata_file.exists())
            self.assertEqual(report["processed"], 1)
            self.assertEqual(report["failed"], 0)

            metadata = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["source_url"],
                "https://example.org/courses/python",
            )

            self.assertFalse(
                (raw_root / "legacy_urls.json").exists()
            )
            self.assertFalse(
                (processed_root / "pages").exists()
            )


class TestProcessorCLI(TestCase):

    def test_custom_url_reaches_processor(self):
        from unittest.mock import patch
        from app.ingestion import processor

        report = {"processed": 1, "skipped": 0, "failed": 0}

        with patch.object(
            processor, "clean_pages", return_value=report
        ) as clean:
            processor.main(["--url", "https://example.org/"])

        clean.assert_called_once()
        scope = clean.call_args.kwargs["scope"]
        self.assertEqual(scope.base_url, "https://example.org/")

    def test_default_uses_legacy_processing(self):
        from unittest.mock import patch
        from app.ingestion import processor

        report = {"processed": 0, "skipped": 0, "failed": 0}

        with patch.object(
            processor, "clean_pages", return_value=report
        ) as clean:
            processor.main([])

        clean.assert_called_once_with(scope=None)

    def test_rejects_insecure_url_before_processing(self):
        from unittest.mock import patch
        from app.ingestion import processor

        with patch.object(processor, "clean_pages") as clean:
            with self.assertRaises(ValueError):
                processor.main(["--url", "http://example.org/"])

        clean.assert_not_called()
