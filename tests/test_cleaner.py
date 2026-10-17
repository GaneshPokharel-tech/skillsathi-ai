import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ingestion.cleaner import clean_markdown
from app.ingestion import clean_pages as clean_pages_module


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
                patch.object(clean_pages_module, "RAW_PAGES_DIR", raw_pages),
                patch.object(
                    clean_pages_module, "PROCESSED_PAGES_DIR", processed_pages
                ),
                patch.object(
                    clean_pages_module, "CLEANING_REPORT_PATH", report_path
                ),
            ):
                report = clean_pages_module.clean_pages()

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
