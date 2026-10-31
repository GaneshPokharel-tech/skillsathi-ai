import hashlib
import unittest

from app.ingestion.change_detection import (
    merge_discovered_urls,
    plan_content_changes,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class ChangeDetectionTests(unittest.TestCase):
    def test_detects_added_updated_and_unchanged_pages(self):
        previous = {
            "https://skillshikshya.com/about": digest("old about"),
            "https://skillshikshya.com/faq": digest("old faq"),
        }
        fresh = {
            "https://skillshikshya.com/about": digest("old about"),
            "https://skillshikshya.com/faq": digest("new faq"),
            "https://skillshikshya.com/announcements": digest("new"),
        }

        result = plan_content_changes(previous, fresh)

        self.assertEqual(
            result["added"],
            ["https://skillshikshya.com/announcements"],
        )
        self.assertEqual(
            result["updated"],
            ["https://skillshikshya.com/faq"],
        )
        self.assertEqual(
            result["unchanged"],
            ["https://skillshikshya.com/about"],
        )

    def test_discovery_gap_does_not_delete_page(self):
        url = "https://skillshikshya.com/blogs/accounting-career-path"

        result = plan_content_changes(
            {url: digest("existing")},
            {},
        )

        self.assertEqual(result["removed"], [])
        self.assertEqual(result["unverified_missing"], [url])

    def test_confirmed_removal(self):
        url = "https://skillshikshya.com/blogs/accounting-career-path"

        result = plan_content_changes(
            {url: digest("existing")},
            {},
            confirmed_removed={url},
        )

        self.assertEqual(result["removed"], [url])
        self.assertEqual(result["unverified_missing"], [])

    def test_cannot_remove_page_still_present(self):
        url = "https://skillshikshya.com/about"

        with self.assertRaises(ValueError):
            plan_content_changes(
                {url: digest("old")},
                {url: digest("new")},
                confirmed_removed={url},
            )

    def test_discovery_merge_preserves_known_blog(self):
        result = merge_discovered_urls(
            ["https://skillshikshya.com/blogs/accounting-career-path"],
            [
                "https://skillshikshya.com/about",
                "https://skillshikshya.com/payment",
                "https://example.com/about",
            ],
        )

        self.assertEqual(
            result,
            [
                "https://skillshikshya.com/about",
                "https://skillshikshya.com/blogs/accounting-career-path",
            ],
        )

    def test_invalid_hash_rejected(self):
        with self.assertRaises(ValueError):
            plan_content_changes(
                {"https://skillshikshya.com/about": "invalid"},
                {},
            )

    def test_wrong_domain_rejected(self):
        with self.assertRaises(ValueError):
            plan_content_changes(
                {"https://example.com/about": digest("data")},
                {},
            )


if __name__ == "__main__":
    unittest.main()
