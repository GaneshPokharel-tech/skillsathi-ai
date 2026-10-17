import unittest

from app.ingestion.url_discovery import (
    discover_public_urls,
    normalize_content_url,
)


class TestURLDiscovery(unittest.TestCase):

    def test_allows_course_url(self):
        url = normalize_content_url("/courses/python-training-in-nepal")
        self.assertEqual(
            url,
            "https://skillshikshya.com/" "courses/python-training-in-nepal",
        )

    def test_rejects_external_domain(self):
        self.assertIsNone(normalize_content_url("https://evil.com/courses"))

    def test_rejects_private_area(self):
        self.assertIsNone(normalize_content_url("https://skillshikshya.com/payment"))

    def test_removes_duplicates(self):
        links = [
            {"href": "/faq"},
            {"href": "/faq#section"},
            {"href": "/courses"},
        ]
        self.assertEqual(len(discover_public_urls(links)), 2)


if __name__ == "__main__":
    unittest.main()
