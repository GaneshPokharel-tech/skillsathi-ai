import unittest

from app.ingestion.crawler import validate_public_url


class TestCrawlerSecurity(unittest.TestCase):

    def test_approved_website(self):
        validate_public_url("https://skillshikshya.com/")

    def test_rejects_external_domain(self):
        with self.assertRaises(ValueError):
            validate_public_url("https://skillshikshya.com.evil.com/")

    def test_rejects_http(self):
        with self.assertRaises(ValueError):
            validate_public_url("http://skillshikshya.com/")


if __name__ == "__main__":
    unittest.main()
