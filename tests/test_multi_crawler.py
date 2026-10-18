import unittest

from app.ingestion.crawler import (
    prepare_urls,
    page_id,
)


class TestMultiCrawler(unittest.TestCase):

    def test_filters_duplicates_and_unsafe_urls(self):
        urls = [
            "https://skillshikshya.com/courses/python",
            "https://skillshikshya.com/courses/python#overview",
            "https://skillshikshya.com/faq",
            "https://skillshikshya.com/payment",
            "https://evil.com/courses",
            "https://skillshikshya.com/",
        ]

        result = prepare_urls(urls, limit=5)

        self.assertEqual(len(result), 2)
        self.assertIn(
            "https://skillshikshya.com/faq",
            result,
        )

    def test_page_id_is_stable(self):
        url = "https://skillshikshya.com/faq"

        self.assertEqual(page_id(url), page_id(url))

    def test_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            prepare_urls([], limit=0)

    def test_prioritizes_course_details(self):
        urls = [
            "https://skillshikshya.com/blogs/article-one",
            "https://skillshikshya.com/courses/python-training",
            "https://skillshikshya.com/blogs/article-two",
            "https://skillshikshya.com/courses/mern-stack",
        ]

        result = prepare_urls(urls, limit=2)

        self.assertEqual(
            result,
            [
                "https://skillshikshya.com/courses/mern-stack",
                "https://skillshikshya.com/courses/python-training",
            ],
        )


if __name__ == "__main__":
    unittest.main()
