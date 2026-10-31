import unittest
from app.ingestion.crawler import normalize_content_url


class ExpandedPublicScopeTests(unittest.TestCase):
    def test_informational_pages_are_approved(self):
        for path in ('/announcements', '/career', '/corporate-training', '/it-training-for-plus-2-completed-students', '/it-training-for-see-appeared-students', '/master-class', '/news-media-coverage', '/online-counselling', '/partners', '/privacy-policy', '/success-stories', '/terms-of-service', '/testimonials', '/workshop'):
            url = "https://skillshikshya.com" + path
            with self.subTest(path=path):
                self.assertEqual(normalize_content_url(url), url)

    def test_interactive_and_unapproved_paths_are_rejected(self):
        for path in ('/payment', '/verify-certificates', '/online-compiler', '/enroll-form', '/form', '/announcements-private'):
            url = "https://skillshikshya.com" + path
            with self.subTest(path=path):
                self.assertIsNone(normalize_content_url(url))

    def test_other_domain_is_rejected(self):
        self.assertIsNone(
            normalize_content_url(
                "https://example.com/announcements"
            )
        )


if __name__ == "__main__":
    unittest.main()
