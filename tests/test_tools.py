"""Tests for indexed SkillShikshya course discovery."""

import unittest
from types import SimpleNamespace

from app.tools.courses import list_courses


def record(title, url):
    return SimpleNamespace(
        payload={
            "document_title": title,
            "source_url": url,
        }
    )


class FakeQdrantClient:

    def __init__(self, pages, exists=True):
        self.pages = pages
        self.exists = exists
        self.calls = 0

    def collection_exists(self, name):
        return self.exists

    def scroll(self, **kwargs):
        index = self.calls
        self.calls += 1

        records = self.pages[index]
        next_offset = (
            f"page-{index + 1}"
            if index + 1 < len(self.pages)
            else None
        )

        return records, next_offset

    def get_aliases(self):
        """Simulate a legacy Qdrant database without an active alias."""
        from types import SimpleNamespace

        return SimpleNamespace(aliases=[])



class TestCourseDiscovery(unittest.TestCase):

    def setUp(self):
        self.cyber = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )
        self.revit = (
            "https://skillshikshya.com/courses/"
            "autodesk-revit-bim-course"
        )

    def test_deduplicates_and_filters_across_pages(self):
        client = FakeQdrantClient([
            [
                record("Cybersecurity Program", self.cyber),
                record("About Us", "https://skillshikshya.com/about"),
                record("Fake Course", "https://evil.example/courses/fake"),
            ],
            [
                record("Cybersecurity Program", self.cyber),
                record("Autodesk Revit", self.revit),
            ],
        ])

        result = list_courses(client=client)

        self.assertEqual(len(result), 2)
        self.assertEqual(client.calls, 2)
        self.assertEqual(
            [item["title"] for item in result],
            ["Autodesk Revit", "Cybersecurity Program"],
        )

    def test_search_filters_existing_courses(self):
        client = FakeQdrantClient([[
            record("Cybersecurity Program", self.cyber),
            record("Autodesk Revit", self.revit),
        ]])

        result = list_courses(
            client=client,
            query="cyber",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_url"], self.cyber)

    def test_unrelated_pages_are_not_courses(self):
        client = FakeQdrantClient([[
            record("Blogs", "https://skillshikshya.com/blogs"),
            record("Fake", "http://skillshikshya.com/courses/fake"),
        ]])

        self.assertEqual(list_courses(client=client), [])

    def test_missing_collection_raises_error(self):
        client = FakeQdrantClient([], exists=False)

        with self.assertRaises(ValueError):
            list_courses(client=client)

    def test_invalid_query_is_rejected(self):
        with self.assertRaises(ValueError):
            list_courses(query=123)



class TestCourseDetails(unittest.TestCase):

    def setUp(self):
        self.revit = (
            "https://skillshikshya.com/courses/"
            "autodesk-revit-bim-course"
        )
        self.cyber = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )

    def test_exact_course_filter_and_curriculum_priority(self):
        from app.tools.courses import get_course_details

        client = FakeQdrantClient([[
            SimpleNamespace(id="intro", payload={
                "document_title": "Autodesk Revit",
                "source_url": self.revit,
                "section": "Overview",
                "text": "Learn Revit.",
            }),
            SimpleNamespace(id="other", payload={
                "document_title": "Cybersecurity",
                "source_url": self.cyber,
                "section": "Curriculum",
                "text": "Ethical hacking.",
            }),
            SimpleNamespace(id="curriculum", payload={
                "document_title": "Autodesk Revit",
                "source_url": self.revit,
                "section": "Course Curriculum",
                "text": "BIM fundamentals and modeling.",
            }),
        ]])

        result = get_course_details(
            self.revit,
            client=client,
        )

        self.assertEqual(result["title"], "Autodesk Revit")
        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(
            result["evidence"][0]["section"],
            "Course Curriculum",
        )
        self.assertTrue(all(
            item["source_url"] == self.revit
            for item in result["evidence"]
        ))

    def test_limits_evidence_without_hiding_total(self):
        from app.tools.courses import get_course_details

        client = FakeQdrantClient([[
            SimpleNamespace(id=str(i), payload={
                "document_title": "Autodesk Revit",
                "source_url": self.revit,
                "section": "Curriculum",
                "text": f"Course module {i}",
            })
            for i in range(4)
        ]])

        result = get_course_details(
            self.revit,
            client=client,
            max_chunks=2,
        )

        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["total_chunks"], 4)
        self.assertTrue(result["truncated"])

    def test_unknown_course_returns_none(self):
        from app.tools.courses import get_course_details

        client = FakeQdrantClient([[]])

        result = get_course_details(
            self.revit,
            client=client,
        )

        self.assertIsNone(result)

    def test_invalid_course_url_and_limit(self):
        from app.tools.courses import get_course_details

        invalid_urls = [
            "https://evil.example/courses/revit",
            "http://skillshikshya.com/courses/revit",
            "https://skillshikshya.com/about",
        ]

        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    get_course_details(url)

        for limit in (0, 21, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    get_course_details(
                        self.revit,
                        max_chunks=limit,
                    )

    def test_missing_collection(self):
        from app.tools.courses import get_course_details

        client = FakeQdrantClient([], exists=False)

        with self.assertRaises(ValueError):
            get_course_details(
                self.revit,
                client=client,
            )



class TestCourseComparison(unittest.TestCase):

    def setUp(self):
        self.cyber = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )
        self.analytics = (
            "https://skillshikshya.com/courses/"
            "business-data-analytics-job-ready-program"
        )
        self.cyber_details = {
            "title": "Cybersecurity",
            "source_url": self.cyber,
            "total_chunks": 28,
            "evidence": [{"text": "Network security."}],
            "truncated": True,
        }
        self.analytics_details = {
            "title": "Business Data Analytics",
            "source_url": self.analytics,
            "total_chunks": 20,
            "evidence": [{"text": "SQL and Power BI."}],
            "truncated": True,
        }

    def test_prepares_two_courses_without_inventing_facts(self):
        from unittest.mock import patch
        from app.tools.courses import compare_courses

        fake_client = object()

        with patch(
            "app.tools.courses.get_course_details",
            side_effect=[
                self.cyber_details,
                self.analytics_details,
            ],
        ) as retrieve:
            result = compare_courses(
                self.cyber,
                self.analytics,
                client=fake_client,
                max_chunks_per_course=3,
            )

        self.assertTrue(result["comparison_ready"])
        self.assertEqual(result["missing_courses"], [])
        self.assertEqual(
            [course["title"] for course in result["courses"]],
            ["Cybersecurity", "Business Data Analytics"],
        )
        self.assertEqual(retrieve.call_count, 2)

        for call in retrieve.call_args_list:
            self.assertIs(call.kwargs["client"], fake_client)
            self.assertEqual(call.kwargs["max_chunks"], 3)

    def test_missing_course_is_reported_not_fabricated(self):
        from unittest.mock import patch
        from app.tools.courses import compare_courses

        with patch(
            "app.tools.courses.get_course_details",
            side_effect=[self.cyber_details, None],
        ):
            result = compare_courses(
                self.cyber,
                self.analytics,
                client=object(),
            )

        self.assertFalse(result["comparison_ready"])
        self.assertEqual(len(result["courses"]), 1)
        self.assertEqual(
            result["missing_courses"],
            [self.analytics],
        )

    def test_identical_or_invalid_urls_are_rejected(self):
        from app.tools.courses import compare_courses

        with self.assertRaises(ValueError):
            compare_courses(self.cyber, self.cyber)

        with self.assertRaises(ValueError):
            compare_courses(
                self.cyber,
                "https://evil.example/courses/fake",
            )

    def test_invalid_chunk_limit_is_rejected(self):
        from app.tools.courses import compare_courses

        for value in (0, 11, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compare_courses(
                        self.cyber,
                        self.analytics,
                        max_chunks_per_course=value,
                    )

    def test_owned_qdrant_client_is_closed_on_failure(self):
        from unittest.mock import patch
        from app.tools.courses import compare_courses

        with (
            patch("app.tools.courses.QDRANT_PATH") as db_path,
            patch("qdrant_client.QdrantClient") as client_class,
            patch(
                "app.tools.courses.get_course_details",
                side_effect=RuntimeError("Database failure"),
            ),
        ):
            db_path.is_dir.return_value = True

            with self.assertRaises(RuntimeError):
                compare_courses(self.cyber, self.analytics)

            client_class.assert_called_once()
            client_class.return_value.close.assert_called_once()
