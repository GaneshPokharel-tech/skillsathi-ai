
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.rag.engine import (
    _preferred_source_url,
    answer_with_evidence,
    search_chunks,
)

BASE = "https://skillshikshya.com"


class AuthoritativeSourceTests(unittest.TestCase):

    def test_question_source_mapping(self):
        cases = {
            "Accounting career path ko barema bhannus.":
                BASE + "/blogs/accounting-career-path",
            "SkillShikshya ko mission k ho?":
                BASE + "/about",
            "Online counselling kasari line?":
                BASE + "/online-counselling",
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(
                    _preferred_source_url(question),
                    expected,
                )

    def test_general_course_question_is_not_forced(self):
        self.assertIsNone(
            _preferred_source_url(
                "DevOps Job Ready Program ko syllabus?"
            )
        )

    def test_filtered_qdrant_search(self):
        db = Mock()
        db.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.9,
                    payload={
                        "source_url": BASE + "/about",
                        "section": "Mission",
                        "text": "Industry relevant IT skills",
                    },
                )
            ]
        )

        result = search_chunks(
            db,
            collection_name="test_collection",
            query_vector=[0.1, 0.2],
            limit=3,
            source_url=BASE + "/about",
        )

        self.assertEqual(len(result), 1)

        condition = db.query_points.call_args.kwargs[
            "query_filter"
        ].must[0]

        self.assertEqual(condition.key, "source_url")
        self.assertEqual(
            condition.match.value,
            BASE + "/about",
        )

    def test_same_page_has_one_citation(self):
        evidence = [
            {
                "source_url": BASE + "/about",
                "section": "Mission",
                "text": "Practical IT skills",
            },
            {
                "source_url": BASE + "/about",
                "section": "Vision",
                "text": "Job-ready professionals",
            },
            {
                "source_url": BASE + "/online-counselling",
                "section": "How it works",
                "text": "Book an online slot",
            },
        ]

        client = Mock()
        client.interactions.create.return_value = (
            SimpleNamespace(
                output_text="Mission [1]. Counselling [2]."
            )
        )

        result = answer_with_evidence(
            "Mission and counselling?",
            evidence,
            client=client,
            language="en",
        )

        self.assertEqual(
            len(result["citations"]), 2
        )

        self.assertEqual(
            [
                item["source_url"]
                for item in result["citations"]
            ],
            [
                BASE + "/about",
                BASE + "/online-counselling",
            ],
        )

        prompt = (
            client.interactions.create
            .call_args.kwargs["input"]
        )

        separator = "WEBSITE EVIDENCE (JSON DATA):"
        self.assertIn(separator, prompt)

        sources = json.loads(
            prompt.split(separator, 1)[1].strip()
        )

        self.assertEqual(len(sources), 2)

        self.assertIn(
            "Practical IT skills",
            sources[0]["text"],
        )

        self.assertIn(
            "Job-ready professionals",
            sources[0]["text"],
        )


if __name__ == "__main__":
    unittest.main()
