"""Tests for the SkillSathi Streamlit demonstration."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from app.demo.streamlit_app import (
    ChatUnavailable,
    call_chat,
    trusted_citations,
)


class TestStreamlitDemo(unittest.TestCase):

    def test_chat_uses_existing_rest_api(self):
        expected = {
            "answer": "Accounting courses available.",
            "citations": [],
            "retrieved_chunks": 2,
        }

        response = io.BytesIO(json.dumps(expected).encode())

        with patch(
            "app.demo.streamlit_app.urlopen",
            return_value=response,
        ) as mocked:
            result = call_chat(
                "  Accounting courses?  ",
                language="en",
            )

        self.assertEqual(result, expected)

        request = mocked.call_args.args[0]
        payload = json.loads(request.data)

        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(payload["question"], "Accounting courses?")
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["top_k"], 4)

    def test_course_followup_context_is_preserved(self):
        response = io.BytesIO(json.dumps({
            "answer": "Course details.",
            "citations": [],
            "retrieved_chunks": 1,
        }).encode())

        urls = [
            "https://skillshikshya.com/courses/a",
            "https://skillshikshya.com/courses/b",
        ]

        with patch(
            "app.demo.streamlit_app.urlopen",
            return_value=response,
        ) as mocked:
            call_chat(
                "Compare these",
                pending_course_urls=urls,
            )

        request = mocked.call_args.args[0]
        payload = json.loads(request.data)

        self.assertEqual(
            payload["pending_course_urls"],
            urls,
        )

    def test_backend_failure_has_safe_error(self):
        with patch(
            "app.demo.streamlit_app.urlopen",
            side_effect=URLError("SECRET_CONNECTION_DETAIL"),
        ):
            with self.assertRaises(ChatUnavailable) as ctx:
                call_chat("Hello")

        self.assertNotIn(
            "SECRET_CONNECTION_DETAIL",
            str(ctx.exception),
        )

    def test_only_trusted_citation_links_are_displayed(self):
        citations = [
            {
                "marker": "[1]",
                "section": "Courses",
                "source_url": "https://skillshikshya.com/courses",
            },
            {
                "marker": "[2]",
                "section": "External",
                "source_url": "https://skillshikshya.com.evil.test",
            },
            {
                "marker": "[3]",
                "section": "Unsafe",
                "source_url": "javascript:alert(1)",
            },
        ]

        result = trusted_citations(citations)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["marker"], "[1]")

    def test_malformed_backend_response_is_rejected(self):
        response = io.BytesIO(
            json.dumps({"unexpected": "data"}).encode()
        )

        with patch(
            "app.demo.streamlit_app.urlopen",
            return_value=response,
        ):
            with self.assertRaises(ChatUnavailable):
                call_chat("Hello")


if __name__ == "__main__":
    unittest.main()
