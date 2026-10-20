"""Unit tests for the SkillSathi HTTP API."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)


class TestChatAPI(unittest.TestCase):

    def test_health_endpoint(self):
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_returns_answer_and_citations(self):
        expected = {
            "answer": "Revit modeling sikainchha. [1]",
            "citations": [
                {
                    "marker": "[1]",
                    "section": "Curriculum",
                    "source_url": "https://skillshikshya.com/courses/autodesk-revit-bim-course",
                }
            ],
            "retrieved_chunks": 3,
        }

        with patch(
            "app.api.server.ask_skillsathi",
            return_value=expected,
        ) as mocked_chat:
            response = client.post(
                "/chat",
                json={
                    "question": "  What is Revit BIM?  ",
                    "top_k": 3,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mocked_chat.assert_called_once_with(
            "What is Revit BIM?",
            top_k=3,
            language="ne",
        )

    def test_invalid_requests_are_rejected(self):
        invalid_requests = [
            {"question": ""},
            {"question": "   "},
            {"question": "A" * 2001},
            {"question": "Hello", "top_k": 0},
            {"question": "Hello", "top_k": 11},
            {"question": "Hello", "top_k": True},
            {"question": "Hello", "unexpected": "value"},
        ]

        with patch("app.api.server.ask_skillsathi") as mocked_chat:
            for payload in invalid_requests:
                with self.subTest(payload=payload):
                    response = client.post(
                        "/chat",
                        json=payload,
                    )
                    self.assertEqual(
                        response.status_code,
                        422,
                        response.text,
                    )

        mocked_chat.assert_not_called()

    def test_backend_failure_is_sanitized(self):
        with (
            patch(
                "app.api.server.ask_skillsathi",
                side_effect=RuntimeError(
                    "SECRET_INTERNAL_ERROR"
                ),
            ),
            patch("app.api.server.logger.error"),
        ):
            response = client.post(
                "/chat",
                json={"question": "Tell me about Python"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(
            "SECRET_INTERNAL_ERROR",
            response.text,
        )
        self.assertEqual(
            response.json()["detail"],
            "Chat service temporarily unavailable.",
        )

    def test_default_top_k_and_empty_evidence(self):
        expected = {
            "answer": "Paryapta jaankari bhetiyena.",
            "citations": [],
            "retrieved_chunks": 0,
        }

        with patch(
            "app.api.server.ask_skillsathi",
            return_value=expected,
        ) as mocked_chat:
            response = client.post(
                "/chat",
                json={"question": "Unknown course?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        mocked_chat.assert_called_once_with(
            "Unknown course?",
            top_k=4,
            language="ne",
        )


if __name__ == "__main__":
    unittest.main()


class TestChatPage(unittest.TestCase):

    def test_chat_page_is_served_at_root(self):
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("SkillSathi AI", response.text)
        self.assertIn('fetch("/chat"', response.text)

    def test_swagger_docs_remain_available(self):
        response = client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("swagger-ui", response.text)


class TestChatLanguageAPI(unittest.TestCase):

    def test_english_selection_reaches_chatbot(self):
        expected = {
            "answer": "Python basics are covered. [1]",
            "citations": [],
            "retrieved_chunks": 1,
        }

        with patch(
            "app.api.server.ask_skillsathi",
            return_value=expected,
        ) as mocked:
            response = client.post(
                "/chat",
                json={
                    "question": "What is taught?",
                    "language": "en",
                },
            )

        self.assertEqual(response.status_code, 200)
        mocked.assert_called_once_with(
            "What is taught?",
            top_k=4,
            language="en",
        )

    def test_unsupported_language_returns_422(self):
        response = client.post(
            "/chat",
            json={"question": "Hello", "language": "fr"},
        )
        self.assertEqual(response.status_code, 422)
