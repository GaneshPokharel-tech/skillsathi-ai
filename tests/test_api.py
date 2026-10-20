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
        )


if __name__ == "__main__":
    unittest.main()
