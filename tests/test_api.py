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
            "app.api.server.ask_agent",
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

        with patch("app.api.server.ask_agent") as mocked_chat:
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
                "app.api.server.ask_agent",
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
            "app.api.server.ask_agent",
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
            "app.api.server.ask_agent",
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



class TestAgentHTTPIntegration(unittest.TestCase):
    """Verify HTTP requests reach the agent service."""

    def test_course_discovery_without_gemini(self):
        from unittest.mock import patch

        url = (
            "https://skillshikshya.com/courses/"
            "autodesk-revit-bim-course"
        )

        plan = {
            "tool": "list_courses",
            "data": {
                "courses": [{
                    "title": "Autodesk Revit BIM",
                    "source_url": url,
                }],
            },
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ) as executor,
            patch(
                "app.agent.service.answer_with_evidence",
            ) as generate,
        ):
            response = client.post(
                "/chat",
                json={
                    "question": "List available courses",
                    "language": "en",
                },
            )

        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertIn("Autodesk Revit BIM", body["answer"])
        self.assertEqual(body["citations"][0]["source_url"], url)
        self.assertEqual(body["retrieved_chunks"], 0)

        executor.assert_called_once_with(
            "List available courses",
            top_k=4,
        )
        generate.assert_not_called()

    def test_course_comparison_uses_agent_evidence(self):
        import os
        from unittest.mock import patch

        analytics = (
            "https://skillshikshya.com/courses/"
            "business-data-analytics-job-ready-program"
        )
        cyber = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )

        evidence = [
            {
                "source_url": analytics,
                "section": "Curriculum",
                "text": "SQL and Power BI.",
            },
            {
                "source_url": cyber,
                "section": "Curriculum",
                "text": "Network security and ethical hacking.",
            },
        ]

        plan = {
            "tool": "compare_courses",
            "data": {
                "comparison_ready": True,
                "courses": [
                    {
                        "evidence": [evidence[0]],
                        "truncated": False,
                    },
                    {
                        "evidence": [evidence[1]],
                        "truncated": False,
                    },
                ],
            },
        }

        generated = {
            "answer": "Analytics uses SQL; cybersecurity teaches security. [1, 2]",
            "citations": [
                {
                    "marker": "[1]",
                    "section": "Curriculum",
                    "source_url": analytics,
                },
                {
                    "marker": "[2]",
                    "section": "Curriculum",
                    "source_url": cyber,
                },
            ],
        }

        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "mock-test-key",
                    "GEMINI_MODEL": "gemini-3.5-flash-lite",
                },
            ),
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
                return_value=generated,
            ) as generate,
            patch("google.genai.Client") as gemini_client,
        ):
            response = client.post(
                "/chat",
                json={
                    "question": "Compare Data Analytics vs Cybersecurity",
                    "language": "en",
                },
            )

        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["retrieved_chunks"], 2)
        self.assertEqual(len(body["citations"]), 2)
        self.assertEqual(
            {item["source_url"] for item in body["citations"]},
            {analytics, cyber},
        )

        self.assertEqual(
            generate.call_args.args[1],
            evidence,
        )
        self.assertEqual(
            generate.call_args.kwargs["language"],
            "en",
        )
        gemini_client.return_value.close.assert_called_once()
