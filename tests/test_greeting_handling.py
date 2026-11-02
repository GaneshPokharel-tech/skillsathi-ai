"""Greeting-only messages should not require RAG or Gemini."""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.agent.service import ask_agent
from app.api.server import app


class TestGreetingHandling(unittest.TestCase):

    def test_english_greeting(self):
        with patch("app.agent.service.execute_question") as agent:
            result = ask_agent(" Hello!? ", language="en")

        self.assertIn("Hello", result["answer"])
        self.assertIn("SkillSathi", result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["retrieved_chunks"], 0)
        agent.assert_not_called()

    def test_nepali_and_romanized_greetings(self):
        with patch("app.agent.service.execute_question") as agent:
            for question in (
                "namaste", "namaskar", "नमस्ते",
                "नमस्कार!", "hi"
            ):
                with self.subTest(question=question):
                    result = ask_agent(question, language="ne")
                    self.assertIn("नमस्ते", result["answer"])
                    self.assertEqual(result["citations"], [])
                    self.assertEqual(result["retrieved_chunks"], 0)

        agent.assert_not_called()

    def test_course_question_still_uses_rag(self):
        question = "Hello, what courses do you offer?"

        with patch(
            "app.agent.service.execute_question",
            return_value={
                "tool": "knowledge_search",
                "data": {"evidence": []},
            },
        ) as agent:
            result = ask_agent(question, language="en")

        agent.assert_called_once_with(question, top_k=4)
        self.assertIn(
            "could not find enough information",
            result["answer"],
        )

    def test_safety_response_still_has_priority(self):
        with patch("app.agent.service.execute_question") as agent:
            result = ask_agent(
                "Hi, I want to die",
                language="en",
            )

        self.assertIn("safety", result["answer"].lower())
        agent.assert_not_called()

    def test_rest_api_handles_hello(self):
        with patch("app.agent.service.execute_question") as agent:
            response = TestClient(app).post(
                "/chat",
                json={
                    "question": "hello",
                    "language": "en",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("SkillSathi", response.json()["answer"])
        agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
