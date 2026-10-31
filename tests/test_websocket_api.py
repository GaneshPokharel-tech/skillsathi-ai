"""Regression tests for the WebSocket chat transport."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.server import app


client = TestClient(app)


class TestWebSocketChat(unittest.TestCase):

    def connect(self):
        return client.websocket_connect("/ws/chat")

    def test_answer_preserves_existing_response_contract(self):
        expected = {
            "answer": "Revit BIM course available.",
            "citations": [],
            "retrieved_chunks": 2,
        }

        with patch(
            "app.api.server.ask_agent",
            return_value=expected,
        ) as mocked:
            try:
                with self.connect() as ws:
                    ws.send_json({
                        "question": "Tell me about Revit",
                        "language": "en",
                        "top_k": 3,
                    })
                    received = ws.receive_json()
            except WebSocketDisconnect as exc:
                self.fail(
                    f"WebSocket connection rejected: {exc.code}"
                )

        self.assertEqual(
            received,
            {"type": "answer", "data": expected},
        )
        mocked.assert_called_once_with(
            "Tell me about Revit",
            language="en",
            top_k=3,
        )

    def test_connection_supports_multiple_questions(self):
        result = {
            "answer": "Course information.",
            "citations": [],
            "retrieved_chunks": 1,
        }

        with patch(
            "app.api.server.ask_agent",
            return_value=result,
        ) as mocked:
            with self.connect() as ws:
                for question in ("First question", "Second question"):
                    ws.send_json({"question": question})
                    self.assertEqual(
                        ws.receive_json(),
                        {"type": "answer", "data": result},
                    )

        self.assertEqual(mocked.call_count, 2)

    def test_invalid_question_returns_safe_error(self):
        with patch("app.api.server.ask_agent") as mocked:
            with self.connect() as ws:
                ws.send_json({"question": "   "})
                received = ws.receive_json()

        self.assertEqual(received, {
            "type": "error",
            "code": "invalid_request",
            "message": "Invalid chat request.",
        })
        mocked.assert_not_called()

    def test_backend_failure_does_not_leak_details(self):
        with patch(
            "app.api.server.ask_agent",
            side_effect=RuntimeError("SECRET_INTERNAL_ERROR"),
        ):
            with self.connect() as ws:
                ws.send_json({"question": "Hello"})
                received = ws.receive_json()

        self.assertEqual(received, {
            "type": "error",
            "code": "service_unavailable",
            "message": "Chat service temporarily unavailable.",
        })
        self.assertNotIn("SECRET_INTERNAL_ERROR", str(received))


if __name__ == "__main__":
    unittest.main()
