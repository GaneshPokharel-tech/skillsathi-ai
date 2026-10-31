"""Guard the browser WebSocket transport and HTTP fallback."""

import unittest
from pathlib import Path


class TestWebSocketFrontend(unittest.TestCase):

    def test_chat_frontend_preserves_websocket_and_http_fallback(self):
        html = (
            Path(__file__).resolve().parents[1]
            / "app/api/chat.html"
        ).read_text(encoding="utf-8")

        self.assertIn('new WebSocket(', html)
        self.assertIn('/ws/chat', html)
        self.assertIn('fetch("/chat"', html)
        self.assertIn('async function requestChat(', html)
        self.assertIn('await requestChat({', html)
        self.assertIn('pending_course_urls', html)
        self.assertIn('AbortError', html)


if __name__ == "__main__":
    unittest.main()
