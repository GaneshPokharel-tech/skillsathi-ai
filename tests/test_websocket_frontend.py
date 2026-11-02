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

    def test_whatsapp_click_to_chat_opens_with_encoded_message_on_click(self):
        html = (
            Path(__file__).resolve().parents[1]
            / "app/api/chat.html"
        ).read_text(encoding="utf-8")

        self.assertIn(
            '<button id="whatsapp-contact" class="whatsapp-contact" type="button">'
            'WhatsApp मा सम्पर्क गर्नुहोस्</button>',
            html,
        )
        handler_start = html.index(
            'whatsappButton.addEventListener("click", () => {'
        )
        handler_end = html.index("\n    });", handler_start)
        click_handler = html[handler_start:handler_end]

        self.assertIn(
            '"Namaste SkillShikshya! I have an inquiry about your courses '
            'and admission. Please help me with the details."',
            click_handler,
        )
        self.assertIn(
            '"https://wa.me/9779868730959?text=" + encodeURIComponent(message)',
            click_handler,
        )
        self.assertIn('window.open(whatsappUrl, "_blank"', click_handler)
        self.assertNotIn("window.open(", html[:handler_start])


if __name__ == "__main__":
    unittest.main()
