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



class TestCourseFollowupHTTP(unittest.TestCase):

    def test_followup_context_reaches_agent(self):
        first = (
            "https://skillshikshya.com/courses/"
            "advanced-ocr-tax-vat-training-in-nepal"
        )
        second = (
            "https://skillshikshya.com/courses/"
            "accounting-to-tax-closure-training"
        )

        options = [first, second]

        with patch(
            "app.api.server.ask_agent",
            return_value={
                "answer": "Selected course information. [1]",
                "citations": [],
                "retrieved_chunks": 2,
            },
        ) as agent:
            response = client.post(
                "/chat",
                json={
                    "question": "second one",
                    "language": "en",
                    "pending_course_urls": options,
                },
            )

        self.assertEqual(response.status_code, 200)
        agent.assert_called_once_with(
            "second one",
            top_k=4,
            language="en",
            pending_course_urls=options,
        )

    def test_too_many_pending_options_returns_422(self):
        response = client.post(
            "/chat",
            json={
                "question": "second one",
                "language": "en",
                "pending_course_urls": ["x"] * 6,
            },
        )

        self.assertEqual(response.status_code, 422)



class TestCourseFollowupFrontend(unittest.TestCase):

    def test_browser_chat_includes_followup_context(self):
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("let pendingCourseUrls = null", response.text)
        self.assertIn(
            "pending_course_urls: pendingCourseUrls",
            response.text,
        )
        self.assertIn(
            "const options = data.pending_course_urls",
            response.text,
        )



class TestAnswerMarkdownRenderer(unittest.TestCase):

    def test_renderer_handles_headings_lists_and_untrusted_html(self):
        import shutil
        import subprocess
        from pathlib import Path

        if not shutil.which("node"):
            self.skipTest("Node.js is required for renderer test.")

        html = Path("app/api/chat.html").read_text(
            encoding="utf-8"
        )

        renderer = (
            "function addInline("
            + html.split("    function addInline(", 1)[1]
                  .split("    function appendMessage(", 1)[0]
        )

        javascript = r"""
const assert = require("node:assert/strict");

class FakeNode {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
  }

  appendChild(child) {
    this.children.push(child);
  }

  set textContent(value) {
    this.children = [{ tagName: "#TEXT", value }];
  }
}

const document = {
  createElement: tag => new FakeNode(tag),
  createTextNode: value => ({ tagName: "#TEXT", value })
};

""" + renderer + r"""

const root = new FakeNode("div");

formatAnswer(
  root,
  "### **Curriculum**\n" +
  "1. Module One\n" +
  "2. Module Two\n" +
  "- Learn <img src=x onerror=alert(1)>\n" +
  "---\n" +
  "Course summary."
);

assert.deepEqual(
  root.children.map(node => node.tagName),
  ["H3", "OL", "UL", "HR", "P"]
);

assert.equal(root.children[0].children[0].tagName, "STRONG");
assert.equal(root.children[1].children.length, 2);

const bullet = root.children[2].children[0];

assert.equal(
  bullet.children[0].value,
  "Learn <img src=x onerror=alert(1)>"
);

// AI-generated HTML must remain plain text.
assert.ok(
  !root.children.some(node => node.tagName === "IMG")
);

console.log("Markdown rendering and HTML safety: OK");
"""

        result = subprocess.run(
            ["node", "-"],
            input=javascript,
            text=True,
            capture_output=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            result.stderr,
        )
        self.assertIn(
            "Markdown rendering and HTML safety: OK",
            result.stdout,
        )



class TestBrandedChatLayout(unittest.TestCase):

    def test_sidebar_and_language_choices_are_present(self):
        response = client.get("/")

        self.assertEqual(response.status_code, 200)

        page = response.text

        self.assertIn('class="brand-sidebar"', page)
        self.assertIn(
            'href="https://skillshikshya.com/"',
            page,
        )
        self.assertIn("Choose your language", page)
        self.assertIn('data-language="ne"', page)
        self.assertIn('data-language="en"', page)
        self.assertIn('class="choice-arrow"', page)

    def test_existing_chat_controls_are_preserved(self):
        response = client.get("/")

        page = response.text

        self.assertIn('id="chat-view"', page)
        self.assertIn('id="conversation"', page)
        self.assertIn('id="change-language"', page)
        self.assertIn('id="chat-form"', page)
        self.assertIn('pending_course_urls: pendingCourseUrls', page)



class TestDeveloperCredit(unittest.TestCase):

    def test_sidebar_credit_and_language_icon_style(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Made by Ganesh Pokharel", response.text)
        self.assertIn('class="sidebar-credit"', response.text)
        self.assertIn("white-space: nowrap;", response.text)



class TestBrandIcon(unittest.TestCase):

    def test_official_icon_is_served_locally(self):
        response = client.get("/brand-icon")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("image/png")
        )
        self.assertTrue(
            response.content.startswith(bytes.fromhex("89504e470d0a1a0a"))
        )

    def test_sidebar_uses_local_brand_icon(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('class="sidebar-brand-logo"', response.text)
        self.assertIn('src="/brand-icon"', response.text)
        self.assertIn("SkillShikshya", response.text)
