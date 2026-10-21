"""Tests for SkillSathi tool selection."""

import unittest

from app.agent.router import decide_route


class TestAgentRouting(unittest.TestCase):

    def setUp(self):
        base = "https://skillshikshya.com/courses/"
        self.catalog = [
            {
                "title": "Cybersecurity Job Ready Program",
                "source_url": base + "cybersecurity",
            },
            {
                "title": "Business Data Analytics Job Ready Program",
                "source_url": base + "business-data-analytics",
            },
            {
                "title": "Autodesk Revit (BIM) Professional Course",
                "source_url": base + "autodesk-revit-bim-course",
            },
            {
                "title": "Advanced OCR & Tax VAT Training",
                "source_url": base + "advanced-tax",
            },
            {
                "title": "Complete Accounting to Tax Closure Training",
                "source_url": base + "accounting-tax",
            },
        ]

    def route(self, question):
        return decide_route(question, self.catalog)

    def test_english_and_nepali_course_discovery(self):
        for question in (
            "List available courses",
            "के के कोर्सहरू उपलब्ध छन्?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    self.route(question)["tool"],
                    "list_courses",
                )

    def test_course_details(self):
        plan = self.route("Tell me about the Revit BIM course")

        self.assertEqual(plan["tool"], "get_course_details")
        self.assertIn(
            "autodesk-revit-bim-course",
            plan["arguments"]["course_url"],
        )

    def test_comparison_respects_question_order(self):
        plan = self.route(
            "Compare data analytics vs cybersecurity"
        )

        self.assertEqual(plan["tool"], "compare_courses")
        self.assertIn(
            "business-data-analytics",
            plan["arguments"]["first_url"],
        )
        self.assertIn(
            "cybersecurity",
            plan["arguments"]["second_url"],
        )

    def test_incomplete_comparison_requires_clarification(self):
        plan = self.route(
            "Compare Cybersecurity with an unknown course"
        )

        self.assertEqual(plan["tool"], "clarify")

    def test_ambiguous_course_reference_requires_clarification(self):
        plan = self.route("Tell me about the tax course")

        self.assertEqual(plan["tool"], "clarify")
        self.assertEqual(
            len(plan["arguments"]["matched_courses"]),
            2,
        )

    def test_general_question_uses_knowledge_search(self):
        plan = self.route("How can students enroll?")

        self.assertEqual(plan["tool"], "knowledge_search")

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.route("  ")

        with self.assertRaises(ValueError):
            decide_route("Hello", None)


class TestAgentExecution(unittest.TestCase):

    def setUp(self):
        self.cyber_url = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )
        self.analytics_url = (
            "https://skillshikshya.com/courses/"
            "business-data-analytics-job-ready-program"
        )
        self.tax_url = (
            "https://skillshikshya.com/courses/"
            "accounting-to-tax-closure-training"
        )
        self.ocr_url = (
            "https://skillshikshya.com/courses/"
            "advanced-ocr-tax-vat-training-in-nepal"
        )
        self.catalog = [
            {
                "title": "Cybersecurity Job Ready Program",
                "source_url": self.cyber_url,
            },
            {
                "title": "Business Data Analytics Job Ready Program",
                "source_url": self.analytics_url,
            },
            {
                "title": "Complete Accounting to Tax Closure Training",
                "source_url": self.tax_url,
            },
            {
                "title": "Advanced OCR & Tax VAT Training",
                "source_url": self.ocr_url,
            },
        ]

    def test_discovery_returns_only_indexed_catalog(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ) as catalog_tool,
            patch(
                "app.agent.executor.get_course_details",
            ) as details_tool,
        ):
            result = execute_question("List available courses")

        self.assertEqual(result["tool"], "list_courses")
        self.assertEqual(result["data"]["courses"], self.catalog)
        self.assertEqual(
            result["data"]["scope"],
            "indexed_courses_only",
        )
        catalog_tool.assert_called_once_with()
        details_tool.assert_not_called()

    def test_details_tool_is_executed(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        details = {
            "title": "Cybersecurity Job Ready Program",
            "source_url": self.cyber_url,
            "evidence": [{"text": "Network security."}],
        }

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ),
            patch(
                "app.agent.executor.get_course_details",
                return_value=details,
            ) as details_tool,
        ):
            result = execute_question(
                "Tell me about the Cybersecurity course"
            )

        self.assertEqual(result["tool"], "get_course_details")
        self.assertTrue(result["data"]["found"])
        self.assertEqual(result["data"]["course"], details)
        details_tool.assert_called_once_with(
            self.cyber_url,
            max_chunks=8,
        )

    def test_comparison_executes_in_question_order(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        comparison = {
            "comparison_ready": True,
            "courses": [],
            "missing_courses": [],
        }

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ),
            patch(
                "app.agent.executor.compare_courses",
                return_value=comparison,
            ) as compare_tool,
        ):
            result = execute_question(
                "Compare Data Analytics vs Cybersecurity"
            )

        self.assertEqual(result["tool"], "compare_courses")
        self.assertTrue(result["data"]["comparison_ready"])
        compare_tool.assert_called_once_with(
            self.analytics_url,
            self.cyber_url,
            max_chunks_per_course=4,
        )

    def test_ambiguity_returns_clarification(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ),
            patch(
                "app.agent.executor.get_course_details",
            ) as details_tool,
            patch(
                "app.agent.executor.compare_courses",
            ) as compare_tool,
            patch(
                "app.agent.executor.retrieve_evidence",
            ) as search_tool,
        ):
            result = execute_question("Tell me about the tax course")

        self.assertEqual(result["tool"], "clarify")
        self.assertEqual(
            len(result["data"]["matched_courses"]),
            2,
        )
        details_tool.assert_not_called()
        compare_tool.assert_not_called()
        search_tool.assert_not_called()

    def test_general_question_uses_knowledge_search(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        evidence = [{
            "source_url": "https://skillshikshya.com/about",
            "section": "Enrollment",
            "text": "Enrollment information.",
        }]

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ),
            patch(
                "app.agent.executor.retrieve_evidence",
                return_value=evidence,
            ) as search_tool,
        ):
            result = execute_question(
                "How can students enroll?",
                top_k=3,
            )

        self.assertEqual(result["tool"], "knowledge_search")
        self.assertEqual(result["data"]["evidence"], evidence)
        self.assertEqual(result["data"]["retrieved_chunks"], 1)
        search_tool.assert_called_once_with(
            "How can students enroll?",
            limit=3,
        )

    def test_missing_course_details_are_explicit(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        with (
            patch(
                "app.agent.executor.list_courses",
                return_value=self.catalog,
            ),
            patch(
                "app.agent.executor.get_course_details",
                return_value=None,
            ),
        ):
            result = execute_question(
                "Tell me about the Cybersecurity course"
            )

        self.assertFalse(result["data"]["found"])
        self.assertIsNone(result["data"]["course"])

    def test_invalid_inputs_do_not_call_tools(self):
        from unittest.mock import patch
        from app.agent.executor import execute_question

        with patch(
            "app.agent.executor.list_courses",
        ) as catalog_tool:
            for question in ("", "   ", "a" * 2001):
                with self.subTest(question_length=len(question)):
                    with self.assertRaises(ValueError):
                        execute_question(question)

            for top_k in (0, 11, True, 2.5):
                with self.subTest(top_k=top_k):
                    with self.assertRaises(ValueError):
                        execute_question("Hello", top_k=top_k)

        catalog_tool.assert_not_called()


class TestAgentAnswerService(unittest.TestCase):

    def setUp(self):
        self.url = (
            "https://skillshikshya.com/courses/"
            "autodesk-revit-bim-course"
        )
        self.evidence = [{
            "source_url": self.url,
            "section": "Curriculum",
            "text": "BIM fundamentals and architectural modeling.",
        }]

    def test_course_listing_skips_gemini(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        plan = {
            "tool": "list_courses",
            "data": {
                "courses": [{
                    "title": "Autodesk Revit",
                    "source_url": self.url,
                }],
            },
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
            ) as generate,
        ):
            result = ask_agent("List courses", language="en")

        self.assertIn("Autodesk Revit", result["answer"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["retrieved_chunks"], 0)
        generate.assert_not_called()

    def test_clarification_skips_gemini(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        plan = {
            "tool": "clarify",
            "data": {
                "matched_courses": [{
                    "title": "Autodesk Revit",
                    "source_url": self.url,
                }],
            },
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
            ) as generate,
        ):
            result = ask_agent("Which one?", language="ne")

        self.assertIn("कुन कोर्स", result["answer"])
        self.assertEqual(len(result["citations"]), 1)
        generate.assert_not_called()

    def test_details_pass_exact_course_evidence(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        plan = {
            "tool": "get_course_details",
            "data": {
                "found": True,
                "course": {
                    "evidence": self.evidence,
                    "truncated": False,
                },
            },
        }
        expected = {
            "answer": "Revit teaches BIM. [1]",
            "citations": [{
                "marker": "[1]",
                "section": "Curriculum",
                "source_url": self.url,
            }],
        }
        fake_client = object()

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
                return_value=expected,
            ) as generate,
        ):
            result = ask_agent(
                "Tell me about Revit",
                language="en",
                client=fake_client,
            )

        self.assertEqual(result["retrieved_chunks"], 1)
        self.assertEqual(result["citations"], expected["citations"])
        self.assertEqual(
            generate.call_args.args[1],
            self.evidence,
        )
        self.assertIs(
            generate.call_args.kwargs["client"],
            fake_client,
        )
        self.assertEqual(
            generate.call_args.kwargs["language"],
            "en",
        )

    def test_comparison_combines_both_sources(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        other_url = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )
        other_evidence = [{
            "source_url": other_url,
            "section": "Curriculum",
            "text": "Network security and ethical hacking.",
        }]

        plan = {
            "tool": "compare_courses",
            "data": {
                "comparison_ready": True,
                "courses": [
                    {
                        "evidence": self.evidence,
                        "truncated": True,
                    },
                    {
                        "evidence": other_evidence,
                        "truncated": True,
                    },
                ],
            },
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
                return_value={
                    "answer": "Both courses teach different skills. [1, 2]",
                    "citations": [],
                },
            ) as generate,
        ):
            result = ask_agent(
                "Compare Revit and Cybersecurity",
                language="en",
                client=object(),
            )

        evidence = generate.call_args.args[1]
        self.assertEqual(
            [item["source_url"] for item in evidence],
            [self.url, other_url],
        )
        self.assertEqual(result["retrieved_chunks"], 2)
        self.assertIn("not a complete syllabus", result["answer"])

    def test_missing_comparison_skips_generation(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        plan = {
            "tool": "compare_courses",
            "data": {
                "comparison_ready": False,
                "courses": [],
                "missing_courses": [self.url],
            },
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
            ) as generate,
        ):
            result = ask_agent(
                "Compare courses",
                language="en",
            )

        self.assertEqual(result["citations"], [])
        generate.assert_not_called()

    def test_knowledge_search_uses_retrieved_evidence(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        plan = {
            "tool": "knowledge_search",
            "data": {"evidence": self.evidence},
        }

        with (
            patch(
                "app.agent.service.execute_question",
                return_value=plan,
            ),
            patch(
                "app.agent.service.answer_with_evidence",
                return_value={
                    "answer": "Information found. [1]",
                    "citations": [],
                },
            ) as generate,
        ):
            result = ask_agent(
                "How can students enroll?",
                client=object(),
            )

        self.assertEqual(result["retrieved_chunks"], 1)
        generate.assert_called_once()

    def test_invalid_language_rejected_before_tools(self):
        from unittest.mock import patch
        from app.agent.service import ask_agent

        with patch(
            "app.agent.service.execute_question",
        ) as executor:
            with self.assertRaises(ValueError):
                ask_agent("Hello", language="fr")

        executor.assert_not_called()
