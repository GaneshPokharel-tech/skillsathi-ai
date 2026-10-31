import unittest
from unittest.mock import patch

from app.agent.router import decide_route
from app.agent.executor import execute_question
from app.agent.service import ask_agent


CATALOG = [
    {
        "title": "Professional Accounting Training",
        "source_url": "https://skillshikshya.com/courses/professional-accounting-training-in-nepal",
    },
    {
        "title": "Accounting to Tax Closure Training",
        "source_url": "https://skillshikshya.com/courses/accounting-to-tax-closure-training",
    },
    {
        "title": "DevOps Job Ready Program",
        "source_url": "https://skillshikshya.com/courses/devops-job-ready-program-in-nepal",
    },
]


class WebsiteIntentRoutingTests(unittest.TestCase):
    def test_accounting_career_path_routes_to_knowledge(self):
        plan = decide_route(
            "Accounting career path ko barema bhannus.",
            CATALOG,
        )
        self.assertEqual(plan["tool"], "knowledge_search")

    def test_institute_and_counselling_questions(self):
        for question in (
            "SkillShikshya ko mission k ho?",
            "Online counselling kasari line?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    decide_route(question, CATALOG)["tool"],
                    "knowledge_search",
                )

    def test_course_routing_still_works(self):
        self.assertEqual(
            decide_route("List available courses", CATALOG)["tool"],
            "list_courses",
        )
        self.assertEqual(
            decide_route(
                "DevOps Job Ready Program ko syllabus?",
                CATALOG,
            )["tool"],
            "get_course_details",
        )
        self.assertEqual(
            decide_route(
                "Which accounting course should I choose?",
                CATALOG,
            )["tool"],
            "clarify",
        )

    def test_multi_question_retrieves_each_topic(self):
        question = (
            "1. Accounting career path ko barema bhannus.\n"
            "2. SkillShikshya ko mission k ho?\n"
            "3. Online counselling kasari line?"
        )

        def fake_search(query, *, limit):
            return [{
                "source_url": "https://skillshikshya.com/" + str(len(query)),
                "section": query,
                "text": "Verified content: " + query,
            }]

        with patch(
            "app.agent.executor.list_courses",
            return_value=CATALOG,
        ), patch(
            "app.agent.executor.retrieve_evidence",
            side_effect=fake_search,
        ) as search:
            result = execute_question(question, top_k=4)

        self.assertEqual(result["tool"], "knowledge_search")
        self.assertEqual(search.call_count, 3)
        self.assertEqual(len(result["data"]["evidence"]), 3)

    def test_multi_question_answer_prompt_covers_all_topics(self):
        question = (
            "1. Accounting career path ko barema bhannus.\n"
            "2. SkillShikshya ko mission k ho?\n"
            "3. Online counselling kasari line?"
        )
        evidence = [{
            "source_url": "https://skillshikshya.com/about",
            "section": "Mission",
            "text": "Website evidence",
        }]
        plan = {
            "tool": "knowledge_search",
            "data": {"evidence": evidence},
        }

        with patch(
            "app.agent.service.execute_question",
            return_value=plan,
        ), patch(
            "app.agent.service.answer_with_evidence",
            return_value={"answer": "Test answer", "citations": []},
        ) as answer:
            ask_agent(question, client=object())

        prompt = answer.call_args.args[0]
        self.assertIn("Accounting career path", prompt)
        self.assertIn("mission", prompt)
        self.assertIn("counselling", prompt)
        self.assertIn("separately", prompt)



    def test_three_inline_numbered_questions(self):
        from app.agent.router import split_user_questions

        question = (
            "1. Accounting career path ko barema bhannus. "
            "2. SkillShikshya ko mission k ho? "
            "3. Online counselling kasari line?"
        )

        parts = split_user_questions(question)

        self.assertEqual(len(parts), 3)
        self.assertIn("Accounting career path", parts[0])
        self.assertIn("mission", parts[1])
        self.assertIn("counselling", parts[2])


    def test_mixed_sentence_and_questions_split(self):
        from app.agent.router import split_user_questions

        question = (
            "Accounting career path ko barema bhannus. "
            "SkillShikshya ko mission k ho? "
            "Online counselling kasari line?"
        )

        parts = split_user_questions(question)

        self.assertEqual(len(parts), 3)
        self.assertIn("Accounting career path", parts[0])
        self.assertIn("mission", parts[1])
        self.assertIn("counselling", parts[2])

if __name__ == "__main__":
    unittest.main()
