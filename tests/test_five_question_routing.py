
import unittest
from app.agent.router import split_user_questions


class FiveQuestionRoutingTests(unittest.TestCase):

    def test_exact_live_screenshot(self):
        question = (
            "Accounting career path ko barema bhannus. "
            "SkillShikshya ko mission k ho? "
            "skillshikshya nepal ko prime institute ho? "
            "Online counselling kasari line?"
            "nepal ko prime minister ko ho?"
        )

        expected = [
            "Accounting career path ko barema bhannus.",
            "SkillShikshya ko mission k ho?",
            "skillshikshya nepal ko prime institute ho?",
            "Online counselling kasari line?",
            "nepal ko prime minister ko ho?",
        ]

        self.assertEqual(
            split_user_questions(question), expected
        )

    def test_lowercase_without_space(self):
        self.assertEqual(
            split_user_questions(
                "Mission k ho?online counselling kasari line?"
            ),
            [
                "Mission k ho?",
                "online counselling kasari line?",
            ],
        )


    def test_url_query_parameter_is_preserved(self):
        question = (
            "What is https://example.org/search?q=ai?"
        )
        self.assertEqual(
            split_user_questions(question), [question]
        )


    def test_five_numbered_questions(self):
        question = (
            "1. First question? "
            "2. Second question? "
            "3. Third question? "
            "4. Fourth question? "
            "5. Fifth question?"
        )
        self.assertEqual(
            len(split_user_questions(question)), 5
        )


if __name__ == "__main__":
    unittest.main()
