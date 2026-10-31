
import unittest
from app.agent.router import split_user_questions
from app.rag.engine import _preferred_source_url


class FourQuestionRoutingTests(unittest.TestCase):

    def test_four_questions_with_lowercase_followup(self):
        question = (
            "Accounting career path ko barema bhannus. "
            "SkillShikshya ko mission k ho? "
            "skillshikshya nepal ko prime institute ho? "
            "Online counselling kasari line?"
        )

        expected = [
            "Accounting career path ko barema bhannus.",
            "SkillShikshya ko mission k ho?",
            "skillshikshya nepal ko prime institute ho?",
            "Online counselling kasari line?",
        ]

        self.assertEqual(
            split_user_questions(question),
            expected,
        )

    def test_lowercase_followup_questions(self):
        question = (
            "Mission k ho? "
            "online counselling kasari line? "
            "accounting career path k ho?"
        )

        self.assertEqual(
            len(split_user_questions(question)),
            3,
        )

    def test_existing_three_question_case(self):
        question = (
            "Accounting career path ko barema bhannus. "
            "SkillShikshya ko mission k ho? "
            "Online counselling kasari line?"
        )

        self.assertEqual(
            len(split_user_questions(question)),
            3,
        )

    def test_single_question_remains_intact(self):
        question = "What does U.S. tax training cover?"

        self.assertEqual(
            split_user_questions(question),
            [question],
        )

    def test_prime_institute_maps_to_about(self):
        url = "https://skillshikshya.com/about"

        self.assertEqual(
            _preferred_source_url(
                "skillshikshya nepal ko prime institute ho?"
            ),
            url,
        )

        self.assertEqual(
            _preferred_source_url(
                "Is SkillShikshya Nepal's number one IT institute?"
            ),
            url,
        )

    def test_general_course_query_not_forced(self):
        self.assertIsNone(
            _preferred_source_url(
                "Which is the best DevOps course?"
            )
        )


if __name__ == "__main__":
    unittest.main()
