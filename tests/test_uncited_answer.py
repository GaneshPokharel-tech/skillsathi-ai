"""Regression tests for uncited Gemini responses."""

import unittest
from unittest.mock import Mock, patch

from app.rag.engine import answer_with_evidence


class TestUncitedAnswer(unittest.TestCase):

    def test_uncited_answer_returns_safe_localized_fallback(self):
        evidence = [{
            "source_url": "https://skillshikshya.com/faq",
            "section": "FAQ",
            "text": "Website information.",
        }]

        for language in ("en", "ne"):
            with self.subTest(language=language):
                client = Mock()
                client.interactions.create.return_value.output_text = (
                    "UNVERIFIED_ADMISSION_INSTRUCTIONS"
                )

                with patch(
                    "app.rag.engine._group_evidence_by_source",
                    return_value=[{
                        "source_url": "https://skillshikshya.com/faq",
                        "section": "FAQ",
                        "text": "Website information.",
                    }],
                ):
                    result = answer_with_evidence(
                        "BHARNA KO K CHA PROCESS?",
                        evidence,
                        client=client,
                        language=language,
                    )

                self.assertEqual(result["citations"], [])
                self.assertNotIn(
                    "UNVERIFIED_ADMISSION_INSTRUCTIONS",
                    result["answer"],
                )

                if language == "en":
                    self.assertIn("verify", result["answer"])
                else:
                    self.assertIn("पुष्टि", result["answer"])

                client.interactions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
