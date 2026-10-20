import unittest


class TestMarkdownChunking(unittest.TestCase):

    def test_preserves_section_and_source(self):
        from app.rag.engine import chunk_markdown

        markdown = """
# Cybersecurity Job Ready Program

Learn cybersecurity with practical training.

## Who Is This Course For?

Beginners interested in cybersecurity careers.

## Course Highlights

Practical labs and career preparation.
"""

        chunks = chunk_markdown(
            markdown,
            source_url="https://skillshikshya.com/courses/cybersecurity",
            max_chars=300,
            overlap_chars=30,
        )

        self.assertGreater(len(chunks), 0)

        self.assertTrue(
            any(
                chunk["section"] == "Who Is This Course For?"
                and "Beginners" in chunk["text"]
                for chunk in chunks
            )
        )

        for chunk in chunks:
            self.assertIn("source_url", chunk)
            self.assertIn("chunk_id", chunk)
            self.assertIn("section", chunk)
            self.assertIn("text", chunk)

    def test_long_sections_are_split_without_losing_content(self):
        from app.rag.engine import chunk_markdown

        content = " ".join(f"topic{i}" for i in range(120))
        markdown = f"# Python Course\n\n## Syllabus\n\n{content}"

        chunks = chunk_markdown(
            markdown,
            source_url="https://example.org/courses/python",
            max_chars=240,
            overlap_chars=30,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(len(chunk["text"]) <= 240 for chunk in chunks)
        )

        combined = " ".join(chunk["text"] for chunk in chunks)

        for index in range(120):
            self.assertIn(f"topic{index}", combined.split())

    def test_chunk_ids_are_stable_and_source_specific(self):
        from app.rag.engine import chunk_markdown

        markdown = "# Python Course\n\n## Overview\n\nLearn Python."

        first = chunk_markdown(
            markdown,
            source_url="https://example.org/courses/python",
        )
        repeated = chunk_markdown(
            markdown,
            source_url="https://example.org/courses/python",
        )
        different_site = chunk_markdown(
            markdown,
            source_url="https://another.org/courses/python",
        )

        self.assertEqual(
            [chunk["chunk_id"] for chunk in first],
            [chunk["chunk_id"] for chunk in repeated],
        )
        self.assertNotEqual(
            first[0]["chunk_id"],
            different_site[0]["chunk_id"],
        )


if __name__ == "__main__":
    unittest.main()


class TestQdrantVectorStore(unittest.TestCase):

    def test_indexes_and_retrieves_relevant_course(self):
        from qdrant_client import QdrantClient
        from app.rag.engine import index_chunks, search_chunks

        chunks = [
            {
                "chunk_id": "00000000-0000-4000-8000-000000000001",
                "source_url": "https://skillshikshya.com/courses/cybersecurity",
                "document_title": "Cybersecurity Course",
                "section": "Course Highlights",
                "text": "Learn cybersecurity and practical security labs.",
            },
            {
                "chunk_id": "00000000-0000-4000-8000-000000000002",
                "source_url": "https://skillshikshya.com/courses/accounting",
                "document_title": "Accounting Course",
                "section": "Course Highlights",
                "text": "Learn accounting, taxation, and VAT.",
            },
        ]

        # Deterministic vectors: no model download required.
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        client = QdrantClient(":memory:")

        try:
            count = index_chunks(
                client,
                collection_name="test_courses",
                chunks=chunks,
                vectors=vectors,
            )

            self.assertEqual(count, 2)

            results = search_chunks(
                client,
                collection_name="test_courses",
                query_vector=[1.0, 0.0, 0.0],
                limit=2,
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                results[0]["source_url"],
                chunks[0]["source_url"],
            )
            self.assertIn(
                "cybersecurity",
                results[0]["text"].lower(),
            )
            self.assertIn("score", results[0])

        finally:
            client.close()

    def test_reindexing_same_chunks_is_idempotent(self):
        from qdrant_client import QdrantClient
        from app.rag.engine import index_chunks

        chunk = {
            "chunk_id": "00000000-0000-4000-8000-000000000003",
            "source_url": "https://example.org/courses/python",
            "document_title": "Python Course",
            "section": "Overview",
            "text": "Learn Python programming.",
        }

        client = QdrantClient(":memory:")

        try:
            for _ in range(2):
                index_chunks(
                    client,
                    collection_name="test_idempotency",
                    chunks=[chunk],
                    vectors=[[1.0, 0.0, 0.0]],
                )

            info = client.get_collection("test_idempotency")

            self.assertEqual(info.points_count, 1)

        finally:
            client.close()


class TestRAGRetrievalInterface(unittest.TestCase):

    def test_rejects_empty_questions(self):
        from app.rag.engine import retrieve_evidence

        for question in ("", "   "):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    retrieve_evidence(question)

    def test_cli_forwards_question_and_displays_citation(self):
        import io
        from unittest.mock import patch
        from app.rag import engine

        question = "What does the cybersecurity course teach?"
        source = (
            "https://skillshikshya.com/courses/"
            "cybersecurity-job-ready-program-in-nepal"
        )

        fake_results = [{
            "score": 0.85,
            "document_title": "Cybersecurity Course",
            "section": "Skills You Will Learn",
            "source_url": source,
            "text": "Network security and ethical hacking.",
        }]

        with (
            patch(
                "sys.argv",
                ["engine.py", "--question", question, "--top-k", "2"],
            ),
            patch.object(
                engine,
                "retrieve_evidence",
                return_value=fake_results,
            ) as retrieve,
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            engine.main()

        retrieve.assert_called_once_with(question, limit=2)

        printed = output.getvalue()
        self.assertIn(source, printed)
        self.assertIn("Skills You Will Learn", printed)
        self.assertIn("Network security", printed)


class TestGroundedAnswerGeneration(unittest.TestCase):

    def test_returns_only_cited_sources(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from app.rag.engine import answer_with_evidence

        evidence = [
            {
                "source_url": "https://example.org/courses/python",
                "section": "Syllabus",
                "text": "Python basics and web development.",
            },
            {
                "source_url": "https://example.org/courses/design",
                "section": "Overview",
                "text": "Graphic design fundamentals.",
            },
        ]

        client = MagicMock()
        client.interactions.create.return_value = SimpleNamespace(
            output_text="Python basics sikaincha. [1]"
        )

        result = answer_with_evidence(
            "What does the Python course teach?",
            evidence,
            client=client,
            model="gemini-3.8-flash",
        )

        self.assertIn("[1]", result["answer"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(
            result["citations"][0]["source_url"],
            "https://example.org/courses/python",
        )
        client.interactions.create.assert_called_once()

    def test_rejects_invented_citation_marker(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from app.rag.engine import answer_with_evidence

        evidence = [{
            "source_url": "https://example.org/courses/python",
            "section": "Syllabus",
            "text": "Learn Python basics.",
        }]

        client = MagicMock()
        client.interactions.create.return_value = SimpleNamespace(
            output_text="Python sikaincha. [9]"
        )

        with self.assertRaises(ValueError):
            answer_with_evidence(
                "What is taught?",
                evidence,
                client=client,
                model="gemini-3.8-flash",
            )

    def test_no_evidence_skips_llm_call(self):
        from unittest.mock import MagicMock
        from app.rag.engine import answer_with_evidence

        client = MagicMock()

        result = answer_with_evidence(
            "What is the course fee?",
            [],
            client=client,
            model="gemini-3.8-flash",
        )

        self.assertEqual(result["citations"], [])
        self.assertTrue(result["answer"])
        client.interactions.create.assert_not_called()


class TestAskSkillsathi(unittest.TestCase):

    def test_retrieval_and_generation_are_connected(self):
        import os
        from unittest.mock import patch
        from app.rag.engine import ask_skillsathi

        evidence = [{
            "source_url": "https://example.org/courses/python",
            "section": "Curriculum",
            "text": "Python programming basics.",
        }]

        expected = {
            "answer": "Python basics sikainchha. [1]",
            "citations": [{
                "marker": "[1]",
                "source_url": "https://example.org/courses/python",
                "section": "Curriculum",
            }],
        }

        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "test-key",
                    "GEMINI_MODEL": "gemini-3.8-flash",
                },
            ),
            patch(
                "app.rag.engine.retrieve_evidence",
                return_value=evidence,
            ) as retrieve,
            patch(
                "app.rag.engine.answer_with_evidence",
                return_value=expected,
            ) as generate,
            patch("google.genai.Client") as client_class,
        ):
            result = ask_skillsathi("What is taught?", top_k=3)

        retrieve.assert_called_once_with(
            "What is taught?", limit=3
        )
        generate.assert_called_once_with(
            "What is taught?",
            evidence,
            client=client_class.return_value,
            model="gemini-3.8-flash",
            language="ne",
        )
        client_class.return_value.close.assert_called_once()
        self.assertEqual(result["retrieved_chunks"], 1)
        self.assertEqual(result["answer"], expected["answer"])

    def test_no_evidence_skips_gemini(self):
        from unittest.mock import patch
        from app.rag.engine import ask_skillsathi

        with (
            patch(
                "app.rag.engine.retrieve_evidence",
                return_value=[],
            ),
            patch("google.genai.Client") as client_class,
        ):
            result = ask_skillsathi("Unknown course?")

        self.assertEqual(result["retrieved_chunks"], 0)
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["answer"])
        client_class.assert_not_called()

    def test_rejects_invalid_questions_and_limits(self):
        from app.rag.engine import ask_skillsathi

        for question in ("", "   "):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    ask_skillsathi(question)

        for top_k in (0, 11, True, 2.5):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    ask_skillsathi("Test question", top_k=top_k)


class TestLanguageSelection(unittest.TestCase):

    def test_english_answer_prompt(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from app.rag.engine import answer_with_evidence

        evidence = [{
            "source_url": "https://skillshikshya.com/courses/python",
            "section": "Curriculum",
            "text": "Python basics.",
        }]
        client = MagicMock()
        client.interactions.create.return_value = SimpleNamespace(
            output_text="The course teaches Python basics. [1]"
        )

        result = answer_with_evidence(
            "What does this course teach?",
            evidence,
            client=client,
            language="en",
        )

        prompt = client.interactions.create.call_args.kwargs["input"]
        self.assertIn("Respond only in English.", prompt)
        self.assertEqual(len(result["citations"]), 1)

    def test_english_fallback_without_evidence(self):
        from unittest.mock import patch
        from app.rag.engine import ask_skillsathi

        with patch(
            "app.rag.engine.retrieve_evidence",
            return_value=[],
        ):
            result = ask_skillsathi(
                "Unknown course?",
                language="en",
            )

        self.assertIn("I could not find", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_invalid_language_is_rejected(self):
        from app.rag.engine import ask_skillsathi

        with self.assertRaises(ValueError):
            ask_skillsathi("Hello", language="fr")
