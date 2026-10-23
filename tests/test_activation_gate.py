"""Regression tests for mandatory content verification."""

import unittest

from qdrant_client import QdrantClient, models

from app.rag.engine import chunk_markdown, index_chunks
from app.ingestion.sync import (
    activate_collection_snapshot,
    refresh_page_snapshot,
)


SOURCE_URL = (
    "https://skillshikshya.com/courses/test-course"
)


class TestMandatoryActivationGate(unittest.TestCase):

    def setUp(self):
        self.client = QdrantClient(":memory:")

        self.original_chunks = chunk_markdown(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n",
            source_url=SOURCE_URL,
        )

        self.client.create_collection(
            collection_name="gate_v1",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        self.assertTrue(
            self.original_chunks,
            "Original fixture must produce nonempty chunks.",
        )

        index_chunks(
            self.client,
            collection_name="gate_v1",
            chunks=self.original_chunks,
            vectors=[
                [1.0, 0.0, 0.0]
                for _ in self.original_chunks
            ],
        )

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="gate_v1",
                        alias_name="gate_active",
                    )
                )
            ]
        )

    def tearDown(self):
        self.client.close()

    def active_collection(self):
        aliases = {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }
        return aliases["gate_active"]

    def test_direct_activation_rejects_missing_crawl_proof(self):
        self.client.create_collection(
            collection_name="gate_v2",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        index_chunks(
            self.client,
            collection_name="gate_v2",
            chunks=self.original_chunks,
            vectors=[
                [1.0, 0.0, 0.0]
                for _ in self.original_chunks
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            "(?i)verif|manifest|proof|crawl",
        ):
            activate_collection_snapshot(
                self.client,
                alias_name="gate_active",
                expected_active_collection="gate_v1",
                staged_collection="gate_v2",
                expected_chunk_ids=[
                    chunk["chunk_id"]
                    for chunk in self.original_chunks
                ],
            )

        self.assertEqual(
            self.active_collection(),
            "gate_v1",
        )

    def test_page_refresh_rejects_missing_crawl_proof(self):
        fresh_chunks = chunk_markdown(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn Python fundamentals.\n",
            source_url=SOURCE_URL,
        )

        self.assertTrue(
            fresh_chunks,
            "Fresh fixture must produce nonempty chunks.",
        )

        embedding_calls = []

        def embed_texts(texts):
            embedding_calls.append(list(texts))
            return [
                [0.0, 1.0, 0.0]
                for _ in texts
            ]

        with self.assertRaisesRegex(
            ValueError,
            "(?i)verif|manifest|proof|crawl",
        ):
            refresh_page_snapshot(
                self.client,
                source_collection="gate_v1",
                staged_collection="gate_v2",
                alias_name="gate_active",
                source_url=SOURCE_URL,
                fresh_chunks=fresh_chunks,
                embed_texts=embed_texts,
                vector_size=3,
            )

        self.assertEqual(
            embedding_calls,
            [],
        )

        self.assertEqual(
            self.active_collection(),
            "gate_v1",
        )


    def published_proof(self, markdown):
        """Create a complete, consistent published-run fixture."""
        import hashlib
        import json
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from app.ingestion.crawler import page_id

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        root = Path(temporary.name)
        pages = root / "processed"
        pages.mkdir()

        run_id = "verified-test-run"
        identifier = page_id(SOURCE_URL)
        filename = identifier + ".md"

        content = markdown.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()

        (pages / filename).write_bytes(content)

        (pages / (identifier + ".json")).write_text(
            json.dumps({
                "run_id": run_id,
                "source_url": SOURCE_URL,
                "content_hash": digest,
                "cleaned_character_count": len(markdown),
                "crawled_at": "2026-10-23T00:00:00+00:00",
            }),
            encoding="utf-8",
        )

        manifest_path = root / "crawl_manifest.json"
        manifest_path.write_text(
            json.dumps({
                "run_id": run_id,
                "status": "complete",
                "expected_urls": [SOURCE_URL],
                "attempted": 1,
                "successful": 1,
                "failed": 0,
                "pages": [{
                    "run_id": run_id,
                    "source_url": SOURCE_URL,
                    "final_url": SOURCE_URL,
                    "status": "success",
                }],
            }),
            encoding="utf-8",
        )

        report_path = root / "cleaning_report.json"
        report_path.write_text(
            json.dumps({
                "run_id": run_id,
                "processed": 1,
                "skipped": 0,
                "failed": 0,
                "pages": [{
                    "run_id": run_id,
                    "source_url": SOURCE_URL,
                    "status": "processed",
                    "file": filename,
                    "content_hash": digest,
                    "cleaned_character_count": len(markdown),
                }],
            }),
            encoding="utf-8",
        )

        return {
            "manifest_path": manifest_path,
            "processed_pages_dir": pages,
            "report_path": report_path,
        }

    def test_matching_verified_markdown_allows_refresh(self):
        markdown = (
            "# Test Course\n"
            "## Curriculum\n"
            "Learn Python fundamentals.\n"
        )

        proof = self.published_proof(markdown)
        fresh = chunk_markdown(
            markdown,
            source_url=SOURCE_URL,
        )

        result = refresh_page_snapshot(
            self.client,
            source_collection="gate_v1",
            staged_collection="gate_v2",
            alias_name="gate_active",
            source_url=SOURCE_URL,
            fresh_chunks=fresh,
            embed_texts=lambda texts: [
                [0.0, 1.0, 0.0] for _ in texts
            ],
            vector_size=3,
            **proof,
        )

        self.assertEqual(result["status"], "activated")
        self.assertEqual(self.active_collection(), "gate_v2")
        self.assertTrue(self.client.collection_exists("gate_v1"))

    def test_verified_markdown_rejects_unrelated_fresh_chunks(self):
        proof = self.published_proof(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n"
        )

        unrelated = chunk_markdown(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn fabricated curriculum.\n",
            source_url=SOURCE_URL,
        )

        embedding_calls = []

        def embed_texts(texts):
            embedding_calls.append(list(texts))
            return [[0.0, 1.0, 0.0] for _ in texts]

        with self.assertRaises(ValueError):
            refresh_page_snapshot(
                self.client,
                source_collection="gate_v1",
                staged_collection="gate_v2",
                alias_name="gate_active",
                source_url=SOURCE_URL,
                fresh_chunks=unrelated,
                embed_texts=embed_texts,
                vector_size=3,
                **proof,
            )

        self.assertEqual(embedding_calls, [])
        self.assertEqual(self.active_collection(), "gate_v1")
        self.assertFalse(self.client.collection_exists("gate_v2"))

    def test_verified_markdown_rejects_falsified_staged_payload(self):
        proof = self.published_proof(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n"
        )

        self.client.create_collection(
            collection_name="gate_v2",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        falsified = [
            {
                **chunk,
                "text": "Unverified fabricated course curriculum.",
            }
            for chunk in self.original_chunks
        ]

        index_chunks(
            self.client,
            collection_name="gate_v2",
            chunks=falsified,
            vectors=[
                [0.0, 1.0, 0.0]
                for _ in falsified
            ],
        )

        with self.assertRaises(ValueError):
            activate_collection_snapshot(
                self.client,
                alias_name="gate_active",
                expected_active_collection="gate_v1",
                staged_collection="gate_v2",
                expected_chunk_ids=[
                    chunk["chunk_id"]
                    for chunk in falsified
                ],
                **proof,
            )

        self.assertEqual(self.active_collection(), "gate_v1")


    def prepare_copied_course_snapshot(
        self,
        *,
        tamper_payload=False,
        omit_copied=False,
        tamper_vector=False,
    ):
        """Prepare verified source page plus an unverified copied page."""
        from app.rag.engine import index_chunks

        other_url = (
            "https://skillshikshya.com/courses/other-course"
        )

        other_chunks = chunk_markdown(
            "# Other Course\n"
            "## Curriculum\n"
            "Learn cybersecurity fundamentals.\n",
            source_url=other_url,
        )

        self.assertTrue(other_chunks)

        index_chunks(
            self.client,
            collection_name="gate_v1",
            chunks=other_chunks,
            vectors=[[1.0, 0.0, 0.0] for _ in other_chunks],
        )

        self.client.create_collection(
            collection_name="gate_v2",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        staged_chunks = list(self.original_chunks)

        if not omit_copied:
            for chunk in other_chunks:
                staged_chunks.append(
                    {
                        **chunk,
                        "text": (
                            "Fabricated copied course curriculum."
                            if tamper_payload
                            else chunk["text"]
                        ),
                    }
                )

        vectors = [
            (
                [0.0, 1.0, 0.0]
                if tamper_vector
                and chunk["source_url"] == other_url
                else [1.0, 0.0, 0.0]
            )
            for chunk in staged_chunks
        ]

        index_chunks(
            self.client,
            collection_name="gate_v2",
            chunks=staged_chunks,
            vectors=vectors,
        )

        proof = self.published_proof(
            "# Test Course\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n"
        )

        return {
            "alias_name": "gate_active",
            "expected_active_collection": "gate_v1",
            "staged_collection": "gate_v2",
            "expected_chunk_ids": [
                chunk["chunk_id"] for chunk in staged_chunks
            ],
            **proof,
        }

    def test_rejects_tampered_copied_course_payload(self):
        arguments = self.prepare_copied_course_snapshot(
            tamper_payload=True,
        )

        with self.assertRaises(ValueError):
            activate_collection_snapshot(
                self.client,
                **arguments,
            )

        self.assertEqual(self.active_collection(), "gate_v1")

    def test_rejects_missing_copied_course(self):
        arguments = self.prepare_copied_course_snapshot(
            omit_copied=True,
        )

        with self.assertRaises(ValueError):
            activate_collection_snapshot(
                self.client,
                **arguments,
            )

        self.assertEqual(self.active_collection(), "gate_v1")

    def test_rejects_changed_copied_course_vector(self):
        arguments = self.prepare_copied_course_snapshot(
            tamper_vector=True,
        )

        with self.assertRaises(ValueError):
            activate_collection_snapshot(
                self.client,
                **arguments,
            )

        self.assertEqual(self.active_collection(), "gate_v1")


    def test_rejects_markdown_modified_during_activation(self):
        """Published content changing mid-validation must block activation."""
        from pathlib import Path
        from app.ingestion.crawler import page_id

        markdown = (
            "# Test Course\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n"
        )
        proof = self.published_proof(markdown)

        self.client.create_collection(
            collection_name="gate_v2",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.COSINE,
            ),
        )

        index_chunks(
            self.client,
            collection_name="gate_v2",
            chunks=self.original_chunks,
            vectors=[
                [1.0, 0.0, 0.0]
                for _ in self.original_chunks
            ],
        )

        markdown_path = (
            Path(proof["processed_pages_dir"])
            / (page_id(SOURCE_URL) + ".md")
        )

        real_client = self.client

        class MutatingClient:
            changed = False

            def __getattr__(self, name):
                return getattr(real_client, name)

            def scroll(self, *, collection_name, **kwargs):
                result = real_client.scroll(
                    collection_name=collection_name,
                    **kwargs,
                )

                # Simulate a concurrent file modification AFTER
                # initial content verification, during Qdrant checks.
                if (
                    collection_name == "gate_v1"
                    and kwargs.get("with_vectors") is True
                    and not self.changed
                ):
                    markdown_path.write_text(
                        markdown + "\nUnverified concurrent edit.\n",
                        encoding="utf-8",
                    )
                    self.changed = True

                return result

        client = MutatingClient()

        with self.assertRaises(ValueError):
            activate_collection_snapshot(
                client,
                alias_name="gate_active",
                expected_active_collection="gate_v1",
                staged_collection="gate_v2",
                expected_chunk_ids=[
                    chunk["chunk_id"]
                    for chunk in self.original_chunks
                ],
                **proof,
            )

        self.assertTrue(
            client.changed,
            "Test must exercise the simulated concurrent edit.",
        )
        self.assertEqual(
            self.active_collection(),
            "gate_v1",
        )


if __name__ == "__main__":
    unittest.main()
