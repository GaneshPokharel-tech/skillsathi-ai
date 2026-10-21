"""Tests for non-mutating page refresh planning."""

import unittest

from app.ingestion.sync import plan_page_refresh
from app.rag.engine import chunk_markdown


SOURCE = "https://skillshikshya.com/courses/test-course"


class TestPageRefreshPlanning(unittest.TestCase):

    def chunks(self, curriculum="SQL"):
        return chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for practical reporting.\n"
            "## Curriculum\n"
            f"Learn {curriculum} for data work.\n",
            source_url=SOURCE,
        )

    def test_unchanged_page_needs_no_updates(self):
        chunks = self.chunks()
        plan = plan_page_refresh(
            source_url=SOURCE,
            existing_chunk_ids=[c["chunk_id"] for c in chunks],
            fresh_chunks=chunks,
        )

        self.assertEqual(plan["upsert_chunks"], [])
        self.assertEqual(plan["delete_ids"], [])
        self.assertEqual(len(plan["unchanged_ids"]), len(chunks))

    def test_changed_page_replaces_only_changed_chunks(self):
        original = self.chunks("SQL")
        updated = self.chunks("Python")

        plan = plan_page_refresh(
            source_url=SOURCE,
            existing_chunk_ids=[c["chunk_id"] for c in original],
            fresh_chunks=updated,
        )

        self.assertEqual(len(plan["upsert_chunks"]), 1)
        self.assertEqual(len(plan["delete_ids"]), 1)
        self.assertEqual(len(plan["unchanged_ids"]), 1)

    def test_first_index_adds_all_chunks(self):
        chunks = self.chunks()
        plan = plan_page_refresh(
            source_url=SOURCE,
            existing_chunk_ids=[],
            fresh_chunks=chunks,
        )

        self.assertEqual(len(plan["upsert_chunks"]), len(chunks))
        self.assertEqual(plan["delete_ids"], [])

    def test_empty_refresh_is_rejected(self):
        with self.assertRaises(ValueError):
            plan_page_refresh(
                source_url=SOURCE,
                existing_chunk_ids=["old-chunk"],
                fresh_chunks=[],
            )

    def test_other_website_chunks_are_rejected(self):
        chunks = self.chunks()
        chunks[0]["source_url"] = "https://other.example/course"

        with self.assertRaises(ValueError):
            plan_page_refresh(
                source_url=SOURCE,
                existing_chunk_ids=[],
                fresh_chunks=chunks,
            )

    def test_duplicate_chunk_ids_are_rejected(self):
        chunks = self.chunks()

        with self.assertRaises(ValueError):
            plan_page_refresh(
                source_url=SOURCE,
                existing_chunk_ids=[],
                fresh_chunks=[chunks[0], chunks[0]],
            )

    def test_duplicate_existing_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            plan_page_refresh(
                source_url=SOURCE,
                existing_chunk_ids=["same", "same"],
                fresh_chunks=self.chunks(),
            )


if __name__ == "__main__":
    unittest.main()



class TestIndexedPageLookup(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient
        from app.rag.engine import index_chunks

        self.client = QdrantClient(":memory:")
        self.collection = "page_refresh_test"
        self.other_url = (
            "https://skillshikshya.com/courses/another-course"
        )

        self.original = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for practical reporting.\n"
            "## Curriculum\n"
            "Learn SQL for data work.\n",
            source_url=SOURCE,
        )

        self.other = chunk_markdown(
            "# Another Course\n"
            "## Overview\n"
            "Learn cybersecurity fundamentals.\n",
            source_url=self.other_url,
        )

        all_chunks = self.original + self.other

        index_chunks(
            self.client,
            collection_name=self.collection,
            chunks=all_chunks,
            vectors=[[1.0, 0.0, 0.0] for _ in all_chunks],
        )

    def tearDown(self):
        self.client.close()

    def test_reads_only_requested_source_page(self):
        from app.ingestion.sync import get_indexed_page_chunk_ids

        result = get_indexed_page_chunk_ids(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
        )

        self.assertEqual(
            result,
            sorted(chunk["chunk_id"] for chunk in self.original),
        )

        self.assertNotIn(self.other[0]["chunk_id"], result)

    def test_pagination_returns_all_chunks(self):
        from app.ingestion.sync import get_indexed_page_chunk_ids

        result = get_indexed_page_chunk_ids(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
            batch_size=1,
        )

        self.assertEqual(
            len(result),
            len(self.original),
        )

    def test_missing_collection_returns_empty_list(self):
        from app.ingestion.sync import get_indexed_page_chunk_ids

        result = get_indexed_page_chunk_ids(
            self.client,
            collection_name="nonexistent_collection",
            source_url=SOURCE,
        )

        self.assertEqual(result, [])

    def test_existing_ids_feed_refresh_planner(self):
        from app.ingestion.sync import (
            get_indexed_page_chunk_ids,
            plan_page_refresh,
        )

        previous_ids = get_indexed_page_chunk_ids(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
        )

        updated = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for practical reporting.\n"
            "## Curriculum\n"
            "Learn Python for data work.\n",
            source_url=SOURCE,
        )

        plan = plan_page_refresh(
            source_url=SOURCE,
            existing_chunk_ids=previous_ids,
            fresh_chunks=updated,
        )

        self.assertEqual(len(plan["upsert_chunks"]), 1)
        self.assertEqual(len(plan["delete_ids"]), 1)
        self.assertEqual(len(plan["unchanged_ids"]), 1)

        # Planning must not modify indexed data.
        stored_ids = get_indexed_page_chunk_ids(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
        )

        self.assertEqual(stored_ids, previous_ids)



class TestStagedPageRefresh(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient
        from app.rag.engine import index_chunks

        self.client = QdrantClient(":memory:")
        self.collection = "staged_refresh_test"

        self.original = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for reporting.\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n",
            source_url=SOURCE,
        )

        index_chunks(
            self.client,
            collection_name=self.collection,
            chunks=self.original,
            vectors=[[1.0, 0.0, 0.0] for _ in self.original],
        )

    def tearDown(self):
        self.client.close()

    def fresh(self, curriculum="Python"):
        return chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for reporting.\n"
            "## Curriculum\n"
            f"Learn {curriculum} fundamentals.\n",
            source_url=SOURCE,
        )

    def stored_ids(self):
        from app.ingestion.sync import get_indexed_page_chunk_ids

        return get_indexed_page_chunk_ids(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
        )

    def test_changed_page_stages_without_writing(self):
        from unittest.mock import Mock
        from app.ingestion.sync import stage_page_refresh

        before = self.stored_ids()

        embedder = Mock(
            side_effect=lambda texts: [
                [0.0, 1.0, 0.0] for _ in texts
            ]
        )

        plan = stage_page_refresh(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
            fresh_chunks=self.fresh(),
            embed_texts=embedder,
            vector_size=3,
        )

        self.assertEqual(len(plan["upsert_chunks"]), 1)
        self.assertEqual(len(plan["upsert_vectors"]), 1)
        self.assertEqual(len(plan["delete_ids"]), 1)
        self.assertEqual(len(plan["unchanged_ids"]), 1)

        embedder.assert_called_once()
        self.assertEqual(len(embedder.call_args.args[0]), 1)

        self.assertEqual(self.stored_ids(), before)

    def test_unchanged_page_skips_embedding(self):
        from unittest.mock import Mock
        from app.ingestion.sync import stage_page_refresh

        embedder = Mock()

        plan = stage_page_refresh(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
            fresh_chunks=self.fresh("SQL"),
            embed_texts=embedder,
            vector_size=3,
        )

        self.assertEqual(plan["upsert_chunks"], [])
        self.assertEqual(plan["upsert_vectors"], [])
        self.assertEqual(plan["delete_ids"], [])
        embedder.assert_not_called()

    def test_embedding_failure_preserves_database(self):
        from unittest.mock import Mock
        from app.ingestion.sync import stage_page_refresh

        before = self.stored_ids()

        embedder = Mock(
            side_effect=RuntimeError("Embedding unavailable")
        )

        with self.assertRaises(RuntimeError):
            stage_page_refresh(
                self.client,
                collection_name=self.collection,
                source_url=SOURCE,
                fresh_chunks=self.fresh(),
                embed_texts=embedder,
                vector_size=3,
            )

        self.assertEqual(self.stored_ids(), before)

    def test_invalid_vector_dimension_is_rejected(self):
        from app.ingestion.sync import stage_page_refresh

        before = self.stored_ids()

        with self.assertRaises(ValueError):
            stage_page_refresh(
                self.client,
                collection_name=self.collection,
                source_url=SOURCE,
                fresh_chunks=self.fresh(),
                embed_texts=lambda texts: [
                    [1.0, 0.0] for _ in texts
                ],
                vector_size=3,
            )

        self.assertEqual(self.stored_ids(), before)

    def test_incomplete_embedding_result_is_rejected(self):
        from app.ingestion.sync import stage_page_refresh

        before = self.stored_ids()

        with self.assertRaises(ValueError):
            stage_page_refresh(
                self.client,
                collection_name=self.collection,
                source_url=SOURCE,
                fresh_chunks=self.fresh(),
                embed_texts=lambda texts: [],
                vector_size=3,
            )

        self.assertEqual(self.stored_ids(), before)
