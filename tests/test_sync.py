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



class TestSnapshotActivation(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient, models

        self.client = QdrantClient(":memory:")
        self.models = models

        for name, identifier, generation in (
            ("snapshot_old", 101, "old"),
            ("snapshot_new", 202, "new"),
        ):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=3,
                    distance=models.Distance.COSINE,
                ),
            )

            self.client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=identifier,
                        vector=[1.0, 0.0, 0.0],
                        payload={"generation": generation},
                    )
                ],
                wait=True,
            )

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="snapshot_old",
                        alias_name="active_test",
                    )
                )
            ]
        )

    def tearDown(self):
        self.client.close()

    def active_generation(self):
        records = self.client.scroll(
            collection_name="active_test",
            limit=10,
            with_payload=True,
        )[0]
        return records[0].payload["generation"]

    def activate(self, **overrides):
        from app.ingestion.sync import activate_collection_snapshot

        arguments = {
            "alias_name": "active_test",
            "expected_active_collection": "snapshot_old",
            "staged_collection": "snapshot_new",
            "expected_chunk_ids": ["202"],
        }

        arguments.update(overrides)

        return activate_collection_snapshot(
            self.client,
            **arguments,
        )

    def test_valid_snapshot_switches_active_alias(self):
        self.assertEqual(self.active_generation(), "old")

        result = self.activate()

        self.assertEqual(self.active_generation(), "new")
        self.assertEqual(result["verified_chunks"], 1)
        self.assertEqual(
            result["previous_collection"],
            "snapshot_old",
        )

        # Old snapshot remains available for rollback.
        old_records = self.client.scroll(
            collection_name="snapshot_old",
            limit=10,
            with_payload=True,
        )[0]

        self.assertEqual(
            old_records[0].payload["generation"],
            "old",
        )

    def test_incomplete_snapshot_never_activates(self):
        with self.assertRaises(ValueError):
            self.activate(
                expected_chunk_ids=["202", "999"],
            )

        self.assertEqual(self.active_generation(), "old")

    def test_unexpected_extra_chunks_are_rejected(self):
        with self.assertRaises(ValueError):
            self.activate(
                expected_chunk_ids=["999"],
            )

        self.assertEqual(self.active_generation(), "old")

    def test_stale_alias_is_rejected(self):
        with self.assertRaises(ValueError):
            self.activate(
                expected_active_collection="wrong_collection",
            )

        self.assertEqual(self.active_generation(), "old")

    def test_missing_staged_collection_is_rejected(self):
        with self.assertRaises(ValueError):
            self.activate(
                staged_collection="missing_snapshot",
            )

        self.assertEqual(self.active_generation(), "old")

    def test_duplicate_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            self.activate(
                expected_chunk_ids=["202", "202"],
            )

        self.assertEqual(self.active_generation(), "old")



class TestStagedSnapshotBuilder(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient, models
        from app.rag.engine import index_chunks

        self.client = QdrantClient(":memory:")
        self.collection = "builder_source"
        self.other_url = (
            "https://skillshikshya.com/courses/another-course"
        )

        self.original = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for reporting.\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n",
            source_url=SOURCE,
        )

        self.other = chunk_markdown(
            "# Another Course\n"
            "## Curriculum\n"
            "Learn cybersecurity.\n",
            source_url=self.other_url,
        )

        chunks = self.original + self.other

        index_chunks(
            self.client,
            collection_name=self.collection,
            chunks=chunks,
            vectors=[[1.0, 0.0, 0.0] for _ in chunks],
        )

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection,
                        alias_name="builder_active",
                    )
                )
            ]
        )

    def tearDown(self):
        self.client.close()

    def prepare(self):
        from app.ingestion.sync import stage_page_refresh

        fresh = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for reporting.\n"
            "## Curriculum\n"
            "Learn Python fundamentals.\n",
            source_url=SOURCE,
        )

        return stage_page_refresh(
            self.client,
            collection_name=self.collection,
            source_url=SOURCE,
            fresh_chunks=fresh,
            embed_texts=lambda texts: [
                [0.0, 1.0, 0.0] for _ in texts
            ],
            vector_size=3,
        )

    def test_build_replaces_changed_chunk_and_preserves_other_course(self):
        from app.ingestion.sync import build_staged_page_snapshot

        plan = self.prepare()

        result = build_staged_page_snapshot(
            self.client,
            source_collection=self.collection,
            staged_collection="builder_v2",
            plan=plan,
        )

        records, _ = self.client.scroll(
            collection_name="builder_v2",
            limit=100,
            with_payload=True,
        )

        texts = [record.payload["text"] for record in records]
        urls = {record.payload["source_url"] for record in records}

        self.assertTrue(any("Python" in text for text in texts))
        self.assertFalse(any("SQL" in text for text in texts))
        self.assertIn(self.other_url, urls)
        self.assertEqual(result["updated_chunks"], 1)
        self.assertEqual(result["removed_chunks"], 1)
        self.assertEqual(
            sorted(str(record.id) for record in records),
            result["expected_chunk_ids"],
        )

        # No alias activation occurred.
        aliases = {
            a.alias_name: a.collection_name
            for a in self.client.get_aliases().aliases
        }
        self.assertEqual(
            aliases["builder_active"],
            self.collection,
        )

    def test_invalid_manifest_does_not_create_snapshot(self):
        from app.ingestion.sync import build_staged_page_snapshot

        plan = self.prepare()
        plan["delete_ids"] = ["not-an-indexed-point"]

        with self.assertRaises(ValueError):
            build_staged_page_snapshot(
                self.client,
                source_collection=self.collection,
                staged_collection="builder_invalid",
                plan=plan,
            )

        self.assertFalse(
            self.client.collection_exists("builder_invalid")
        )

    def test_failed_stage_write_preserves_original(self):
        from app.ingestion.sync import build_staged_page_snapshot

        plan = self.prepare()
        real_client = self.client

        class FailingClient:
            def __getattr__(self, name):
                return getattr(real_client, name)

            def upsert(self, *, collection_name, **kwargs):
                if collection_name == "builder_failure":
                    raise RuntimeError("Simulated staged write failure")
                return real_client.upsert(
                    collection_name=collection_name,
                    **kwargs,
                )

        with self.assertRaises(RuntimeError):
            build_staged_page_snapshot(
                FailingClient(),
                source_collection=self.collection,
                staged_collection="builder_failure",
                plan=plan,
            )

        self.assertFalse(
            self.client.collection_exists("builder_failure")
        )

        original_records, _ = self.client.scroll(
            collection_name=self.collection,
            limit=100,
            with_payload=True,
        )

        self.assertTrue(
            any(
                "SQL" in record.payload["text"]
                for record in original_records
            )
        )



class TestPageRefreshWorkflow(unittest.TestCase):

    def setUp(self):
        from qdrant_client import QdrantClient, models
        from app.rag.engine import index_chunks

        self.client = QdrantClient(":memory:")
        self.collection = "workflow_v1"
        self.alias = "workflow_active"
        self.other_url = (
            "https://skillshikshya.com/courses/other-course"
        )

        self.original = chunk_markdown(
            "# Example Course\n"
            "## Overview\n"
            "Learn Excel for reporting.\n"
            "## Curriculum\n"
            "Learn SQL fundamentals.\n",
            source_url=SOURCE,
        )

        self.other = chunk_markdown(
            "# Other Course\n"
            "## Curriculum\n"
            "Learn cybersecurity.\n",
            source_url=self.other_url,
        )

        chunks = self.original + self.other

        index_chunks(
            self.client,
            collection_name=self.collection,
            chunks=chunks,
            vectors=[[1.0, 0.0, 0.0] for _ in chunks],
        )

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=self.collection,
                        alias_name=self.alias,
                    )
                )
            ]
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

    def active_collection(self):
        return {
            item.alias_name: item.collection_name
            for item in self.client.get_aliases().aliases
        }[self.alias]

    def refresh(self, **overrides):
        from app.ingestion.sync import refresh_page_snapshot

        arguments = {
            "source_collection": self.collection,
            "staged_collection": "workflow_v2",
            "alias_name": self.alias,
            "source_url": SOURCE,
            "fresh_chunks": self.fresh(),
            "embed_texts": lambda texts: [
                [0.0, 1.0, 0.0] for _ in texts
            ],
            "vector_size": 3,
        }

        arguments.update(overrides)

        return refresh_page_snapshot(
            self.client,
            **arguments,
        )

    def test_successful_refresh_activates_new_snapshot(self):
        result = self.refresh()

        self.assertEqual(result["status"], "activated")
        self.assertEqual(self.active_collection(), "workflow_v2")
        self.assertEqual(result["updated_chunks"], 1)
        self.assertEqual(result["removed_chunks"], 1)

        records, _ = self.client.scroll(
            collection_name=self.alias,
            limit=100,
            with_payload=True,
        )

        texts = [record.payload["text"] for record in records]
        urls = {record.payload["source_url"] for record in records}

        self.assertTrue(any("Python" in text for text in texts))
        self.assertFalse(any("SQL" in text for text in texts))
        self.assertIn(self.other_url, urls)

        self.assertTrue(
            self.client.collection_exists(self.collection)
        )

    def test_unchanged_page_skips_snapshot_creation(self):
        from unittest.mock import Mock

        embedder = Mock()

        result = self.refresh(
            fresh_chunks=self.fresh("SQL"),
            embed_texts=embedder,
        )

        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(self.active_collection(), self.collection)
        self.assertFalse(
            self.client.collection_exists("workflow_v2")
        )
        embedder.assert_not_called()

    def test_embedding_failure_preserves_active_collection(self):
        from unittest.mock import Mock

        embedder = Mock(
            side_effect=RuntimeError("Embedding unavailable")
        )

        with self.assertRaises(RuntimeError):
            self.refresh(embed_texts=embedder)

        self.assertEqual(self.active_collection(), self.collection)
        self.assertFalse(
            self.client.collection_exists("workflow_v2")
        )

    def test_wrong_active_alias_rejected_before_embedding(self):
        from unittest.mock import Mock

        embedder = Mock()

        with self.assertRaises(ValueError):
            self.refresh(
                source_collection="unexpected_collection",
                embed_texts=embedder,
            )

        embedder.assert_not_called()
        self.assertEqual(self.active_collection(), self.collection)

    def test_failed_activation_preserves_existing_alias(self):
        from unittest.mock import patch

        with patch(
            "app.ingestion.sync.activate_collection_snapshot",
            side_effect=RuntimeError("Activation unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                self.refresh()

        self.assertEqual(self.active_collection(), self.collection)

        # A built but inactive snapshot may remain for inspection.
        self.assertTrue(
            self.client.collection_exists("workflow_v2")
        )
