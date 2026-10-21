"""Tests for active Qdrant collection resolution."""

import unittest

from qdrant_client import QdrantClient, models

from app.ingestion.sync import ACTIVE_COLLECTION_ALIAS
from app.rag.engine import (
    COLLECTION_NAME,
    resolve_read_collection,
)


class TestActiveCollectionResolver(unittest.TestCase):

    def setUp(self):
        self.client = QdrantClient(":memory:")

        for name in (COLLECTION_NAME, "snapshot_v2"):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=3,
                    distance=models.Distance.COSINE,
                ),
            )

    def tearDown(self):
        self.client.close()

    def test_missing_alias_uses_legacy_collection(self):
        result = resolve_read_collection(
            self.client,
            COLLECTION_NAME,
        )

        self.assertEqual(result, COLLECTION_NAME)

    def test_active_alias_is_preferred(self):
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="snapshot_v2",
                        alias_name=ACTIVE_COLLECTION_ALIAS,
                    )
                )
            ]
        )

        result = resolve_read_collection(
            self.client,
            COLLECTION_NAME,
        )

        self.assertEqual(result, ACTIVE_COLLECTION_ALIAS)

    def test_explicit_collection_is_not_redirected(self):
        result = resolve_read_collection(
            self.client,
            "snapshot_v2",
        )

        self.assertEqual(result, "snapshot_v2")


if __name__ == "__main__":
    unittest.main()



class TestActiveCollectionIntegration(unittest.TestCase):

    def setUp(self):
        from app.rag.engine import chunk_markdown, index_chunks

        self.client = QdrantClient(":memory:")

        self.old_url = (
            "https://skillshikshya.com/courses/legacy-training"
        )
        self.analytics_url = (
            "https://skillshikshya.com/courses/analytics-training"
        )
        self.security_url = (
            "https://skillshikshya.com/courses/security-training"
        )

        old_chunks = chunk_markdown(
            "# Legacy Training\n"
            "## Curriculum\n"
            "Old training information.\n",
            source_url=self.old_url,
        )

        new_chunks = (
            chunk_markdown(
                "# Analytics Training\n"
                "## Curriculum\n"
                "Learn SQL and dashboards.\n",
                source_url=self.analytics_url,
            )
            +
            chunk_markdown(
                "# Security Training\n"
                "## Curriculum\n"
                "Learn network security.\n",
                source_url=self.security_url,
            )
        )

        index_chunks(
            self.client,
            collection_name=COLLECTION_NAME,
            chunks=old_chunks,
            vectors=[
                [1.0, 0.0, 0.0] for _ in old_chunks
            ],
        )

        index_chunks(
            self.client,
            collection_name="snapshot_v2",
            chunks=new_chunks,
            vectors=[
                [1.0, 0.0, 0.0] for _ in new_chunks
            ],
        )

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="snapshot_v2",
                        alias_name=ACTIVE_COLLECTION_ALIAS,
                    )
                )
            ]
        )

    def tearDown(self):
        self.client.close()

    def test_course_catalog_uses_active_snapshot(self):
        from app.tools.courses import list_courses

        catalog = list_courses(client=self.client)
        urls = {item["source_url"] for item in catalog}

        self.assertEqual(
            urls,
            {self.analytics_url, self.security_url},
        )
        self.assertNotIn(self.old_url, urls)

    def test_course_details_and_comparison_use_active_snapshot(self):
        from app.tools.courses import (
            get_course_details,
            compare_courses,
        )

        details = get_course_details(
            self.analytics_url,
            client=self.client,
        )

        self.assertIsNotNone(details)
        self.assertTrue(
            any(
                "SQL" in item["text"]
                for item in details["evidence"]
            )
        )

        self.assertIsNone(
            get_course_details(
                self.old_url,
                client=self.client,
            )
        )

        comparison = compare_courses(
            self.analytics_url,
            self.security_url,
            client=self.client,
        )

        self.assertTrue(comparison["comparison_ready"])

    def test_missing_alias_restores_legacy_catalog(self):
        from app.tools.courses import list_courses

        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(
                        alias_name=ACTIVE_COLLECTION_ALIAS,
                    )
                )
            ]
        )

        catalog = list_courses(client=self.client)

        self.assertEqual(
            {item["source_url"] for item in catalog},
            {self.old_url},
        )

    def test_rag_retrieval_reads_active_snapshot(self):
        from unittest.mock import patch
        from app.rag.engine import retrieve_evidence

        class FakeVector:
            def tolist(self):
                return [1.0, 0.0, 0.0]

        class NonClosingClient:
            def __init__(self, client):
                self.client = client

            def __getattr__(self, name):
                return getattr(self.client, name)

            def close(self):
                pass

        proxy = NonClosingClient(self.client)

        with (
            patch("app.rag.engine.QDRANT_PATH") as path,
            patch(
                "qdrant_client.QdrantClient",
                return_value=proxy,
            ),
            patch("fastembed.TextEmbedding") as embedding,
        ):
            path.is_dir.return_value = True
            embedding.return_value.embed.return_value = iter(
                [FakeVector()]
            )

            evidence = retrieve_evidence(
                "Compare course skills",
                limit=10,
            )

        self.assertTrue(evidence)

        urls = {
            item["source_url"]
            for item in evidence
        }

        self.assertEqual(
            urls,
            {self.analytics_url, self.security_url},
        )
        self.assertNotIn(self.old_url, urls)
