
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.ingestion.vector_activation import activate_prepared_refresh
from app.rag.engine import index_chunks


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def embed(self, texts, **kwargs):
        if self.fail:
            raise RuntimeError("Embedding unavailable")
        self.calls.extend(texts)
        return [FakeVector([0.2] * 384) for _ in texts]


class VectorActivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        self.db = QdrantClient(":memory:")
        self.addCleanup(self.db.close)

        self.old_url = "https://skillshikshya.com/about"
        self.new_url = (
            "https://skillshikshya.com/blogs/new-article"
        )

        self.old_chunk = {
            "chunk_id": str(UUID(int=1)),
            "source_url": self.old_url,
            "document_title": "About",
            "section": "Mission",
            "text": "Existing verified mission",
        }
        self.new_chunk = {
            "chunk_id": str(UUID(int=2)),
            "source_url": self.new_url,
            "document_title": "New Article",
            "section": "Introduction",
            "text": "New verified content",
        }

        self.old_pointer = self.root / "old" / "active-generation.json"
        self.new_pointer = self.root / "new" / "active-generation.json"
        self.receipt = self.root / "activation.json"

        self.original_receipt = {
            "active_alias": "skillshikshya_active",
            "active_collection": "old_collection",
            "snapshot_pointer": str(self.old_pointer),
            "verified_pages": 1,
            "active_vectors": 1,
        }

        self.receipt.write_text(
            json.dumps(self.original_receipt),
            encoding="utf-8",
        )

        self.db.create_collection(
            collection_name="old_collection",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

        index_chunks(
            self.db,
            collection_name="old_collection",
            chunks=[self.old_chunk],
            vectors=[[0.1] * 384],
        )

        self.db.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name="old_collection",
                        alias_name="skillshikshya_active",
                    )
                )
            ]
        )

        self.prepared = {
            "run_id": "a" * 32,
            "snapshot_pointer": str(self.new_pointer),
            "verified_pages": 2,
            "previous_collection": "old_collection",
        }

    def alias_target(self):
        return {
            item.alias_name: item.collection_name
            for item in self.db.get_aliases().aliases
        }["skillshikshya_active"]

    def patches(self):
        def snapshot(pointer):
            if Path(pointer) == self.old_pointer:
                return {}, {
                    "status": "complete",
                    "validated_pages": 1,
                    "validated_urls": [self.old_url],
                }
            if Path(pointer) == self.new_pointer:
                return {
                    "processed_pages_dir": self.root,
                    "report_path": self.root / "report.json",
                }, {
                    "status": "complete",
                    "run_id": "a" * 32,
                    "validated_pages": 2,
                    "validated_urls": [
                        self.old_url, self.new_url
                    ],
                }
            raise ValueError("Unexpected snapshot pointer")

        def chunks(*, source_url, **kwargs):
            if source_url == self.old_url:
                return [self.old_chunk]
            if source_url == self.new_url:
                return [self.new_chunk]
            raise ValueError("Unexpected source")

        return (
            patch(
                "app.ingestion.vector_activation._verified_snapshot",
                side_effect=snapshot,
            ),
            patch(
                "app.ingestion.vector_activation.sync._load_verified_page_chunks",
                side_effect=chunks,
            ),
        )

    def test_reuses_old_vector_and_activates_verified_new(self):
        embedding = FakeEmbedder()
        first, second = self.patches()

        with first, second:
            result = activate_prepared_refresh(
                self.prepared,
                db=self.db,
                receipt_path=self.receipt,
                embedder=embedding,
            )

        self.assertEqual(result["reused_vectors"], 1)
        self.assertEqual(result["new_embeddings"], 1)
        self.assertEqual(result["active_vectors"], 2)
        self.assertEqual(
            embedding.calls,
            ["New verified content"],
        )
        self.assertEqual(
            self.alias_target(),
            result["active_collection"],
        )
        self.assertTrue(
            self.db.collection_exists("old_collection")
        )
        self.assertEqual(
            json.loads(self.receipt.read_text())["active_vectors"],
            2,
        )

    def test_embedding_failure_preserves_baseline(self):
        first, second = self.patches()

        with first, second:
            with self.assertRaises(RuntimeError):
                activate_prepared_refresh(
                    self.prepared,
                    db=self.db,
                    receipt_path=self.receipt,
                    embedder=FakeEmbedder(fail=True),
                )

        self.assertEqual(self.alias_target(), "old_collection")
        self.assertEqual(
            json.loads(self.receipt.read_text()),
            self.original_receipt,
        )
        self.assertEqual(
            self.db.count(
                collection_name="old_collection",
                exact=True,
            ).count,
            1,
        )

    def test_rejects_unexpected_old_collection(self):
        self.prepared["previous_collection"] = "other"

        with self.assertRaises(ValueError):
            activate_prepared_refresh(
                self.prepared,
                db=self.db,
                receipt_path=self.receipt,
            )

        self.assertEqual(self.alias_target(), "old_collection")


if __name__ == "__main__":
    unittest.main()
