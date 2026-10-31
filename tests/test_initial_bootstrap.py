import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from qdrant_client import QdrantClient, models

from app.ingestion import bootstrap


class FakeEmbedder:
    def embed(self, texts):
        for _ in texts:
            yield [0.01] * 384


class BadEmbedder:
    def embed(self, texts):
        for _ in texts:
            yield [0.01] * 100


class BootstrapTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipt = self.root / "knowledge_activation.json"
        self.pointer = self.root / "active-generation.json"
        self.db = QdrantClient(":memory:")
        self.addCleanup(self.db.close)

        self.url = "https://skillshikshya.com/about"
        self.proof = {
            "manifest_path": "mock-manifest",
            "processed_pages_dir": "mock-processed",
            "report_path": "mock-report",
        }
        self.verified = {
            "status": "complete",
            "run_id": "a" * 32,
            "validated_urls": [self.url],
            "validated_pages": 1,
        }
        self.chunks = [{
            "chunk_id": str(uuid4()),
            "source_url": self.url,
            "document_title": "About SkillShikshya",
            "section": "Mission",
            "text": "Practical, job-relevant IT education.",
        }]

    def activate(self, *, embedder=None, verified=None):
        with (
            patch.object(
                bootstrap.publication,
                "resolve_active_generation",
                return_value=self.proof,
            ),
            patch.object(
                bootstrap.sync,
                "verify_published_crawl_run",
                return_value=verified or self.verified,
            ),
            patch.object(
                bootstrap.sync,
                "_load_verified_page_chunks",
                return_value=self.chunks,
            ),
        ):
            return bootstrap.bootstrap_verified_generation(
                db=self.db,
                pointer=self.pointer,
                receipt_path=self.receipt,
                embedder=embedder or FakeEmbedder(),
            )

    def test_fresh_database_initializes_and_verifies(self):
        result = self.activate()

        self.assertEqual(result["status"], "activated")
        self.assertEqual(result["verified_pages"], 1)
        self.assertEqual(result["active_vectors"], 1)
        self.assertTrue(self.receipt.is_file())

        aliases = {
            item.alias_name: item.collection_name
            for item in self.db.get_aliases().aliases
        }
        self.assertEqual(
            aliases["skillshikshya_active"],
            result["active_collection"],
        )

    def test_existing_receipt_is_never_overwritten(self):
        self.receipt.write_text("DO NOT OVERWRITE")
        with self.assertRaises(RuntimeError):
            self.activate()
        self.assertEqual(
            self.receipt.read_text(),
            "DO NOT OVERWRITE",
        )

    def test_nonempty_qdrant_path_is_rejected(self):
        folder = self.root / "qdrant"
        folder.mkdir()
        (folder / "existing").write_text("important")

        with self.assertRaises(RuntimeError):
            bootstrap._preflight(self.receipt, folder)

        self.assertTrue((folder / "existing").exists())

    def test_invalid_snapshot_does_not_create_collection(self):
        incomplete = dict(self.verified, status="failed")

        with self.assertRaises(ValueError):
            self.activate(verified=incomplete)

        self.assertEqual(
            self.db.get_collections().collections, []
        )

    def test_bad_embedding_does_not_create_collection(self):
        with self.assertRaises(ValueError):
            self.activate(embedder=BadEmbedder())

        self.assertFalse(self.receipt.exists())
        self.assertEqual(
            self.db.get_collections().collections, []
        )


if __name__ == "__main__":
    unittest.main()
