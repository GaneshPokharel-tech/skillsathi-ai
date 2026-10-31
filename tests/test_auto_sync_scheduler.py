
import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event, Lock
from unittest.mock import MagicMock, patch

from app.ingestion.auto_sync_scheduler import (
    AutoSyncScheduler,
    run_refresh_job,
)


class AutoSyncSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock_path = self.root / "refresh-run.lock"
        self.receipt = self.root / "activation.json"
        self.receipt.write_text(
            json.dumps({"active_collection": "old"}),
            encoding="utf-8",
        )

    def test_unchanged_website_skips_qdrant(self):
        prepared = {
            "verified_pages": 40,
            "changes": {
                "added": [],
                "updated": [],
                "removed": [],
                "unverified_missing": [],
            },
        }

        with patch(
            "app.ingestion.auto_sync_scheduler.prepare_refresh_once",
            return_value=prepared,
        ), patch(
            "app.ingestion.auto_sync_scheduler.QdrantClient",
        ) as qdrant:
            result = run_refresh_job(
                chat_lock=Lock(),
                receipt_path=self.receipt,
                runs_root=self.root,
                run_lock_path=self.lock_path,
            )

        self.assertEqual(result["status"], "unchanged")
        qdrant.assert_not_called()

    def test_changed_website_activates_under_chat_lock(self):
        chat_lock = Lock()
        prepared = {
            "verified_pages": 41,
            "changes": {
                "added": [
                    "https://skillshikshya.com/blogs/new-article"
                ],
                "updated": [],
                "removed": [],
                "unverified_missing": [],
            },
        }

        client = MagicMock()
        observed = []

        def activate(data, *, db, receipt_path):
            observed.append(chat_lock.locked())
            self.assertIs(db, client)
            return {
                "status": "activated",
                "active_vectors": 930,
            }

        with patch(
            "app.ingestion.auto_sync_scheduler.prepare_refresh_once",
            return_value=prepared,
        ), patch(
            "app.ingestion.auto_sync_scheduler.QdrantClient",
            return_value=client,
        ), patch(
            "app.ingestion.auto_sync_scheduler.activate_prepared_refresh",
            side_effect=activate,
        ):
            result = run_refresh_job(
                chat_lock=chat_lock,
                receipt_path=self.receipt,
                runs_root=self.root,
                run_lock_path=self.lock_path,
            )

        self.assertEqual(result["status"], "activated")
        self.assertEqual(observed, [True])
        self.assertFalse(chat_lock.locked())
        client.close.assert_called_once()

    def test_preparation_failure_never_opens_qdrant(self):
        with patch(
            "app.ingestion.auto_sync_scheduler.prepare_refresh_once",
            side_effect=RuntimeError("Crawl failed"),
        ), patch(
            "app.ingestion.auto_sync_scheduler.QdrantClient",
        ) as qdrant:
            with self.assertRaises(RuntimeError):
                run_refresh_job(
                    chat_lock=Lock(),
                    receipt_path=self.receipt,
                    runs_root=self.root,
                    run_lock_path=self.lock_path,
                )

        qdrant.assert_not_called()
        self.assertEqual(
            json.loads(self.receipt.read_text()),
            {"active_collection": "old"},
        )

    def test_disabled_scheduler_does_not_start(self):
        with patch.dict(
            os.environ,
            {"SKILLSHIKSHYA_AUTO_SYNC_ENABLED": "0"},
        ):
            scheduler = AutoSyncScheduler(
                chat_lock=Lock(),
                initial_delay_seconds=0,
            )
            scheduler.start()
            self.assertIsNone(scheduler.thread)
            scheduler.stop()

    def test_enabled_scheduler_runs_and_stops(self):
        triggered = Event()

        def fake_job(**kwargs):
            triggered.set()
            return {"status": "unchanged"}

        with patch.dict(
            os.environ,
            {
                "SKILLSHIKSHYA_AUTO_SYNC_ENABLED": "1",
                "SKILLSHIKSHYA_AUTO_SYNC_INTERVAL_HOURS": "6",
            },
        ), patch(
            "app.ingestion.auto_sync_scheduler.run_refresh_job",
            side_effect=fake_job,
        ):
            scheduler = AutoSyncScheduler(
                chat_lock=Lock(),
                initial_delay_seconds=0,
            )
            scheduler.start()

            try:
                self.assertTrue(triggered.wait(timeout=3))
            finally:
                scheduler.stop()

            self.assertFalse(scheduler.thread.is_alive())

    def test_fastapi_lifespan_starts_and_stops_scheduler(self):
        from fastapi.testclient import TestClient
        from app.api.server import app

        with patch(
            "app.api.server.AutoSyncScheduler"
        ) as factory:
            with TestClient(app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 200)

            factory.return_value.start.assert_called_once()
            factory.return_value.stop.assert_called_once()



    def test_refresh_failure_reports_location_safely(self):
        from app.ingestion.auto_sync_scheduler import (
            _log_refresh_failure,
        )

        try:
            raise ValueError("PRIVATE_SOURCE_CONTENT_123")
        except ValueError as exc:
            with self.assertLogs(
                "app.ingestion.auto_sync_scheduler",
                level="WARNING",
            ) as captured:
                _log_refresh_failure(exc)

        message = captured.output[0]

        self.assertIn("ValueError", message)
        self.assertIn("test_auto_sync_scheduler.py", message)
        self.assertNotIn(
            "PRIVATE_SOURCE_CONTENT_123",
            message,
        )

if __name__ == "__main__":
    unittest.main()
