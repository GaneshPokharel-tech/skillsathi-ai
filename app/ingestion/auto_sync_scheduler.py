
"""Periodic verified website refresh for a single-worker FastAPI server.

Crawl without holding chat_lock. Serialize embedded Qdrant operations
with chat requests. The default is disabled until explicitly enabled.
"""

import logging
import os
from pathlib import Path
from threading import Event, Thread

from qdrant_client import QdrantClient

from app.ingestion.auto_sync import prepare_refresh_once
from app.ingestion.locking import refresh_lock
from app.ingestion.vector_activation import (
    activate_prepared_refresh,
)
from app.rag.engine import QDRANT_PATH


logger = logging.getLogger(__name__)


def _log_refresh_failure(exc: Exception) -> None:
    """Log the traceback location without exposing source content."""
    import traceback

    frames = traceback.extract_tb(exc.__traceback__)

    if frames:
        frame = frames[-1]
        logger.warning(
            "Website refresh failed: %s at %s:%d in %s",
            type(exc).__name__,
            Path(frame.filename).name,
            frame.lineno,
            frame.name,
        )
    else:
        logger.warning(
            "Website refresh failed: %s",
            type(exc).__name__,
        )




def run_refresh_job(
    *,
    chat_lock,
    receipt_path=Path("data/knowledge_activation.json"),
    runs_root=Path("data/refresh_runs"),
    run_lock_path=Path("data/auto-sync-run.lock"),
):
    """Run one complete refresh; activate only after verification.

    A process-wide file lock prevents overlapping scheduled runs.
    The API chat lock guards Qdrant indexing and alias activation.
    """
    receipt_path = Path(receipt_path)
    runs_root = Path(runs_root)
    run_lock_path = Path(run_lock_path)

    with refresh_lock(run_lock_path):
        prepared = prepare_refresh_once(
            receipt_path=receipt_path,
            runs_root=runs_root,
            lock_path=run_lock_path.with_name(
                "auto-sync-prepare.lock"
            ),
        )

        changes = prepared["changes"]

        if changes.get("unverified_missing"):
            raise RuntimeError(
                "Refresh contains unverified missing sources."
            )

        if not any(
            changes.get(key)
            for key in ("added", "updated", "removed")
        ):
            return {
                "status": "unchanged",
                "verified_pages": prepared["verified_pages"],
            }

        with chat_lock:
            db = QdrantClient(path=str(QDRANT_PATH))

            try:
                return activate_prepared_refresh(
                    prepared,
                    db=db,
                    receipt_path=receipt_path,
                )
            finally:
                db.close()


class AutoSyncScheduler:
    """Run periodic refresh without blocking normal API startup."""

    def __init__(
        self,
        *,
        chat_lock,
        initial_delay_seconds=60,
        interval_seconds=None,
    ):
        self.chat_lock = chat_lock
        self.initial_delay_seconds = initial_delay_seconds
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self.thread = None

    def start(self):
        if self.thread is not None:
            return

        if os.getenv(
            "SKILLSHIKSHYA_AUTO_SYNC_ENABLED", "0"
        ) != "1":
            logger.info("Website auto-sync is disabled.")
            return

        if self.interval_seconds is None:
            hours = float(os.getenv(
                "SKILLSHIKSHYA_AUTO_SYNC_INTERVAL_HOURS",
                "6",
            ))

            if not 1 <= hours <= 168:
                raise ValueError(
                    "Auto-sync interval must be 1-168 hours."
                )

            self.interval_seconds = hours * 3600

        if (
            self.interval_seconds <= 0
            or self.initial_delay_seconds < 0
        ):
            raise ValueError("Invalid auto-sync schedule.")

        self.thread = Thread(
            target=self._run,
            name="skillshikshya-auto-sync",
            daemon=False,
        )
        self.thread.start()
        logger.info("Website auto-sync scheduler started.")

    def _run(self):
        if self._stop.wait(self.initial_delay_seconds):
            return

        while not self._stop.is_set():
            try:
                result = run_refresh_job(
                    chat_lock=self.chat_lock
                )
                logger.info(
                    "Website refresh finished: %s",
                    result.get("status", "unknown"),
                )
            except Exception as exc:
                _log_refresh_failure(exc)

            if self._stop.wait(self.interval_seconds):
                break

    def stop(self):
        self._stop.set()

        if self.thread is not None:
            # Finish the active job before process shutdown.
            self.thread.join()
