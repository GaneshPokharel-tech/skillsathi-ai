"""Cross-process safety tests for the refresh lock."""

import stat
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.ingestion.locking import refresh_lock


CHILD_CODE = """
import sys
from app.ingestion.locking import refresh_lock

try:
    with refresh_lock(sys.argv[1]):
        print("ACQUIRED")
except RuntimeError:
    print("BUSY")
    sys.exit(23)
"""


class TestRefreshLock(unittest.TestCase):

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock_path = self.root / "refresh.lock"

    def child_attempt(self):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_CODE,
                str(self.lock_path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_second_process_cannot_acquire_held_lock(self):
        with refresh_lock(self.lock_path):
            result = self.child_attempt()
            self.assertEqual(
                result.returncode,
                23,
                result.stderr,
            )
            self.assertIn("BUSY", result.stdout)

        result = self.child_attempt()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ACQUIRED", result.stdout)

    def test_exception_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with refresh_lock(self.lock_path):
                raise RuntimeError("test failure")

        result = self.child_attempt()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lock_file_has_private_permissions(self):
        with refresh_lock(self.lock_path):
            permissions = stat.S_IMODE(
                self.lock_path.stat().st_mode
            )
            self.assertEqual(permissions, 0o600)

    def test_symbolic_link_lock_path_is_rejected(self):
        target = self.root / "target.txt"
        target.write_text("Do not modify", encoding="utf-8")

        self.lock_path.symlink_to(target)

        with self.assertRaises((ValueError, OSError)):
            with refresh_lock(self.lock_path):
                pass

        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "Do not modify",
        )


if __name__ == "__main__":
    unittest.main()
