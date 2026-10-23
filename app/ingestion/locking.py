"""Cooperative inter-process locking for content refresh operations.

Requires macOS or Linux. All participating processes must use the
same stable lock path. The lock file must not be deleted on release.
"""

import errno
import fcntl
import os
import stat
from contextlib import contextmanager


@contextmanager
def refresh_lock(lock_path):
    """Hold an exclusive, nonblocking process lock during refresh work.

    A busy lock raises RuntimeError. The operating system releases
    the lock if the process exits or crashes.
    """
    path = os.fspath(lock_path)

    if not isinstance(path, str) or not path.strip():
        raise ValueError("A nonempty lock path is required.")

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "Secure refresh locking requires O_NOFOLLOW."
        )

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )

    descriptor = os.open(path, flags, 0o600)

    try:
        details = os.fstat(descriptor)

        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
        ):
            raise ValueError(
                "Refresh lock must be a regular, unlinked-once file."
            )

        if details.st_uid != os.getuid():
            raise PermissionError(
                "Refresh lock belongs to another user."
            )

        os.fchmod(descriptor, 0o600)

        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise RuntimeError(
                    "Another content refresh already holds the lock."
                ) from exc
            raise

        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)

    finally:
        os.close(descriptor)
