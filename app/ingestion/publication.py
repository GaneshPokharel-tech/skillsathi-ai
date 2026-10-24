"""Preparation of independently verified crawl generations.

Each generation is stored in a separate directory and published
using a same-filesystem directory rename. Existing generations
must not be overwritten.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

from app.ingestion.crawler import page_id
from app.ingestion.sync import verify_published_crawl_run


def prepare_verified_generation(
    *,
    manifest_path,
    processed_pages_dir,
    report_path,
    generations_dir,
) -> dict:
    """Prepare a verified generation without changing active content.

    Returns fixed paths to its manifest, processed files and report.
    This function does not switch an active-generation pointer.
    """
    manifest_path = Path(manifest_path)
    processed_pages_dir = Path(processed_pages_dir)
    report_path = Path(report_path)
    generations_dir = Path(generations_dir)

    # Reject incomplete or inconsistent source content.
    verified = verify_published_crawl_run(
        manifest_path=manifest_path,
        processed_pages_dir=processed_pages_dir,
        report_path=report_path,
    )

    # Hash the run ID so it can never introduce path traversal.
    identifier = hashlib.sha256(
        verified["run_id"].encode("utf-8")
    ).hexdigest()

    generations_dir.mkdir(parents=True, exist_ok=True)

    destination = generations_dir / ("run-" + identifier)

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Generation already exists: {destination}"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=".preparing-",
            dir=generations_dir,
        )
    )

    try:
        staged_pages = staging / "processed"
        staged_pages.mkdir()

        staged_manifest = staging / "crawl_manifest.json"
        staged_report = staging / "cleaning_report.json"

        shutil.copyfile(manifest_path, staged_manifest)
        shutil.copyfile(report_path, staged_report)

        for url in verified["validated_urls"]:
            identifier = page_id(url)

            for extension in (".md", ".json"):
                filename = identifier + extension

                shutil.copyfile(
                    processed_pages_dir / filename,
                    staged_pages / filename,
                )

        staged_proof = {
            "manifest_path": staged_manifest,
            "processed_pages_dir": staged_pages,
            "report_path": staged_report,
        }

        # A source modification during copying must not produce
        # a published generation with inconsistent content.
        staged_verified = verify_published_crawl_run(
            **staged_proof
        )

        if staged_verified != verified:
            raise ValueError(
                "Crawl generation changed while being prepared."
            )

        # Discourage accidental mutation of prepared files.
        # File permissions are not a security-grade immutability
        # guarantee against the filesystem owner.
        for file in staging.rglob("*"):
            if file.is_file():
                file.chmod(0o444)

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Generation already exists: {destination}"
            )

        # Atomic visibility on the same filesystem. Nothing
        # replaces any existing published page or report.
        try:
            staging.rename(destination)
        except OSError as exc:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Generation already exists: {destination}"
                ) from exc
            raise

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {
        "manifest_path": destination / "crawl_manifest.json",
        "processed_pages_dir": destination / "processed",
        "report_path": destination / "cleaning_report.json",
    }



def _generation_proof(active_pointer_path, generation_name):
    """Resolve one generation without accepting arbitrary paths."""
    import re

    pointer = Path(active_pointer_path)

    if (
        not isinstance(generation_name, str)
        or re.fullmatch(r"run-[0-9a-f]{64}", generation_name) is None
    ):
        raise ValueError("Invalid generation identifier.")

    generations = pointer.parent / "generations"

    if generations.is_symlink():
        raise ValueError("Generation storage must not be a symlink.")

    directory = generations / generation_name

    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Missing or unsafe generation directory.")

    proof = {
        "manifest_path": directory / "crawl_manifest.json",
        "processed_pages_dir": directory / "processed",
        "report_path": directory / "cleaning_report.json",
    }

    if any(item.is_symlink() for item in proof.values()):
        raise ValueError("Generation paths must not be symlinks.")

    verified = verify_published_crawl_run(**proof)

    expected_name = "run-" + hashlib.sha256(
        verified["run_id"].encode("utf-8")
    ).hexdigest()

    if expected_name != generation_name:
        raise ValueError("Generation identifier does not match its run.")

    return proof


def _publish_active_generation_locked(
    *,
    generation_proof,
    active_pointer_path,
    lock_path,
) -> None:
    """Atomically select an already verified, immutable generation.

    This updates only the active pointer, never published course files.
    All cooperating publishers must use the same coordination lock.
    """
    import json
    import os

    pointer = Path(active_pointer_path)
    lock = Path(lock_path)

    if not pointer.is_absolute() or not lock.is_absolute():
        raise ValueError("Pointer and lock paths must be absolute.")

    if lock != pointer.parent / "refresh.lock":
        raise ValueError("Incorrect shared refresh lock path.")

    if pointer.is_symlink():
        raise ValueError("Active pointer must not be a symlink.")

    if not isinstance(generation_proof, dict):
        raise ValueError("Invalid generation proof.")

    try:
        manifest = Path(generation_proof["manifest_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing generation manifest.") from exc

    generation_name = manifest.parent.name

    expected = _generation_proof(pointer, generation_name)

    for key, expected_path in expected.items():
        try:
            supplied_path = Path(generation_proof[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Incomplete generation proof.") from exc

        if supplied_path != expected_path:
            raise ValueError(
                "Generation proof does not match fixed snapshot paths."
            )

    if pointer.exists() and not pointer.is_file():
        raise ValueError("Active pointer is not a regular file.")

    document = {
        "format_version": 1,
        "generation": generation_name,
    }

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".active-generation-",
            suffix=".tmp",
            dir=pointer.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                (
                    json.dumps(document, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())

        # A same-directory replacement makes the pointer switch
        # atomic to other filesystem readers.
        os.replace(temporary, pointer)

        # Request durability of the updated directory entry.
        descriptor = os.open(
            pointer.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_active_generation(
    *,
    generation_proof,
    active_pointer_path,
    lock_path,
) -> None:
    """Publish a verified generation under the shared refresh lock."""
    from app.ingestion.locking import refresh_lock

    lock = Path(lock_path)

    if not lock.is_absolute():
        raise ValueError("Refresh lock path must be absolute.")

    with refresh_lock(lock):
        _publish_active_generation_locked(
            generation_proof=generation_proof,
            active_pointer_path=active_pointer_path,
            lock_path=lock,
        )


def resolve_active_generation(*, active_pointer_path) -> dict:
    """Return fixed paths for the currently selected generation.

    Callers should retain these returned paths for the entire operation,
    rather than resolving the active pointer again midway through it.
    """
    import json

    pointer = Path(active_pointer_path)

    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("Missing or unsafe active-generation pointer.")

    try:
        document = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid active-generation pointer.") from exc

    if (
        not isinstance(document, dict)
        or set(document) != {"format_version", "generation"}
        or document["format_version"] != 1
        or isinstance(document["format_version"], bool)
    ):
        raise ValueError("Unsupported active-generation pointer.")

    return _generation_proof(
        pointer,
        document["generation"],
    )



def _cleanup_abandoned_processing_workspaces(parent: Path) -> None:
    """Remove only recognizable, owned, abandoned processing workspaces.

    This function runs while the shared refresh lock is held.
    Unknown or unmarked directories are always preserved.
    """
    import os
    import re
    import shutil
    import stat

    marker_name = ".skillsathi-processing-owner"
    expected_marker = b"skillsathi-processing-workspace-v1\\n"

    for entry in parent.iterdir():
        # tempfile.mkdtemp generates an eight-character random suffix.
        if not re.fullmatch(
            r"\.processing-generation-[a-z0-9_]{8}",
            entry.name,
        ):
            continue

        details = entry.lstat()

        if (
            not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_mode & 0o077
        ):
            continue

        marker = entry / marker_name

        try:
            marker_details = marker.lstat()
        except FileNotFoundError:
            continue

        if (
            not stat.S_ISREG(marker_details.st_mode)
            or marker_details.st_uid != os.getuid()
            or marker_details.st_nlink != 1
            or marker_details.st_size != len(expected_marker)
        ):
            continue

        if marker.read_bytes() != expected_marker:
            continue

        # Refuse unexpected top-level content rather than risk
        # deleting unrelated files.
        permitted = {
            marker_name,
            "processed",
            "cleaning_report.json",
        }

        if any(
            child.name not in permitted
            for child in entry.iterdir()
        ):
            continue

        shutil.rmtree(entry)


def process_and_publish_generation(
    *,
    manifest_path,
    raw_pages_dir,
    active_pointer_path,
    lock_path,
    scope=None,
) -> dict:
    """Process and publish one crawl run using one shared lock.

    Processing output is isolated from previous generations.
    The active pointer changes only after generation verification.
    """
    import os
    from tempfile import TemporaryDirectory

    from app.ingestion.locking import refresh_lock
    from app.ingestion.sync import _process_crawl_run_locked

    pointer = Path(active_pointer_path)
    lock = Path(lock_path)

    if not pointer.is_absolute() or not lock.is_absolute():
        raise ValueError(
            "Active pointer and refresh lock must be absolute paths."
        )

    if lock != pointer.parent / "refresh.lock":
        raise ValueError("Incorrect shared refresh lock path.")

    if pointer.is_symlink():
        raise ValueError("Active pointer must not be a symlink.")

    if not pointer.parent.is_dir():
        raise ValueError("Active pointer parent directory is missing.")

    with refresh_lock(lock):
        # The shared lock prevents another cooperating refresh
        # from using these abandoned processing workspaces.
        _cleanup_abandoned_processing_workspaces(pointer.parent)

        # Work entirely in a private temporary directory. Legacy
        # processed files and the active generation stay unchanged.
        with TemporaryDirectory(
            prefix=".processing-generation-",
            dir=pointer.parent,
        ) as directory:
            workspace = Path(directory)

            # Private ownership marker enables safe cleanup after
            # an abrupt process exit. It is created before processing.
            marker_path = workspace / ".skillsathi-processing-owner"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL

            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            descriptor = os.open(marker_path, flags, 0o600)

            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    b"skillsathi-processing-workspace-v1\\n"
                )

            processed_pages_dir = workspace / "processed"
            report_path = workspace / "cleaning_report.json"

            _process_crawl_run_locked(
                manifest_path=manifest_path,
                raw_pages_dir=raw_pages_dir,
                processed_pages_dir=processed_pages_dir,
                report_path=report_path,
                scope=scope,
            )

            # Publish a verified generation directory, but do not
            # change the active pointer until verification succeeds.
            generation = _prepare_or_reuse_verified_generation(
                manifest_path=manifest_path,
                processed_pages_dir=processed_pages_dir,
                report_path=report_path,
                active_pointer_path=pointer,
            )

            # The outer workflow already owns the shared lock.
            # Calling the public function here would double-lock.
            _publish_active_generation_locked(
                generation_proof=generation,
                active_pointer_path=pointer,
                lock_path=lock,
            )

            return generation



def _prepare_or_reuse_verified_generation(
    *,
    manifest_path,
    processed_pages_dir,
    report_path,
    active_pointer_path,
) -> dict:
    """Reuse a completed generation only when its content matches.

    Called only while the outer shared refresh lock is held.
    An existing generation is never overwritten.
    """
    import json

    from app.ingestion.crawler import page_id

    pointer = Path(active_pointer_path)

    candidate = {
        "manifest_path": Path(manifest_path),
        "processed_pages_dir": Path(processed_pages_dir),
        "report_path": Path(report_path),
    }

    candidate_verified = verify_published_crawl_run(
        **candidate
    )

    name = "run-" + hashlib.sha256(
        candidate_verified["run_id"].encode("utf-8")
    ).hexdigest()

    generations = pointer.parent / "generations"
    destination = generations / name

    if not destination.exists() and not destination.is_symlink():
        return prepare_verified_generation(
            **candidate,
            generations_dir=generations,
        )

    # A matching run ID alone is not proof of identical content.
    existing = _generation_proof(pointer, name)

    existing_verified = verify_published_crawl_run(
        **existing
    )

    if existing_verified != candidate_verified:
        raise ValueError(
            "Existing generation does not match the retry."
        )

    def compare_json(first, second, *, volatile=()):
        left = json.loads(
            Path(first).read_text(encoding="utf-8")
        )
        right = json.loads(
            Path(second).read_text(encoding="utf-8")
        )

        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError("Invalid generation metadata.")

        for field in volatile:
            left.pop(field, None)
            right.pop(field, None)

        if left != right:
            raise ValueError(
                "Existing generation metadata differs from retry."
            )

    compare_json(
        candidate["manifest_path"],
        existing["manifest_path"],
    )

    # Processing timestamps can differ on a retry; report contents,
    # page identity, content digests and counts must still match.
    compare_json(
        candidate["report_path"],
        existing["report_path"],
        volatile=("generated_at",),
    )

    for url in candidate_verified["validated_urls"]:
        identifier = page_id(url)

        candidate_markdown = (
            candidate["processed_pages_dir"]
            / (identifier + ".md")
        )
        existing_markdown = (
            existing["processed_pages_dir"]
            / (identifier + ".md")
        )

        if (
            candidate_markdown.read_bytes()
            != existing_markdown.read_bytes()
        ):
            raise ValueError(
                "Existing generation Markdown differs from retry."
            )

        compare_json(
            candidate["processed_pages_dir"]
            / (identifier + ".json"),
            existing["processed_pages_dir"]
            / (identifier + ".json"),
            volatile=("cleaned_at",),
        )

    # A successful comparison does not mutate the old generation.
    return existing
