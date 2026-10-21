"""Course discovery from indexed SkillShikshya evidence."""

from urllib.parse import urlsplit

from app.rag.engine import COLLECTION_NAME, QDRANT_PATH


def _course_url(value: object) -> str | None:
    """Accept only canonical SkillShikshya course URLs."""
    if not isinstance(value, str):
        return None

    try:
        url = urlsplit(value.strip())

        if url.scheme != "https":
            return None

        if url.hostname not in (
            "skillshikshya.com",
            "www.skillshikshya.com",
        ):
            return None

        if url.username or url.password:
            return None

        if url.port not in (None, 443):
            return None

        path = url.path.rstrip("/")

        if not path.startswith("/courses/"):
            return None

        if not path[len("/courses/"):]:
            return None

        return "https://skillshikshya.com" + path

    except ValueError:
        return None


def list_courses(
    *,
    query: str | None = None,
    client=None,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, str]]:
    """List unique courses actually present in the vector database.

    This is an indexed-course catalog, not the entire live website.
    """
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a string or None.")

    from qdrant_client import QdrantClient

    owns_client = client is None

    if owns_client:
        if not QDRANT_PATH.is_dir():
            raise FileNotFoundError("Qdrant database not found.")
        client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not client.collection_exists(collection_name):
            raise ValueError(
                f"Qdrant collection not found: {collection_name}"
            )

        courses = {}
        offset = None
        scanned = 0

        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            scanned += len(records)

            if scanned > 10000:
                raise RuntimeError(
                    "Course catalog scan limit exceeded."
                )

            for record in records:
                payload = record.payload or {}

                if not isinstance(payload, dict):
                    continue

                url = _course_url(payload.get("source_url"))

                if not url:
                    continue

                title = payload.get("document_title")

                if not isinstance(title, str) or not title.strip():
                    title = (
                        url.rsplit("/", 1)[-1]
                        .replace("-", " ")
                        .title()
                    )

                courses.setdefault(
                    url,
                    {
                        "title": title.strip(),
                        "source_url": url,
                    },
                )

            if next_offset is None:
                break

            if next_offset == offset:
                raise RuntimeError(
                    "Qdrant pagination did not advance."
                )

            offset = next_offset

        results = sorted(
            courses.values(),
            key=lambda item: item["title"].casefold(),
        )

        if query and query.strip():
            search = query.strip().casefold()
            results = [
                course
                for course in results
                if search in course["title"].casefold()
                or search in course["source_url"].casefold()
            ]

        return results

    finally:
        if owns_client:
            client.close()



def get_course_details(
    course_url: str,
    *,
    client=None,
    collection_name: str = COLLECTION_NAME,
    max_chunks: int = 8,
) -> dict | None:
    """Retrieve evidence belonging to one exact indexed course.

    Returns None when no indexed evidence exists for that course.
    This is not a complete or live course syllabus.
    """
    canonical_url = _course_url(course_url)

    if canonical_url is None:
        raise ValueError("A valid SkillShikshya course URL is required.")

    if (
        isinstance(max_chunks, bool)
        or not isinstance(max_chunks, int)
        or not 1 <= max_chunks <= 20
    ):
        raise ValueError("max_chunks must be an integer from 1 to 20.")

    from qdrant_client import QdrantClient, models

    owns_client = client is None

    if owns_client:
        if not QDRANT_PATH.is_dir():
            raise FileNotFoundError("Qdrant database not found.")
        client = QdrantClient(path=str(QDRANT_PATH))

    try:
        if not client.collection_exists(collection_name):
            raise ValueError("Qdrant collection not found.")

        exact_course_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_url",
                    match=models.MatchValue(value=canonical_url),
                )
            ]
        )

        evidence = []
        seen_ids = set()
        title = None
        offset = None

        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=exact_course_filter,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = record.payload or {}

                if not isinstance(payload, dict):
                    continue

                if _course_url(payload.get("source_url")) != canonical_url:
                    continue

                text = payload.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                chunk_id = str(
                    payload.get("chunk_id")
                    or getattr(record, "id", "")
                )

                if chunk_id and chunk_id in seen_ids:
                    continue

                if chunk_id:
                    seen_ids.add(chunk_id)

                document_title = payload.get("document_title")

                if (
                    title is None
                    and isinstance(document_title, str)
                    and document_title.strip()
                ):
                    title = document_title.strip()

                section = payload.get("section", "")

                evidence.append({
                    "chunk_id": chunk_id,
                    "section": (
                        section.strip()
                        if isinstance(section, str)
                        else ""
                    ),
                    "source_url": canonical_url,
                    "text": text.strip(),
                })

            if len(evidence) > 2048:
                raise RuntimeError("Course evidence scan limit exceeded.")

            if next_offset is None:
                break

            if next_offset == offset:
                raise RuntimeError("Qdrant pagination did not advance.")

            offset = next_offset

        if not evidence:
            return None

        def priority(item: dict) -> tuple:
            section = item["section"].casefold()

            if "curriculum" in section or "syllabus" in section:
                rank = 0
            elif "overview" in section:
                rank = 1
            elif "frequently asked" in section:
                rank = 2
            else:
                rank = 3

            return (
                rank,
                section,
                item["chunk_id"],
            )

        evidence.sort(key=priority)

        if title is None:
            title = (
                canonical_url.rsplit("/", 1)[-1]
                .replace("-", " ")
                .title()
            )

        return {
            "title": title,
            "source_url": canonical_url,
            "total_chunks": len(evidence),
            "evidence": evidence[:max_chunks],
            "truncated": len(evidence) > max_chunks,
        }

    finally:
        if owns_client:
            client.close()



def compare_courses(
    first_url: str,
    second_url: str,
    *,
    client=None,
    collection_name: str = COLLECTION_NAME,
    max_chunks_per_course: int = 4,
) -> dict:
    """Prepare source-grounded evidence for comparing two courses.

    Does not generate recommendations or invent missing facts.
    """
    first = _course_url(first_url)
    second = _course_url(second_url)

    if first is None or second is None:
        raise ValueError("Both URLs must be valid SkillShikshya course URLs.")

    if first == second:
        raise ValueError("Choose two different courses.")

    if (
        isinstance(max_chunks_per_course, bool)
        or not isinstance(max_chunks_per_course, int)
        or not 1 <= max_chunks_per_course <= 10
    ):
        raise ValueError("max_chunks_per_course must be between 1 and 10.")

    from qdrant_client import QdrantClient

    owns_client = client is None

    if owns_client:
        if not QDRANT_PATH.is_dir():
            raise FileNotFoundError("Qdrant database not found.")
        client = QdrantClient(path=str(QDRANT_PATH))

    try:
        details = [
            get_course_details(
                url,
                client=client,
                collection_name=collection_name,
                max_chunks=max_chunks_per_course,
            )
            for url in (first, second)
        ]

        available = [
            detail for detail in details
            if detail is not None
        ]

        missing = [
            url
            for url, detail in zip((first, second), details)
            if detail is None
        ]

        return {
            "comparison_ready": len(available) == 2,
            "courses": available,
            "missing_courses": missing,
            "notice": (
                "Evidence comes from indexed website content, "
                "not a live verified course catalog."
            ),
        }

    finally:
        if owns_client:
            client.close()
