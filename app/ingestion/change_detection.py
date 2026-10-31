"""Safe, pure planning for website content synchronization."""

import re
from collections.abc import Iterable, Mapping

from app.ingestion.crawler import normalize_content_url


def _validate_hashes(values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError("Expected URL-to-hash mapping.")

    validated = {}

    for url, digest in values.items():
        if normalize_content_url(url) != url:
            raise ValueError("Unapproved or noncanonical URL.")

        if (
            not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Invalid SHA-256 content hash.")

        validated[url] = digest

    return validated


def merge_discovered_urls(
    previous_urls: Iterable[str],
    discovered_urls: Iterable[str],
) -> list[str]:
    """Preserve known pages when fresh discovery misses them."""
    merged = set()

    for candidate in (*previous_urls, *discovered_urls):
        canonical = normalize_content_url(candidate)

        if canonical and canonical != "https://skillshikshya.com/":
            merged.add(canonical)

    return sorted(merged)


def plan_content_changes(
    previous: Mapping[str, str],
    fresh: Mapping[str, str],
    *,
    confirmed_removed: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Plan changes without treating missing discovery as deletion.

    confirmed_removed requires independent removal verification
    by the calling sync pipeline.
    """
    previous = _validate_hashes(previous)
    fresh = _validate_hashes(fresh)

    removed = set(confirmed_removed)

    if not removed.issubset(previous) or removed.intersection(fresh):
        raise ValueError("Invalid confirmed removal set.")

    old_urls = set(previous)
    new_urls = set(fresh)
    common = old_urls.intersection(new_urls)

    return {
        "added": sorted(new_urls - old_urls),
        "updated": sorted(
            url for url in common
            if previous[url] != fresh[url]
        ),
        "unchanged": sorted(
            url for url in common
            if previous[url] == fresh[url]
        ),
        "removed": sorted(removed),
        "unverified_missing": sorted(
            old_urls - new_urls - removed
        ),
    }
