"""Markdown cleaning and page processing for SkillSathi AI.

Run the local cleaning pipeline with:
    uv run python -m app.ingestion.processor

Preserves the existing cleaning behavior and metadata/report format.
"""

import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.ingestion.crawler import normalize_content_url


_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:\\.|[^)])*\)")
_REFERENCE_IMAGE_RE = re.compile(r"!\[([^\]]*)\](?:\[[^\]]*\])?")
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
_ANY_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]+\)")
_ESCAPED_BLOCK_MARKER_RE = re.compile(r"^(\s*)\\(#{1,6}|[*+-])(?=\s)")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def _is_link_only_navigation(line: str) -> bool:
    links = _LINK_RE.findall(line)
    if len(links) < 2:
        return False

    remainder = _LINK_RE.sub("", line)
    return not remainder.strip(" \t|·•/")


def _is_link_dominant_navigation(line: str) -> bool:
    links = _ANY_LINK_RE.findall(line)
    if not links:
        return False

    remainder = _ANY_LINK_RE.sub("", line).strip(" \t|·•/")
    if len(remainder) > 8:
        return False

    if len(links) > 1:
        return True

    destination = re.search(r"\]\(([^)]+)\)", links[0])
    if not destination:
        return False

    path = urlsplit(destination.group(1)).path.rstrip("/")
    return path in (
        "",
        "/courses",
        "/enroll-form",
        "/corporate-training",
        "/online-counselling",
    )


def _looks_like_content(line: str) -> bool:
    stripped = line.strip()
    text_without_links = _ANY_LINK_RE.sub("", stripped)
    return (
        not stripped
        or stripped.startswith(("#", "-", "*", "+", "|", ">"))
        or ":" in text_without_links
        or any(character.isdigit() for character in stripped)
        or "`" in stripped
        or (
            _ANY_LINK_RE.search(stripped) is not None
            and not _is_link_dominant_navigation(stripped)
        )
        or (len(stripped) >= 40 and not _is_link_dominant_navigation(stripped))
    )


def _remove_leading_navigation(lines: list[str]) -> list[str]:
    first_heading = next(
        (index for index, line in enumerate(lines) if _HEADING_RE.match(line)),
        len(lines),
    )
    prefix = lines[:first_heading]
    if not any(
        len(_LINK_RE.findall(line)) >= 2 and _is_link_only_navigation(line)
        for line in prefix
    ):
        return lines

    content_anchor = next(
        (
            index
            for index in range(first_heading - 1, -1, -1)
            if len(lines[index].strip()) >= 40
            and not _is_link_only_navigation(lines[index].strip())
        ),
        first_heading,
    )

    return [
        line
        for index, line in enumerate(lines)
        if index >= content_anchor
        or index >= first_heading
        or (
            _looks_like_content(line)
            and not _is_link_dominant_navigation(line.strip())
        )
    ]


def clean_markdown(content: str) -> str:
    """Clean crawler Markdown while preserving its document structure."""

    if not isinstance(content, str):
        raise TypeError("Markdown content must be a string.")

    if not content.strip():
        return ""

    content = html.unescape(content).replace("\r\n", "\n").replace("\r", "\n")
    content = content.replace("\xa0", " ")
    content = _IMAGE_RE.sub(r"\1", content)
    content = _REFERENCE_IMAGE_RE.sub(r"\1", content)

    lines = _remove_leading_navigation(content.split("\n"))
    link_only_counts = Counter(
        line.strip()
        for line in lines
        if line.strip() and _is_link_only_navigation(line.strip())
    )

    cleaned_lines = []
    in_fenced_block = False

    for line in lines:
        fence = _FENCE_RE.match(line)
        if fence:
            in_fenced_block = not in_fenced_block

        line = line.rstrip()

        if not in_fenced_block:
            line = _ESCAPED_BLOCK_MARKER_RE.sub(r"\1\2", line)
            if (
                line.strip()
                and link_only_counts[line.strip()] > 1
                and _is_link_only_navigation(line.strip())
            ):
                continue

        if not line.strip() and cleaned_lines and not cleaned_lines[-1]:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# Page-processing paths and reporting
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PAGES_DIR = PROJECT_ROOT / "data" / "raw" / "pages"
PROCESSED_PAGES_DIR = PROJECT_ROOT / "data" / "processed" / "pages"
CLEANING_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_report.json"


def _process_page(
    markdown_path: Path,
    *,
    scope=None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": markdown_path.name,
        "status": "failed",
        "raw_character_count": None,
        "cleaned_character_count": None,
    }

    try:
        raw_content = markdown_path.read_text(encoding="utf-8")
        result["raw_character_count"] = len(raw_content)

        metadata_path = markdown_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Page metadata must be a JSON object.")

        if not isinstance(metadata.get("crawled_at"), str) or not metadata[
            "crawled_at"
        ].strip():
            raise ValueError("Page metadata must include a crawl timestamp.")

        source_url = metadata.get("source_url")
        approved_url = (
            normalize_content_url(source_url, scope=scope) if isinstance(source_url, str) else None
        )
        if not approved_url or urlsplit(approved_url).path == "/":
            result.update(status="skipped", reason="Source URL is not approved content.")
            return result

        cleaned_content = clean_markdown(raw_content)
        result["cleaned_character_count"] = len(cleaned_content)

        if not cleaned_content:
            result.update(status="skipped", reason="Page contains no cleanable content.")
            return result

        cleaned_at = datetime.now(timezone.utc).isoformat()
        output_metadata = {
            **metadata,
            "source_url": approved_url,
            "cleaned_at": cleaned_at,
            "raw_character_count": len(raw_content),
            "cleaned_character_count": len(cleaned_content),
            "content_hash": hashlib.sha256(
                cleaned_content.encode("utf-8")
            ).hexdigest(),
        }

        destination = output_dir if output_dir is not None else PROCESSED_PAGES_DIR
        destination.mkdir(parents=True, exist_ok=True)
        (destination / markdown_path.name).write_text(
            cleaned_content,
            encoding="utf-8",
        )
        (destination / metadata_path.name).write_text(
            json.dumps(output_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["status"] = "processed"
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
        return result


def clean_pages(*, scope=None) -> dict[str, Any]:
    """Clean approved raw Markdown pages and write a per-page processing report."""

    if scope is None:
        # Preserve existing SkillShikshya processing behavior.
        raw_dir = RAW_PAGES_DIR
        report_path = CLEANING_REPORT_PATH
        page_results = [
            _process_page(path)
            for path in sorted(raw_dir.glob("*.md"))
        ]
    else:
        from app.ingestion.crawler import site_storage_paths

        raw_dir = site_storage_paths(
            scope, root=RAW_PAGES_DIR.parent
        ).pages_dir

        processed_storage = site_storage_paths(
            scope, root=PROCESSED_PAGES_DIR.parent
        )
        output_dir = processed_storage.pages_dir
        report_path = (
            processed_storage.output_dir / "cleaning_report.json"
        )

        page_results = [
            _process_page(path, scope=scope, output_dir=output_dir)
            for path in sorted(raw_dir.glob("*.md"))
        ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed": sum(page["status"] == "processed" for page in page_results),
        "skipped": sum(page["status"] == "skipped" for page in page_results),
        "failed": sum(page["status"] == "failed" for page in page_results),
        "pages": page_results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """Run cleaning for the selected website."""
    import argparse
    from app.ingestion.crawler import (
        build_crawl_scope,
        site_storage_paths,
    )

    parser = argparse.ArgumentParser(
        description="SkillSathi website content processing"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="HTTPS website URL (default: legacy SkillShikshya data)",
    )
    args = parser.parse_args(argv)

    scope = build_crawl_scope(args.url) if args.url else None
    report = clean_pages(scope=scope)

    report_path = (
        site_storage_paths(
            scope,
            root=PROCESSED_PAGES_DIR.parent,
        ).output_dir / "cleaning_report.json"
        if scope is not None
        else CLEANING_REPORT_PATH
    )

    print(
        "Cleaning complete: "
        f"processed={report['processed']}, "
        f"skipped={report['skipped']}, "
        f"failed={report['failed']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
