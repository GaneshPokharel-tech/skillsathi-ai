import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.ingestion.cleaner import clean_markdown
from app.ingestion.url_discovery import normalize_content_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PAGES_DIR = PROJECT_ROOT / "data" / "raw" / "pages"
PROCESSED_PAGES_DIR = PROJECT_ROOT / "data" / "processed" / "pages"
CLEANING_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_report.json"


def _process_page(markdown_path: Path) -> dict[str, Any]:
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
            normalize_content_url(source_url) if isinstance(source_url, str) else None
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

        PROCESSED_PAGES_DIR.mkdir(parents=True, exist_ok=True)
        (PROCESSED_PAGES_DIR / markdown_path.name).write_text(
            cleaned_content,
            encoding="utf-8",
        )
        (PROCESSED_PAGES_DIR / metadata_path.name).write_text(
            json.dumps(output_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result["status"] = "processed"
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
        return result


def clean_pages() -> dict[str, Any]:
    """Clean approved raw Markdown pages and write a per-page processing report."""

    page_results = [
        _process_page(path) for path in sorted(RAW_PAGES_DIR.glob("*.md"))
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed": sum(page["status"] == "processed" for page in page_results),
        "skipped": sum(page["status"] == "skipped" for page in page_results),
        "failed": sum(page["status"] == "failed" for page in page_results),
        "pages": page_results,
    }

    CLEANING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    cleaning_report = clean_pages()
    print(
        "Cleaning complete: "
        f"processed={cleaning_report['processed']}, "
        f"skipped={cleaning_report['skipped']}, "
        f"failed={cleaning_report['failed']}"
    )
    print(f"Report: {CLEANING_REPORT_PATH}")
