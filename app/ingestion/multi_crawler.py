import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)

from app.ingestion.url_discovery import normalize_content_url

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PAGES_DIR = DATA_DIR / "pages"
URLS_FILE = DATA_DIR / "discovered_urls.json"

MAX_PAGES = 5
REQUEST_DELAY = 2


def prepare_urls(urls: list[str], limit: int) -> list[str]:
    """Validate, normalize, deduplicate, and limit URLs."""

    if limit < 1:
        raise ValueError("Page limit must be positive.")

    approved = set()

    for candidate in urls:
        url = normalize_content_url(candidate)

        if url and urlsplit(url).path != "/":
            approved.add(url)

    def priority(url: str) -> int:
        path = urlsplit(url).path

        if path.startswith("/courses/"):
            return 0
        if path == "/faq":
            return 1
        if path == "/courses":
            return 2
        if path == "/about":
            return 3
        if path.startswith("/blogs/"):
            return 4
        return 5

    return sorted(approved, key=lambda url: (priority(url), url))[:limit]


def page_id(url: str) -> str:
    """Generate a stable filename identifier."""

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


async def crawl_pages() -> None:
    """Crawl approved public pages sequentially."""

    if not URLS_FILE.exists():
        raise FileNotFoundError("Run the homepage crawler first.")

    urls_data = json.loads(URLS_FILE.read_text(encoding="utf-8"))

    if not isinstance(urls_data, list):
        raise ValueError("Invalid discovered URLs file.")

    urls = prepare_urls(urls_data, limit=MAX_PAGES)

    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(headless=True)

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True,
        exclude_external_links=True,
        excluded_tags=["nav", "footer", "form"],
        remove_forms=True,
        page_timeout=30000,
    )

    successful = 0
    failed = 0

    print(f"Starting crawl of {len(urls)} pages")

    async with AsyncWebCrawler(config=browser_config) as crawler:

        for index, url in enumerate(urls, start=1):
            print(f"\n[{index}/{len(urls)}] {url}")

            try:
                result = await crawler.arun(
                    url=url,
                    config=run_config,
                )

                if not result.success:
                    raise RuntimeError(result.error_message or "Unknown error")

                # Validate destination after redirect
                final_url = normalize_content_url(result.url)

                if not final_url:
                    raise ValueError("Crawl redirected outside approved content.")

                markdown = result.markdown

                if isinstance(markdown, str):
                    content = markdown
                elif markdown is not None:
                    content = markdown.raw_markdown
                else:
                    content = ""

                if not content.strip():
                    raise ValueError("Empty page content.")

                identifier = page_id(final_url)

                # Save Markdown
                markdown_path = PAGES_DIR / f"{identifier}.md"

                markdown_path.write_text(
                    content,
                    encoding="utf-8",
                )

                # Save metadata
                metadata = {
                    "source_url": final_url,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "characters": len(content),
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }

                metadata_path = PAGES_DIR / f"{identifier}.json"

                metadata_path.write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                successful += 1
                print(f"Saved: {len(content)} characters")

            except Exception as exc:
                failed += 1
                print(f"Failed: {exc}")

            # Avoid sending too many requests
            if index < len(urls):
                await asyncio.sleep(REQUEST_DELAY)

    print("\n" + "=" * 45)
    print("CRAWLING SUMMARY")
    print("=" * 45)
    print(f"Attempted: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {PAGES_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_pages())
