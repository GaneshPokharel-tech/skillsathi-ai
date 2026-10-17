import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)

from app.ingestion.url_discovery import discover_public_urls

# ==========================================
# CONFIGURATION
# ==========================================

BASE_URL = "https://skillshikshya.com/"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

ALLOWED_HOSTS = {
    "skillshikshya.com",
    "www.skillshikshya.com",
}


# ==========================================
# URL VALIDATION
# ==========================================


def validate_public_url(url: str) -> None:
    """
    Validate that a URL belongs to the
    approved SkillShikshya website.
    """

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs are allowed: {url}")

    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Domain is not allowed: {url}")

    if parsed.username or parsed.password:
        raise ValueError(f"URLs with credentials are not allowed: {url}")

    try:
        if parsed.port not in (None, 443):
            raise ValueError(f"Port is not allowed: {url}")
    except ValueError as exc:
        raise ValueError(f"Invalid or disallowed port: {url}") from exc


# ==========================================
# HOMEPAGE CRAWLER
# ==========================================


async def crawl_homepage() -> None:
    """
    Crawl SkillShikshya homepage.

    1. Validate URL
    2. Extract Markdown
    3. Save Markdown
    4. Save metadata
    5. Discover internal content URLs
    6. Save discovered URLs
    """

    validate_public_url(BASE_URL)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    browser_config = BrowserConfig(
        headless=True,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True,
        exclude_external_links=True,
        page_timeout=30000,
    )

    print("=" * 50)
    print("SkillSathi AI - Website Ingestion")
    print("=" * 50)

    print(f"Starting crawl: {BASE_URL}")

    # --------------------------------------
    # STEP 1: CRAWL WEBSITE
    # --------------------------------------

    async with AsyncWebCrawler(config=browser_config) as crawler:

        result = await crawler.arun(
            url=BASE_URL,
            config=run_config,
        )

    if not result.success:
        raise RuntimeError(f"Website crawling failed: " f"{result.error_message}")

    # Validate final URL after any redirects
    validate_public_url(result.url)

    print("Homepage crawled successfully!")

    # --------------------------------------
    # STEP 2: EXTRACT MARKDOWN
    # --------------------------------------

    markdown = result.markdown

    if isinstance(markdown, str):
        content = markdown
    elif markdown is not None:
        content = markdown.raw_markdown
    else:
        content = ""

    if not content or not content.strip():
        raise RuntimeError("No Markdown content extracted.")

    print(f"Characters extracted: {len(content)}")

    # --------------------------------------
    # STEP 3: SAVE MARKDOWN
    # --------------------------------------

    markdown_path = OUTPUT_DIR / "homepage.md"

    markdown_path.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Markdown saved: {markdown_path}")

    # --------------------------------------
    # STEP 4: SAVE METADATA
    # --------------------------------------

    metadata = {
        "source_url": result.url,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "characters": len(content),
        "crawl_success": True,
    }

    metadata_path = OUTPUT_DIR / "homepage.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Metadata saved: {metadata_path}")

    # --------------------------------------
    # STEP 5: DISCOVER INTERNAL URLS
    # --------------------------------------

    internal_links = (result.links or {}).get("internal", [])

    discovered_urls = discover_public_urls(internal_links)

    print(f"Discovered content URLs: " f"{len(discovered_urls)}")

    # --------------------------------------
    # STEP 6: SAVE DISCOVERED URLS
    # --------------------------------------

    links_path = OUTPUT_DIR / "discovered_urls.json"

    links_path.write_text(
        json.dumps(
            discovered_urls,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Discovered URLs saved: {links_path}")

    # --------------------------------------
    # COMPLETION
    # --------------------------------------

    print("=" * 50)
    print("Homepage ingestion completed!")
    print("=" * 50)


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    asyncio.run(crawl_homepage())
