"""SkillSathi AI: crawl the approved public SkillShikshya website.

Run homepage discovery:
    uv run python -m app.ingestion.crawler

Run the prioritized sample of discovered pages:
    uv run python -m app.ingestion.crawler --pages

This module combines the earlier crawler.py, multi_crawler.py, and
url_discovery.py without changing their public function signatures or
saved data locations.
"""

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

# Current ingestion scope: one approved public website.
# Configurable-domain ingestion will be implemented in a separate step.
BASE_URL = "https://skillshikshya.com/"
ALLOWED_HOSTS = {"skillshikshya.com", "www.skillshikshya.com"}
CONTENT_PATHS = ("/courses", "/faq", "/about", "/blogs", "/contact-us")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = OUTPUT_DIR  # Backward-compatible alias
PAGES_DIR = OUTPUT_DIR / "pages"
URLS_FILE = OUTPUT_DIR / "discovered_urls.json"

MAX_PAGES = 5
REQUEST_DELAY = 2


def validate_public_url(url: str) -> None:
    """Reject URLs outside the explicitly approved HTTPS hostnames."""
    parsed = urlsplit(url)

    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs are allowed: {url}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Domain is not allowed: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"URLs with credentials are not allowed: {url}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port: {url}") from exc
    if port not in (None, 443):
        raise ValueError(f"Port is not allowed: {url}")


def normalize_content_url(href: str) -> str | None:
    """Normalize a permitted content URL; return None for disallowed URLs."""
    if not isinstance(href, str) or not href:
        return None

    try:
        absolute = urljoin(BASE_URL, href)
        validate_public_url(absolute)
        parsed = urlsplit(absolute)
    except ValueError:
        return None

    path = parsed.path.rstrip("/") or "/"
    if path != "/" and not any(
        path == prefix or path.startswith(prefix + "/") for prefix in CONTENT_PATHS
    ):
        return None

    # Discard fragments and query parameters for consistent page identities.
    return urlunsplit(("https", parsed.hostname, path, "", ""))


def discover_public_urls(links: list[dict]) -> list[str]:
    """Extract safe, deduplicated internal content URLs."""
    discovered = set()
    for link in links:
        url = normalize_content_url(link.get("href", ""))
        if url:
            discovered.add(url)
    return sorted(discovered)


def prepare_urls(urls: list[str], limit: int) -> list[str]:
    """Prefer individual course pages when selecting a limited crawl batch."""
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
    """Generate a stable filename identifier for an indexed page."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _extract_markdown(result) -> str:
    markdown = result.markdown
    if isinstance(markdown, str):
        content = markdown
    elif markdown is not None:
        content = markdown.raw_markdown or ""
    else:
        content = ""

    if not content.strip():
        raise ValueError("No Markdown content extracted.")
    return content


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_config(*, content_only: bool) -> CrawlerRunConfig:
    options = dict(
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True,
        exclude_external_links=True,
        page_timeout=30000,
    )
    if content_only:
        options.update(excluded_tags=["nav", "footer", "form"], remove_forms=True)
    return CrawlerRunConfig(**options)


async def crawl_homepage() -> None:
    """Crawl homepage, save Markdown/metadata, discover allowed content URLs."""
    validate_public_url(BASE_URL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting crawl: {BASE_URL}")

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        result = await crawler.arun(
            url=BASE_URL, config=_run_config(content_only=False)
        )

    if not result.success:
        raise RuntimeError(f"Website crawling failed: {result.error_message}")
    validate_public_url(result.url)
    content = _extract_markdown(result)

    markdown_path = OUTPUT_DIR / "homepage.md"
    markdown_path.write_text(content, encoding="utf-8")

    metadata_path = OUTPUT_DIR / "homepage.json"
    _save_json(
        metadata_path,
        {
            "source_url": result.url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "characters": len(content),
            "crawl_success": True,
        },
    )

    urls = discover_public_urls((result.links or {}).get("internal", []))
    links_path = URLS_FILE
    _save_json(links_path, urls)

    print(f"Markdown saved: {markdown_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Characters extracted: {len(content)}")
    print(f"Discovered content URLs: {len(urls)}")
    print(f"Discovered URLs saved: {links_path}")


async def crawl_pages() -> None:
    """Crawl the top approved pages sequentially and save content/metadata."""
    if not URLS_FILE.exists():
        raise FileNotFoundError("Run the homepage crawler first.")

    urls_data = json.loads(URLS_FILE.read_text(encoding="utf-8"))
    if not isinstance(urls_data, list):
        raise ValueError("Invalid discovered URLs file.")

    urls = prepare_urls(urls_data, limit=MAX_PAGES)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    successful = failed = 0

    print(f"Starting crawl of {len(urls)} pages")
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        for index, url in enumerate(urls, start=1):
            print(f"\n[{index}/{len(urls)}] {url}")
            try:
                result = await crawler.arun(
                    url=url, config=_run_config(content_only=True)
                )
                if not result.success:
                    raise RuntimeError(result.error_message or "Unknown error")

                final_url = normalize_content_url(result.url)
                if not final_url:
                    raise ValueError("Crawl redirected outside approved content.")

                content = _extract_markdown(result)
                identifier = page_id(final_url)
                markdown_path = PAGES_DIR / f"{identifier}.md"
                markdown_path.write_text(content, encoding="utf-8")

                metadata_path = PAGES_DIR / f"{identifier}.json"
                _save_json(
                    metadata_path,
                    {
                        "source_url": final_url,
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "characters": len(content),
                        "content_hash": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                    },
                )
                successful += 1
                print(f"Saved: {len(content)} characters")
            except Exception as exc:
                failed += 1
                print(f"Failed: {exc}")

            if index < len(urls):
                await asyncio.sleep(REQUEST_DELAY)

    print("\n" + "=" * 45)
    print("CRAWLING SUMMARY")
    print("=" * 45)
    print(f"Attempted: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {PAGES_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillSathi website ingestion")
    parser.add_argument(
        "--pages",
        action="store_true",
        help="Crawl discovered pages (default: homepage)",
    )
    args = parser.parse_args()
    asyncio.run(crawl_pages() if args.pages else crawl_homepage())


if __name__ == "__main__":
    main()
