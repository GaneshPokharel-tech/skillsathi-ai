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
import ipaddress
import socket
import re
import asyncio
import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CrawlScope:
    """Immutable settings for a single, explicitly selected website."""

    base_url: str
    allowed_hosts: frozenset[str]
    content_paths: tuple[str, ...] | None = None


def build_crawl_scope(base_url: str) -> CrawlScope:
    """Validate a public-looking HTTPS domain and define its crawl scope.

    This is syntax/hostname validation, not a DNS or network safety check.
    Actual network requests still require public-IP and redirect protections.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Base URL must be a nonempty HTTPS URL.")

    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Malformed base URL.") from exc

    if parsed.scheme != "https" or not host or not parsed.netloc:
        raise ValueError("Base URL must be HTTPS with a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Base URL cannot contain credentials.")
    if port not in (None, 443):
        raise ValueError("Only the default HTTPS port is allowed.")
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL cannot contain a query or fragment.")

    # Reject IP literals, including loopback, link-local, and cloud metadata IPs.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP-address base URLs are not permitted.")

    # Single-label, local/reserved hosts and ambiguous non-DNS host spellings
    # are not accepted. DNS resolution is checked separately before crawling.
    host = host.lower()
    if (
        not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", host)
        or any(label.startswith("-") or label.endswith("-") for label in host.split("."))
        or host.endswith((".localhost", ".local", ".internal", ".invalid", ".test"))
        or host.startswith("localhost.")
    ):
        raise ValueError("Base URL must use a valid public DNS hostname.")

    normalized_path = parsed.path.rstrip("/") + "/"
    normalized_base = urlunsplit(("https", host, normalized_path, "", ""))
    return CrawlScope(base_url=normalized_base, allowed_hosts=frozenset({host}))


def validate_dns_scope(scope: CrawlScope) -> None:
    """Reject DNS results containing non-public IP addresses.

    This is a preflight check only. Browser-level request restrictions
    are still required to defend against DNS rebinding and SSRF.
    """
    hostname = urlsplit(scope.base_url).hostname

    if not hostname:
        raise ValueError("Missing hostname for DNS validation.")

    try:
        records = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("DNS resolution failed.") from exc

    if not records:
        raise ValueError("DNS resolution returned no addresses.")

    for record in records:
        try:
            family = record[0]

            if family not in (socket.AF_INET, socket.AF_INET6):
                raise ValueError("Unsupported DNS address family.")

            address = ipaddress.ip_address(record[4][0])

        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Invalid DNS resolution result.") from exc

        # Ensure IPv4-mapped IPv6 addresses receive IPv4 checks.
        if isinstance(address, ipaddress.IPv6Address):
            if address.ipv4_mapped is not None:
                address = address.ipv4_mapped

        if not address.is_global:
            raise ValueError(
                f"DNS resolved to a non-public IP: {address}"
            )


@dataclass(frozen=True)
class SiteStoragePaths:
    """Filesystem locations belonging to one website."""

    output_dir: Path
    pages_dir: Path
    urls_file: Path
    homepage_md: Path
    homepage_json: Path


def site_storage_paths(
    scope: CrawlScope,
    *,
    root: Path | None = None,
) -> SiteStoragePaths:
    """Build isolated paths without creating files or directories."""

    host = urlsplit(scope.base_url).hostname

    if (
        not host
        or host not in scope.allowed_hosts
        or not re.fullmatch(r"[a-z0-9-]+(?:[.][a-z0-9-]+)+", host)
    ):
        raise ValueError("Invalid hostname for website storage.")

    base = Path(root) if root is not None else OUTPUT_DIR
    output_dir = base / host

    return SiteStoragePaths(
        output_dir=output_dir,
        pages_dir=output_dir / "pages",
        urls_file=output_dir / "discovered_urls.json",
        homepage_md=output_dir / "homepage.md",
        homepage_json=output_dir / "homepage.json",
    )


def validate_public_url(url: str, scope: CrawlScope | None = None) -> None:
    """Reject URLs outside the explicitly approved HTTPS hostnames."""
    parsed = urlsplit(url)

    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs are allowed: {url}")
    if parsed.hostname not in (scope.allowed_hosts if scope else ALLOWED_HOSTS):
        raise ValueError(f"Domain is not allowed: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"URLs with credentials are not allowed: {url}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port: {url}") from exc
    if port not in (None, 443):
        raise ValueError(f"Port is not allowed: {url}")


def normalize_content_url(href: str, *, scope: CrawlScope | None = None) -> str | None:
    """Normalize same-scope HTTPS links, with optional legacy policy."""
    if not isinstance(href, str) or not href.strip():
        return None

    active_scope = scope or CrawlScope(
        base_url=BASE_URL,
        allowed_hosts=frozenset(ALLOWED_HOSTS),
        content_paths=CONTENT_PATHS,
    )
    try:
        absolute = urljoin(active_scope.base_url, href)
        validate_public_url(absolute, scope=active_scope)
        parsed = urlsplit(absolute)
    except ValueError:
        return None

    path = parsed.path.rstrip("/") or "/"
    if active_scope.content_paths is not None:
        if path != "/" and not any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in active_scope.content_paths
        ):
            return None

    # Without explicit content paths a custom scope stays on its host.
    # Page type/privacy rules and DNS safety are enforced before network I/O.
    return urlunsplit(("https", parsed.hostname, path, "", ""))


def discover_public_urls(links: list[dict], *, scope: CrawlScope | None = None) -> list[str]:
    """Extract safe, deduplicated internal content URLs."""
    discovered = set()
    for link in links:
        url = normalize_content_url(link.get("href", ""), scope=scope)
        if url:
            discovered.add(url)
    return sorted(discovered)


def prepare_urls(urls: list[str], limit: int, *, scope: CrawlScope | None = None) -> list[str]:
    """Prefer individual course pages when selecting a limited crawl batch."""
    if limit < 1:
        raise ValueError("Page limit must be positive.")

    approved = set()
    for candidate in urls:
        url = normalize_content_url(candidate, scope=scope)
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def install_request_guard(context, scope: CrawlScope) -> None:
    """Restrict intercepted browser requests to the approved HTTPS host.

    This is URL-level filtering, not complete SSRF protection.
    DNS rebinding and network-level access require separate controls.
    """

    async def handle_request(route) -> None:
        try:
            validate_public_url(route.request.url, scope=scope)
        except (ValueError, TypeError):
            await route.abort()
        else:
            await route.continue_()

    await context.route("**/*", handle_request)


def configure_browser_request_guard(
    browser: AsyncWebCrawler,
    scope: CrawlScope,
) -> None:
    """Register request filtering before starting the browser."""

    async def on_page_context_created(page, context, **kwargs):
        await install_request_guard(context, scope)
        return page

    browser.crawler_strategy.set_hook(
        "on_page_context_created",
        on_page_context_created,
    )


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


async def _crawl_homepage_locked(*, scope: CrawlScope | None = None) -> None:
    """Crawl homepage, save Markdown/metadata, discover allowed content URLs."""
    active_scope = scope or CrawlScope(
        base_url=BASE_URL,
        allowed_hosts=frozenset(ALLOWED_HOSTS),
        content_paths=CONTENT_PATHS,
    )
    validate_public_url(active_scope.base_url, scope=active_scope)
    validate_dns_scope(active_scope)
    storage = (
        site_storage_paths(active_scope, root=OUTPUT_DIR)
        if scope is not None else None
    )
    output_dir = storage.output_dir if storage else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting crawl: {active_scope.base_url}")

    browser = AsyncWebCrawler(
        config=BrowserConfig(headless=True)
    )
    configure_browser_request_guard(browser, active_scope)

    async with browser as crawler:
        result = await crawler.arun(url=active_scope.base_url, config=_run_config(content_only=False))

    if not result.success:
        raise RuntimeError(f"Website crawling failed: {result.error_message}")
    validate_public_url(result.url, scope=active_scope)
    content = _extract_markdown(result)

    markdown_path = output_dir / "homepage.md"
    markdown_path.write_text(content, encoding="utf-8")

    metadata_path = output_dir / "homepage.json"
    _save_json(
        metadata_path,
        {
            "source_url": result.url,
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "characters": len(content),
            "crawl_success": True,
        },
    )

    urls = discover_public_urls((result.links or {}).get("internal", []), scope=active_scope)
    links_path = storage.urls_file if storage else URLS_FILE
    _save_json(links_path, urls)

    print(f"Markdown saved: {markdown_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Characters extracted: {len(content)}")
    print(f"Discovered content URLs: {len(urls)}")
    print(f"Discovered URLs saved: {links_path}")


async def _crawl_pages_locked(*, scope: CrawlScope | None = None) -> None:
    """Crawl the top approved pages sequentially and save content/metadata."""
    active_scope = scope or CrawlScope(
        base_url=BASE_URL,
        allowed_hosts=frozenset(ALLOWED_HOSTS),
        content_paths=CONTENT_PATHS,
    )
    validate_public_url(active_scope.base_url, scope=active_scope)
    validate_dns_scope(active_scope)

    storage = (
        site_storage_paths(active_scope, root=OUTPUT_DIR)
        if scope is not None else None
    )
    urls_file = storage.urls_file if storage else URLS_FILE
    pages_dir = storage.pages_dir if storage else PAGES_DIR

    if not urls_file.exists():
        raise FileNotFoundError("Run the homepage crawler first.")

    urls_data = json.loads(urls_file.read_text(encoding="utf-8"))
    if not isinstance(urls_data, list):
        raise ValueError("Invalid discovered URLs file.")

    urls = prepare_urls(urls_data, limit=MAX_PAGES, scope=active_scope)
    pages_dir.mkdir(parents=True, exist_ok=True)

    from uuid import uuid4

    run_id = uuid4().hex
    manifest_path = pages_dir.parent / "crawl_manifest.json"
    outcomes = []

    # Write a new manifest before crawling. An interrupted run must
    # never leave the previous run marked as the current complete run.
    _save_json(manifest_path, {
        "run_id": run_id,
        "expected_urls": urls,
        "status": "running",
        "pages": [],
    })

    successful = failed = 0

    print(f"Starting crawl of {len(urls)} pages")
    browser = AsyncWebCrawler(
        config=BrowserConfig(headless=True)
    )
    configure_browser_request_guard(browser, active_scope)

    async with browser as crawler:
        for index, url in enumerate(urls, start=1):
            print(f"\n[{index}/{len(urls)}] {url}")
            try:
                result = await crawler.arun(url=url, config=_run_config(content_only=True))
                if not result.success:
                    raise RuntimeError(result.error_message or "Unknown error")

                final_url = normalize_content_url(result.url, scope=active_scope)
                if not final_url:
                    raise ValueError("Crawl redirected outside approved content.")

                content = _extract_markdown(result)
                identifier = page_id(final_url)
                markdown_path = pages_dir / f"{identifier}.md"
                markdown_path.write_text(content, encoding="utf-8")

                metadata_path = pages_dir / f"{identifier}.json"
                _save_json(
                    metadata_path,
                    {
                        "source_url": final_url,
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "characters": len(content),
                        "content_hash": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "run_id": run_id,
                    },
                )

                outcomes.append({
                    "run_id": run_id,
                    "source_url": url,
                    "status": "success",
                    "final_url": final_url,
                })

                successful += 1
                print(f"Saved: {len(content)} characters")
            except Exception as exc:
                failed += 1

                outcomes.append({
                    "run_id": run_id,
                    "source_url": url,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                })

                print(f"Failed: {exc}")

            if index < len(urls):
                await asyncio.sleep(REQUEST_DELAY)

    _save_json(manifest_path, {
        "run_id": run_id,
        "expected_urls": urls,
        "status": (
            "complete"
            if urls and failed == 0 and successful == len(urls)
            else "incomplete"
        ),
        "pages": outcomes,
        "attempted": len(urls),
        "successful": successful,
        "failed": failed,
    })

    print("\n" + "=" * 45)
    print("CRAWLING SUMMARY")
    print("=" * 45)
    print(f"Attempted: {len(urls)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {pages_dir}")




def _crawler_refresh_lock_path(lock_path):
    """Validate the coordination path before crawler I/O."""
    from pathlib import Path

    if lock_path is None:
        raise ValueError("A shared refresh lock path is required.")

    try:
        path = Path(lock_path)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid refresh lock path.") from exc

    if not path.is_absolute():
        raise ValueError("Refresh lock path must be absolute.")

    return path


async def crawl_homepage(*, scope: CrawlScope | None = None, lock_path=None) -> None:
    """Crawl homepage while holding the shared refresh lock."""
    from app.ingestion.locking import refresh_lock

    path = _crawler_refresh_lock_path(lock_path)

    with refresh_lock(path):
        await _crawl_homepage_locked(scope=scope)


async def crawl_pages(*, scope: CrawlScope | None = None, lock_path=None) -> None:
    """Crawl selected pages while holding the shared refresh lock."""
    from app.ingestion.locking import refresh_lock

    path = _crawler_refresh_lock_path(lock_path)

    with refresh_lock(path):
        await _crawl_pages_locked(scope=scope)

def main() -> None:
    from app.ingestion.locking import DEFAULT_REFRESH_LOCK_PATH

    parser = argparse.ArgumentParser(description="SkillSathi website ingestion")
    parser.add_argument(
        "--pages", action="store_true", help="Crawl discovered pages (default: homepage)"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="HTTPS base URL to crawl (default: SkillShikshya)",
    )
    args = parser.parse_args()

    scope = build_crawl_scope(args.url) if args.url else None

    asyncio.run(
        crawl_pages(scope=scope, lock_path=DEFAULT_REFRESH_LOCK_PATH)
        if args.pages
        else crawl_homepage(scope=scope, lock_path=DEFAULT_REFRESH_LOCK_PATH)
    )


if __name__ == "__main__":
    main()
