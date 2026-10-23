
from pathlib import Path as _CrawlerTestPath
from tempfile import TemporaryDirectory as _CrawlerTestLockDirectory
import unittest

from app.ingestion.crawler import validate_public_url


class TestCrawlerSecurity(unittest.TestCase):

    def test_approved_website(self):
        validate_public_url("https://skillshikshya.com/")

    def test_rejects_external_domain(self):
        with self.assertRaises(ValueError):
            validate_public_url("https://skillshikshya.com.evil.com/")

    def test_rejects_http(self):
        with self.assertRaises(ValueError):
            validate_public_url("http://skillshikshya.com/")


if __name__ == "__main__":
    unittest.main()


class TestConfigurableCrawlerScope(unittest.TestCase):

    def test_accepts_custom_public_base_url(self):
        from app.ingestion.crawler import (
            build_crawl_scope,
            normalize_content_url,
        )

        scope = build_crawl_scope("https://example.org/")

        self.assertEqual(
            normalize_content_url("/courses/python", scope=scope),
            "https://example.org/courses/python",
        )

    def test_blocks_cross_domain_links(self):
        from app.ingestion.crawler import (
            build_crawl_scope,
            normalize_content_url,
        )

        scope = build_crawl_scope("https://example.org/")

        self.assertIsNone(
            normalize_content_url(
                "https://other-site.com/courses/python",
                scope=scope,
            )
        )

    def test_rejects_private_and_local_addresses(self):
        from app.ingestion.crawler import build_crawl_scope

        unsafe_urls = [
            "https://localhost/",
            "https://127.0.0.1/",
            "https://192.168.1.1/",
            "https://169.254.169.254/",
            "https://10.0.0.1/",
        ]

        for url in unsafe_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    build_crawl_scope(url)

    def test_rejects_credentials_and_unsafe_ports(self):
        from app.ingestion.crawler import build_crawl_scope

        unsafe_urls = [
            "https://user:pass@example.org/",
            "https://example.org:8080/",
            "http://example.org/",
        ]

        for url in unsafe_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    build_crawl_scope(url)


class TestScopedDiscovery(unittest.TestCase):

    def test_discovers_only_selected_domain(self):
        from app.ingestion.crawler import (
            build_crawl_scope,
            discover_public_urls,
        )

        scope = build_crawl_scope("https://example.org/")

        links = [
            {"href": "/courses/python"},
            {"href": "https://other-site.com/courses"},
        ]

        result = discover_public_urls(links, scope=scope)

        self.assertEqual(
            result,
            ["https://example.org/courses/python"],
        )

    def test_prepares_custom_domain_urls(self):
        from app.ingestion.crawler import (
            build_crawl_scope,
            prepare_urls,
        )

        scope = build_crawl_scope("https://example.org/")

        result = prepare_urls(
            [
                "https://example.org/courses/python",
                "https://other-site.com/courses",
            ],
            limit=5,
            scope=scope,
        )

        self.assertEqual(
            result,
            ["https://example.org/courses/python"],
        )


class TestScopedCrawlExecution(unittest.IsolatedAsyncioTestCase):

    async def test_crawls_selected_website(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch
        from app.ingestion import crawler

        scope = crawler.build_crawl_scope(
            "https://example.org/"
        )

        response = SimpleNamespace(
            success=True,
            url="https://example.org/",
            markdown="# Courses\n\nLearn new skills.",
            links={
                "internal": [
                    {"href": "/courses/python"},
                    {"href": "https://evil.com/private"},
                ]
            },
        )

        worker = MagicMock()
        worker.arun = AsyncMock(return_value=response)

        browser = MagicMock()
        browser.__aenter__ = AsyncMock(
            return_value=worker
        )
        browser.__aexit__ = AsyncMock(
            return_value=False
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            links_file = output / "example.org" / "discovered_urls.json"

            with (
                patch.object(crawler, "validate_dns_scope", return_value=None),
                patch.object(
                    crawler,
                    "AsyncWebCrawler",
                    return_value=browser,
                ),
                patch.object(crawler, "OUTPUT_DIR", output),
                patch.object(crawler, "URLS_FILE", links_file),
            ):
                with _CrawlerTestLockDirectory() as _lock_dir:
                    await crawler.crawl_homepage(scope=scope, lock_path=_CrawlerTestPath(_lock_dir) / 'refresh.lock')

            discovered = json.loads(
                links_file.read_text(encoding="utf-8")
            )

        self.assertEqual(
            worker.arun.await_args.kwargs["url"],
            "https://example.org/",
        )

        self.assertEqual(
            discovered,
            ["https://example.org/courses/python"],
        )


class TestScopedPageExecution(unittest.IsolatedAsyncioTestCase):

    async def test_crawls_only_selected_domain_pages(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch
        from app.ingestion import crawler

        scope = crawler.build_crawl_scope("https://example.org/")

        response = SimpleNamespace(
            success=True,
            url="https://example.org/courses/python",
            markdown="# Python Course\n\nLearn Python.",
        )

        worker = MagicMock()
        worker.arun = AsyncMock(return_value=response)

        browser = MagicMock()
        browser.__aenter__ = AsyncMock(return_value=worker)
        browser.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls_file = root / "example.org" / "discovered_urls.json"
            pages_dir = root / "example.org" / "pages"

            urls_file.parent.mkdir(parents=True, exist_ok=True)
            urls_file.write_text(
                json.dumps([
                    "https://example.org/courses/python",
                    "https://skillshikshya.com/courses/other",
                    "https://evil.com/private",
                ]),
                encoding="utf-8",
            )

            with (
                patch.object(crawler, "validate_dns_scope", return_value=None),
                patch.object(crawler, "AsyncWebCrawler", return_value=browser),
                patch.object(crawler, "OUTPUT_DIR", root),
                patch.object(crawler, "URLS_FILE", urls_file),
                patch.object(crawler, "PAGES_DIR", pages_dir),
                patch.object(crawler, "REQUEST_DELAY", 0),
            ):
                with _CrawlerTestLockDirectory() as _lock_dir:
                    await crawler.crawl_pages(scope=scope, lock_path=_CrawlerTestPath(_lock_dir) / 'refresh.lock')

            saved_pages = list(pages_dir.glob("*.json"))

        self.assertEqual(worker.arun.await_count, 1)
        self.assertEqual(
            worker.arun.await_args.kwargs["url"],
            "https://example.org/courses/python",
        )
        self.assertEqual(len(saved_pages), 1)


class TestDNSSecurity(unittest.TestCase):

    def test_accepts_public_dns_resolution(self):
        import socket
        from unittest.mock import patch
        from app.ingestion.crawler import (
            build_crawl_scope,
            validate_dns_scope,
        )

        scope = build_crawl_scope("https://example.org/")
        records = [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("8.8.8.8", 443),
            )
        ]

        with patch("socket.getaddrinfo", return_value=records):
            self.assertIsNone(validate_dns_scope(scope))

    def test_rejects_private_or_mixed_dns_results(self):
        import socket
        from unittest.mock import patch
        from app.ingestion.crawler import (
            build_crawl_scope,
            validate_dns_scope,
        )

        scope = build_crawl_scope("https://example.org/")

        for addresses in [
            ["127.0.0.1"],
            ["192.168.1.10"],
            ["169.254.169.254"],
            ["10.0.0.5"],
            ["8.8.8.8", "127.0.0.1"],
            ["::1"],
            ["fc00::1"],
        ]:
            with self.subTest(addresses=addresses):
                records = [
                    (
                        socket.AF_INET6 if ":" in address else socket.AF_INET,
                        socket.SOCK_STREAM,
                        6,
                        "",
                        (address, 443),
                    )
                    for address in addresses
                ]

                with patch("socket.getaddrinfo", return_value=records):
                    with self.assertRaises(ValueError):
                        validate_dns_scope(scope)

    def test_fails_closed_when_dns_unavailable(self):
        import socket
        from unittest.mock import patch
        from app.ingestion.crawler import (
            build_crawl_scope,
            validate_dns_scope,
        )

        scope = build_crawl_scope("https://example.org/")

        with patch(
            "socket.getaddrinfo",
            side_effect=socket.gaierror("DNS failed"),
        ):
            with self.assertRaises(ValueError):
                validate_dns_scope(scope)

        with patch("socket.getaddrinfo", return_value=[]):
            with self.assertRaises(ValueError):
                validate_dns_scope(scope)


class TestDNSPreflightEnforcement(unittest.IsolatedAsyncioTestCase):

    async def test_homepage_blocks_unsafe_dns_before_browser(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app.ingestion import crawler

        scope = crawler.build_crawl_scope("https://example.org/")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(crawler, "OUTPUT_DIR", Path(tmp)),
                patch.object(
                    crawler,
                    "validate_dns_scope",
                    side_effect=ValueError("Unsafe DNS"),
                ) as dns_check,
                patch.object(crawler, "AsyncWebCrawler") as browser,
            ):
                with self.assertRaisesRegex(ValueError, "Unsafe DNS"):
                    with _CrawlerTestLockDirectory() as _lock_dir:
                        await crawler.crawl_homepage(scope=scope, lock_path=_CrawlerTestPath(_lock_dir) / 'refresh.lock')

                dns_check.assert_called_once_with(scope)
                browser.assert_not_called()

    async def test_pages_block_unsafe_dns_before_browser(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from app.ingestion import crawler

        scope = crawler.build_crawl_scope("https://example.org/")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            urls_file = root / "discovered_urls.json"
            urls_file.write_text(
                json.dumps(["https://example.org/courses/python"]),
                encoding="utf-8",
            )

            with (
                patch.object(crawler, "URLS_FILE", urls_file),
                patch.object(crawler, "PAGES_DIR", root / "pages"),
                patch.object(
                    crawler,
                    "validate_dns_scope",
                    side_effect=ValueError("Unsafe DNS"),
                ) as dns_check,
                patch.object(crawler, "AsyncWebCrawler") as browser,
            ):
                with self.assertRaisesRegex(ValueError, "Unsafe DNS"):
                    with _CrawlerTestLockDirectory() as _lock_dir:
                        await crawler.crawl_pages(scope=scope, lock_path=_CrawlerTestPath(_lock_dir) / 'refresh.lock')

                dns_check.assert_called_once_with(scope)
                browser.assert_not_called()


class TestBrowserRequestGuard(unittest.IsolatedAsyncioTestCase):

    async def test_allows_only_approved_https_requests(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from app.ingestion.crawler import (
            build_crawl_scope,
            install_request_guard,
        )

        scope = build_crawl_scope("https://example.org/")
        context = SimpleNamespace(route=AsyncMock())

        await install_request_guard(context, scope)

        context.route.assert_awaited_once()
        handler = context.route.await_args.args[1]

        allowed_urls = [
            "https://example.org/",
            "https://example.org/courses/python",
            "https://example.org:443/static/main.js",
        ]

        for url in allowed_urls:
            with self.subTest(url=url):
                route = SimpleNamespace(
                    request=SimpleNamespace(url=url),
                    continue_=AsyncMock(),
                    abort=AsyncMock(),
                )

                await handler(route)

                route.continue_.assert_awaited_once()
                route.abort.assert_not_awaited()

    async def test_blocks_unauthorized_requests(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from app.ingestion.crawler import (
            build_crawl_scope,
            install_request_guard,
        )

        scope = build_crawl_scope("https://example.org/")
        context = SimpleNamespace(route=AsyncMock())

        await install_request_guard(context, scope)
        handler = context.route.await_args.args[1]

        blocked_urls = [
            "http://example.org/",
            "https://evil.com/private",
            "https://example.org.evil.com/",
            "https://example.org:8080/",
            "https://user:pass@example.org/",
            "https://127.0.0.1/",
            "https://169.254.169.254/",
            "file:///etc/passwd",
            "https://[::1",
        ]

        for url in blocked_urls:
            with self.subTest(url=url):
                route = SimpleNamespace(
                    request=SimpleNamespace(url=url),
                    continue_=AsyncMock(),
                    abort=AsyncMock(),
                )

                await handler(route)

                route.abort.assert_awaited_once()
                route.continue_.assert_not_awaited()


class TestBrowserGuardHook(unittest.IsolatedAsyncioTestCase):

    async def test_registers_hook_and_blocks_external_request(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock
        from app.ingestion.crawler import (
            build_crawl_scope,
            configure_browser_request_guard,
        )

        scope = build_crawl_scope("https://example.org/")
        browser = MagicMock()

        configure_browser_request_guard(browser, scope)

        set_hook = browser.crawler_strategy.set_hook
        set_hook.assert_called_once()

        hook_name, hook_callback = set_hook.call_args.args

        self.assertEqual(
            hook_name,
            "on_page_context_created",
        )

        page = object()
        context = SimpleNamespace(route=AsyncMock())

        returned_page = await hook_callback(page, context)

        self.assertIs(returned_page, page)
        context.route.assert_awaited_once()

        pattern, request_handler = context.route.await_args.args

        self.assertEqual(pattern, "**/*")

        blocked_route = SimpleNamespace(
            request=SimpleNamespace(
                url="https://evil.com/private"
            ),
            abort=AsyncMock(),
            continue_=AsyncMock(),
        )

        await request_handler(blocked_route)

        blocked_route.abort.assert_awaited_once()
        blocked_route.continue_.assert_not_awaited()

        allowed_route = SimpleNamespace(
            request=SimpleNamespace(
                url="https://example.org/courses/python"
            ),
            abort=AsyncMock(),
            continue_=AsyncMock(),
        )

        await request_handler(allowed_route)

        allowed_route.continue_.assert_awaited_once()
        allowed_route.abort.assert_not_awaited()


class TestSiteStorageIsolation(unittest.TestCase):

    def test_different_domains_use_separate_storage(self):
        import tempfile
        from pathlib import Path
        from app.ingestion.crawler import (
            build_crawl_scope,
            site_storage_paths,
        )

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"

            skill = site_storage_paths(
                build_crawl_scope("https://skillshikshya.com/"),
                root=raw_dir,
            )
            other = site_storage_paths(
                build_crawl_scope("https://example.org/"),
                root=raw_dir,
            )

            self.assertEqual(
                skill.output_dir,
                raw_dir / "skillshikshya.com",
            )
            self.assertEqual(
                other.output_dir,
                raw_dir / "example.org",
            )
            self.assertNotEqual(skill.output_dir, other.output_dir)

            for paths in (skill, other):
                self.assertEqual(
                    paths.urls_file,
                    paths.output_dir / "discovered_urls.json",
                )
                self.assertEqual(
                    paths.pages_dir,
                    paths.output_dir / "pages",
                )

    def test_path_generation_does_not_create_files(self):
        import tempfile
        from pathlib import Path
        from app.ingestion.crawler import (
            build_crawl_scope,
            site_storage_paths,
        )

        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"

            site_storage_paths(
                build_crawl_scope("https://example.org/"),
                root=raw_dir,
            )

            self.assertFalse(raw_dir.exists())


class TestScopedCrawlerStorage(unittest.IsolatedAsyncioTestCase):

    async def test_homepage_writes_into_site_directory(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.ingestion import crawler

        scope = crawler.build_crawl_scope("https://example.org/")

        response = SimpleNamespace(
            success=True,
            url="https://example.org/",
            markdown="# Example Courses\n\nLearn Python.",
            links={
                "internal": [
                    {"href": "/courses/python"}
                ]
            },
        )

        worker = MagicMock()
        worker.arun = AsyncMock(return_value=response)

        browser = MagicMock()
        browser.__aenter__ = AsyncMock(return_value=worker)
        browser.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"

            with (
                patch.object(crawler, "OUTPUT_DIR", raw_root),
                patch.object(
                    crawler, "URLS_FILE",
                    raw_root / "discovered_urls.json"
                ),
                patch.object(
                    crawler, "validate_dns_scope", return_value=None
                ),
                patch.object(
                    crawler, "AsyncWebCrawler", return_value=browser
                ),
            ):
                with _CrawlerTestLockDirectory() as _lock_dir:
                    await crawler.crawl_homepage(scope=scope, lock_path=_CrawlerTestPath(_lock_dir) / 'refresh.lock')

            site_dir = raw_root / "example.org"

            self.assertTrue((site_dir / "homepage.md").exists())
            self.assertTrue((site_dir / "homepage.json").exists())
            self.assertTrue(
                (site_dir / "discovered_urls.json").exists()
            )

            urls = json.loads(
                (site_dir / "discovered_urls.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                urls,
                ["https://example.org/courses/python"],
            )

            self.assertFalse((raw_root / "homepage.md").exists())
            self.assertFalse(
                (raw_root / "discovered_urls.json").exists()
            )


class TestCrawlerCLI(unittest.TestCase):

    def test_custom_url_reaches_homepage_crawler(self):
        from unittest.mock import patch, sentinel
        from app.ingestion import crawler

        with (
            patch("sys.argv", [
                "crawler.py", "--url", "https://example.org/"
            ]),
            patch.object(
                crawler, "crawl_homepage",
                return_value=sentinel.homepage_job
            ) as homepage,
            patch.object(crawler.asyncio, "run") as run,
        ):
            crawler.main()

        homepage.assert_called_once()
        scope = homepage.call_args.kwargs["scope"]
        self.assertEqual(scope.base_url, "https://example.org/")
        run.assert_called_once()
        job = run.call_args.args[0]
        if crawler.asyncio.iscoroutine(job):
            job.close()
        self.assertTrue(crawler.asyncio.iscoroutine(job))

    def test_custom_url_reaches_pages_crawler(self):
        from unittest.mock import patch, sentinel
        from app.ingestion import crawler

        with (
            patch("sys.argv", [
                "crawler.py", "--pages",
                "--url", "https://example.org/"
            ]),
            patch.object(
                crawler, "crawl_pages",
                return_value=sentinel.pages_job
            ) as pages,
            patch.object(crawler.asyncio, "run") as run,
        ):
            crawler.main()

        pages.assert_called_once()
        scope = pages.call_args.kwargs["scope"]
        self.assertEqual(scope.base_url, "https://example.org/")
        run.assert_called_once()
        job = run.call_args.args[0]
        if crawler.asyncio.iscoroutine(job):
            job.close()
        self.assertTrue(crawler.asyncio.iscoroutine(job))

    def test_rejects_insecure_cli_url(self):
        from unittest.mock import patch
        from app.ingestion import crawler

        with (
            patch("sys.argv", [
                "crawler.py", "--url", "http://example.org/"
            ]),
            patch.object(crawler.asyncio, "run") as run,
        ):
            with self.assertRaises(ValueError):
                crawler.main()

        run.assert_not_called()
