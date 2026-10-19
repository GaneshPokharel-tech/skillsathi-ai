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
            links_file = output / "discovered_urls.json"

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
                await crawler.crawl_homepage(scope=scope)

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
            urls_file = root / "discovered_urls.json"
            pages_dir = root / "pages"

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
                patch.object(crawler, "URLS_FILE", urls_file),
                patch.object(crawler, "PAGES_DIR", pages_dir),
                patch.object(crawler, "REQUEST_DELAY", 0),
            ):
                await crawler.crawl_pages(scope=scope)

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
                    await crawler.crawl_homepage(scope=scope)

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
                    await crawler.crawl_pages(scope=scope)

                dns_check.assert_called_once_with(scope)
                browser.assert_not_called()
