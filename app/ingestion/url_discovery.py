from urllib.parse import urljoin, urlsplit, urlunsplit

BASE_URL = "https://skillshikshya.com/"

ALLOWED_HOSTS = {
    "skillshikshya.com",
    "www.skillshikshya.com",
}

CONTENT_PATHS = (
    "/courses",
    "/faq",
    "/about",
    "/blogs",
    "/contact-us",
)


def normalize_content_url(href: str) -> str | None:
    if not href:
        return None

    absolute = urljoin(BASE_URL, href)
    parsed = urlsplit(absolute)

    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None

    try:
        if parsed.port not in (None, 443):
            return None
    except ValueError:
        return None

    path = parsed.path.rstrip("/") or "/"

    allowed = path == "/" or any(
        path == prefix or path.startswith(prefix + "/") for prefix in CONTENT_PATHS
    )

    if not allowed:
        return None

    return urlunsplit(("https", parsed.hostname, path, "", ""))


def discover_public_urls(links: list[dict]) -> list[str]:
    urls = set()

    for link in links:
        url = normalize_content_url(link.get("href", ""))
        if url:
            urls.add(url)

    return sorted(urls)
