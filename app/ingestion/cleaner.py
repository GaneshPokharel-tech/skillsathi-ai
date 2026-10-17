import html
import re
from collections import Counter
from urllib.parse import urlsplit

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
