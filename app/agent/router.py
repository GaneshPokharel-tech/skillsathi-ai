"""Conservative tool routing for SkillSathi course questions."""

import re


GENERIC = {
    "a", "about", "ai", "and", "course", "courses",
    "in", "job", "nepal", "of", "program", "programs",
    "professional", "ready", "the", "to", "training",
    "what", "with",
}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold())) - GENERIC



def split_user_questions(question: str) -> list[str]:
    """Separate explicit multi-question messages without splitting courses."""
    raw = question.strip()
    if not raw:
        return []

    numbered = [
        part.strip()
        for part in re.split(
            r"(?:^|\s)\d{1,2}[.)]\s+(?=\S)",
            raw,
        )
        if part.strip()
    ]

    if (
        2 <= len(numbered) <= 6
        and re.search(r"(?:^|\s)\d{1,2}[.)]\s+", raw)
    ):
        return numbered

    lines = [
        re.sub(r"^[-•]\s*", "", line.strip())
        for line in raw.splitlines()
        if line.strip()
    ]

    if (
        2 <= len(lines) <= 6
        and all(len(part) >= 8 for part in lines)
    ):
        return lines


    # Separate question marks even when the next question has
    # no leading whitespace. Preserve URL query parameters.
    url_spans = [
        (match.start(), match.end())
        for match in re.finditer(r"https?://\S+", raw)
    ]

    boundaries = [
        match
        for match in re.finditer(
            r"(?<=[.!])\s+(?=[A-Z])|(?<=\?)\s*(?=[^\W\d_])",
            raw,
        )
        if not any(
            start <= match.start() < end
            for start, end in url_spans
        )
    ]

    sentence_parts = []
    start = 0

    for boundary in boundaries:
        part = raw[start:boundary.start()].strip()
        if part:
            sentence_parts.append(part)
        start = boundary.end()

    if raw[start:].strip():
        sentence_parts.append(raw[start:].strip())

    if 2 <= len(sentence_parts) <= 6 and "?" in raw:
        return sentence_parts

    if raw.count("?") >= 2 and not url_spans:
        parts = [
            part.strip()
            for part in re.findall(r"[^?]+\??", raw)
            if part.strip()
        ]
        if 2 <= len(parts) <= 6:
            return parts

    return [raw]


def decide_route(question: str, catalog: list[dict]) -> dict:
    """Plan one tool call without executing it or inventing courses."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if not isinstance(catalog, list):
        raise ValueError("Catalog must be a list.")

    text = question.strip().casefold()
    words = _tokens(text)

    comparison = bool(
        re.search(
            r"\b(compare|comparison|versus|vs|difference|better)\b",
            text,
        )
        or any(
            term in text
            for term in ("तुलना", "फरक", "भिन्नता", "कुन राम्रो")
        )
    )

    discovery = bool(
        re.search(
            r"\b(list|available|catalog|offer|offered)\b",
            text,
        )
        and re.search(r"\b(courses?|programs?)\b", text)
        or "which courses" in text
        or (
            "कोर्स" in text
            and any(
                term in text
                for term in ("के के", "कुन कुन", "उपलब्ध", "देखाउ")
            )
        )
    )

    informational_terms = (
        "career path", "career options", "mission",
        "counselling", "counseling", "blog", "article",
        "contact", "address", "phone number",
        "privacy policy", "terms of service",
        "testimonials", "announcements",
    )

    if len(split_user_questions(question)) > 1 or (
        not comparison
        and not discovery
        and any(term in text for term in informational_terms)
    ):
        return {
            "tool": "knowledge_search",
            "arguments": {"question": question.strip()},
        }

    matches = []

    for course in catalog:
        if not isinstance(course, dict):
            raise ValueError("Invalid catalog entry.")

        title = course.get("title")
        url = course.get("source_url")

        if not isinstance(title, str) or not isinstance(url, str):
            raise ValueError("Catalog entry must have a title and URL.")

        title_words = _tokens(title)
        positions = []

        for word in words:
            if (
                word in title_words
                or (
                    len(word) >= 4
                    and any(name.startswith(word) for name in title_words)
                )
            ):
                positions.append(text.find(word))

        if positions:
            matches.append((min(positions), course))

    # Preserve the order courses are mentioned in the question.
    matches.sort(key=lambda item: item[0])
    selected = [course for _, course in matches]

    if comparison:
        if len(selected) == 2:
            return {
                "tool": "compare_courses",
                "arguments": {
                    "first_url": selected[0]["source_url"],
                    "second_url": selected[1]["source_url"],
                },
            }

        return {
            "tool": "clarify",
            "arguments": {
                "reason": "Exactly two identifiable courses are required.",
                "matched_courses": selected,
            },
        }

    if discovery:
        return {"tool": "list_courses", "arguments": {}}

    if len(selected) == 1:
        return {
            "tool": "get_course_details",
            "arguments": {"course_url": selected[0]["source_url"]},
        }

    if len(selected) > 1:
        return {
            "tool": "clarify",
            "arguments": {
                "reason": "Multiple courses mentioned without a clear intent.",
                "matched_courses": selected,
            },
        }

    return {"tool": "knowledge_search", "arguments": {"question": question.strip()}}
