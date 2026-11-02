"""Streamlit demonstration UI for SkillSathi AI."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/chat"
ALLOWED_HOSTS = {"skillshikshya.com", "www.skillshikshya.com"}


class ChatUnavailable(Exception):
    """Safe message for unavailable or invalid chat responses."""


def call_chat(
    question,
    *,
    language="ne",
    pending_course_urls=None,
):
    """Call the existing FastAPI RAG endpoint."""
    question = question.strip()

    if not question or len(question) > 2000:
        raise ValueError("Question must contain 1 to 2000 characters.")

    payload = {
        "question": question,
        "top_k": 4,
        "language": language,
    }

    if pending_course_urls is not None:
        payload["pending_course_urls"] = pending_course_urls

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        raise ChatUnavailable(
            "Chat service temporarily unavailable."
        ) from None

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("answer"), str)
        or not result["answer"].strip()
        or not isinstance(result.get("citations"), list)
    ):
        raise ChatUnavailable(
            "Chat service temporarily unavailable."
        )

    return result


def trusted_citations(citations):
    """Allow links only to HTTPS SkillShikshya pages."""
    approved = []

    if not isinstance(citations, list):
        return approved

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        url = citation.get("source_url")

        if not isinstance(url, str):
            continue

        try:
            parsed = urlsplit(url)

            if (
                parsed.scheme != "https"
                or parsed.hostname not in ALLOWED_HOSTS
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
            ):
                continue

        except ValueError:
            continue

        approved.append({
            "marker": str(citation.get("marker", "")),
            "section": str(citation.get("section", "Source")),
            "source_url": url,
        })

    return approved


def main():
    """Launch the Streamlit student chat demonstration."""
    import streamlit as st

    st.set_page_config(
        page_title="SkillSathi AI",
        page_icon="🎓",
        layout="centered",
    )

    st.title("SkillSathi AI")
    st.caption(
        "SkillShikshya's bilingual AI course assistant"
    )

    st.sidebar.header("Chat Settings")

    language_label = st.sidebar.selectbox(
        "Response language",
        ["Nepali", "English"],
    )

    language = (
        "ne" if language_label == "Nepali" else "en"
    )

    st.sidebar.caption(
        "Powered by FastAPI, RAG, Gemini and Qdrant."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_course_urls" not in st.session_state:
        st.session_state.pending_course_urls = None

    if st.sidebar.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.pending_course_urls = None
        st.rerun()

    st.info(
        "Ask about SkillShikshya courses, training, "
        "admissions or counselling. "
        "Please do not enter sensitive personal information."
    )

    def display_message(message):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for citation in message.get("citations", []):
                st.markdown(
                    f'{citation["marker"]} '
                    f'[{citation["section"]}]'
                    f'({citation["source_url"]})'
                )

    for message in st.session_state.messages:
        display_message(message)

    question = st.chat_input(
        "Ask a question about SkillShikshya..."
    )

    if not question:
        return

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)
    display_message(user_message)

    with st.spinner("Searching SkillShikshya knowledge..."):
        try:
            result = call_chat(
                question,
                language=language,
                pending_course_urls=(
                    st.session_state.pending_course_urls
                ),
            )

            options = result.get("pending_course_urls")
            st.session_state.pending_course_urls = (
                options
                if isinstance(options, list)
                and 2 <= len(options) <= 5
                and all(isinstance(url, str) for url in options)
                else None
            )

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "citations": trusted_citations(
                    result["citations"]
                ),
            }

        except (ChatUnavailable, ValueError):
            assistant_message = {
                "role": "assistant",
                "content": (
                    "Sorry, the chatbot is temporarily "
                    "unavailable. Please try again."
                ),
            }

    st.session_state.messages.append(assistant_message)
    display_message(assistant_message)


if __name__ == "__main__":
    main()
