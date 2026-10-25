"""Turn SkillSathi tool results into student-facing answers."""

import os
from pathlib import Path

from app.agent.executor import execute_question
from app.rag.engine import answer_with_evidence



CHOICE_WORDS = (
    {
        "1", "1st", "first", "first one", "the first one",
        "option 1", "number one", "पहिलो", "पहिलो वाला",
    },
    {
        "2", "2nd", "second", "second one", "the second one",
        "option 2", "number two", "दोस्रो", "दोस्रो वाला",
        "दोश्रो", "दोश्रो वाला",
    },
    {
        "3", "3rd", "third", "third one", "option 3",
        "तेस्रो", "तेस्रो वाला",
    },
    {
        "4", "4th", "fourth", "fourth one", "option 4",
        "चौथो", "चौथो वाला",
    },
    {
        "5", "5th", "fifth", "fifth one", "option 5",
        "पाँचौँ", "पाँचौँ वाला",
    },
)


def _selected_option(question: str) -> int | None:
    """Resolve an explicit numbered follow-up selection."""
    normalized = " ".join(question.casefold().split())

    for index, choices in enumerate(CHOICE_WORDS):
        if normalized in choices:
            return index

    return None



def _immediate_safety_reply(question: str, language: str):
    """Prioritize immediate safety over course retrieval for crisis statements."""
    import re

    message = question.casefold()

    patterns = (
        r"\b(?:kill myself|end my life|want to die|about to die)\b",
        r"\bmarna\s+(?:la+geko|lagya)\s+(?:chu|xu|chhu)\b",
        r"\b(?:ma|mah)\s+marna\s+chahanchu\b",
        r"म\s+मर्न\s+लागे",
        r"म\s+मर्न\s+चाहन्छु",
        r"आत्महत्या\s+गर्न\s+लागे",
    )

    if not any(re.search(pattern, message) for pattern in patterns):
        return None

    if language == "en":
        return (
            "Your immediate safety matters. If you are in danger "
            "right now, call your local emergency service or go to "
            "the nearest emergency department. Move away from "
            "anything you could use to hurt yourself and contact "
            "someone you trust who can stay with you. "
            "Are you safe right now?"
        )

    return (
        "तपाईंको सुरक्षा अहिले सबैभन्दा महत्त्वपूर्ण छ। "
        "यदि अहिले खतरा छ भने तुरुन्त स्थानीय आपत्कालीन "
        "सेवामा फोन गर्नुहोस् वा नजिकको अस्पतालको "
        "आपत्कालीन कक्षमा जानुहोस्। आफूलाई चोट पुर्‍याउन "
        "सक्ने वस्तुबाट टाढा रहनुहोस् र विश्वासिलो "
        "व्यक्तिलाई आफूसँग बस्न बोलाउनुहोस्। "
        "तपाईं अहिले सुरक्षित हुनुहुन्छ?"
    )


def ask_agent(
    question: str,
    *,
    language: str = "ne",
    top_k: int = 4,
    client=None,
    pending_course_urls: list[str] | None = None,
) -> dict:
    """Execute a tool and return an answer with source references."""
    if language not in ("ne", "en"):
        raise ValueError("Unsupported language.")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        raise ValueError("top_k must be between 1 and 10.")

    if pending_course_urls is not None:
        if (
            not isinstance(pending_course_urls, list)
            or not 2 <= len(pending_course_urls) <= 5
            or any(
                not isinstance(url, str) or not url
                for url in pending_course_urls
            )
            or len(set(pending_course_urls)) != len(pending_course_urls)
        ):
            raise ValueError("Invalid pending course options.")

    safety_reply = _immediate_safety_reply(question, language)
    if safety_reply is not None:
        return {
            "answer": safety_reply,
            "citations": [],
            "retrieved_chunks": 0,
        }

    selected = _selected_option(question)
    effective_question = question.strip()

    if selected is not None and pending_course_urls is not None:
        if selected >= len(pending_course_urls):
            return {
                "answer": (
                    "Please choose one of the displayed course numbers."
                    if language == "en"
                    else "कृपया देखाइएका कोर्स नम्बरमध्ये एउटा छान्नुहोस्।"
                ),
                "citations": [],
                "retrieved_chunks": 0,
                "pending_course_urls": pending_course_urls,
            }

        from app.tools.courses import list_courses, get_course_details

        catalog = {
            course["source_url"]: course
            for course in list_courses()
        }

        if any(url not in catalog for url in pending_course_urls):
            raise ValueError("Course context is not in the indexed catalog.")

        chosen_url = pending_course_urls[selected]
        chosen_course = catalog[chosen_url]

        details = get_course_details(
            chosen_url,
            max_chunks=8,
        )

        plan = {
            "tool": "get_course_details",
            "data": {
                "found": details is not None,
                "course": details,
            },
        }

        effective_question = (
            "Explain the curriculum, skills and learning outcomes of "
            f"{chosen_course['title']}, using only website evidence."
        )

    elif selected is not None:
        return {
            "answer": (
                "Please mention the course name first."
                if language == "en"
                else "कृपया पहिले कोर्सको नाम बताउनुहोस्।"
            ),
            "citations": [],
            "retrieved_chunks": 0,
        }

    else:
        plan = execute_question(question.strip(), top_k=top_k)
    tool = plan["tool"]
    data = plan["data"]

    if tool in ("list_courses", "clarify"):
        courses = (
            data["courses"]
            if tool == "list_courses"
            else data["matched_courses"]
        )

        if tool == "list_courses":
            heading = (
                "Indexed SkillShikshya courses:"
                if language == "en"
                else "SkillShikshya का इन्डेक्स गरिएका कोर्सहरू:"
            )
        else:
            heading = (
                "Which course do you mean? Please choose:"
                if language == "en"
                else "तपाईं कुन कोर्सबारे सोध्दै हुनुहुन्छ? कृपया छान्नुहोस्:"
            )

        lines = [heading]
        citations = []

        for index, course in enumerate(courses, 1):
            marker = f"[{index}]"
            lines.append(f"{index}. {course['title']} {marker}")
            citations.append({
                "marker": marker,
                "section": course["title"],
                "source_url": course["source_url"],
            })

        if tool == "list_courses":
            lines.append(
                "\nThis is the indexed catalog, not necessarily "
                "the full live website catalog."
                if language == "en"
                else "\nयो इन्डेक्स गरिएको सूची हो; वेबसाइटका सबै "
                     "कोर्स समावेश नहुन सक्छन्।"
            )

        if not courses:
            lines = [
                "No matching indexed courses found."
                if language == "en"
                else "मिल्ने इन्डेक्स गरिएको कोर्स भेटिएन।"
            ]

        response = {
            "answer": "\n".join(lines),
            "citations": citations,
            "retrieved_chunks": 0,
        }

        if tool == "clarify" and len(courses) >= 2:
            response["pending_course_urls"] = [
                course["source_url"]
                for course in courses
            ]

        return response

    if tool == "get_course_details":
        course = data["course"] if data["found"] else None
        evidence = course["evidence"] if course else []
        incomplete = bool(course and course.get("truncated"))

    elif tool == "compare_courses":
        if not data["comparison_ready"]:
            return {
                "answer": (
                    "There is not enough indexed information to "
                    "compare both courses."
                    if language == "en"
                    else "दुवै कोर्स तुलना गर्न पर्याप्त "
                         "इन्डेक्स गरिएको जानकारी छैन।"
                ),
                "citations": [],
                "retrieved_chunks": 0,
            }

        courses = data["courses"]
        evidence = [
            item
            for course in courses
            for item in course["evidence"]
        ]
        incomplete = any(
            course.get("truncated", False)
            for course in courses
        )

    elif tool == "knowledge_search":
        evidence = data["evidence"]
        incomplete = False

    else:
        raise ValueError(f"Unsupported agent tool: {tool}")

    if not evidence:
        return {
            "answer": (
                "I could not find enough information on the "
                "SkillShikshya website."
                if language == "en"
                else "SkillShikshya वेबसाइटमा पर्याप्त जानकारी भेटिएन।"
            ),
            "citations": [],
            "retrieved_chunks": 0,
        }

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(
        dotenv_path=Path(__file__).resolve().parents[2] / ".env"
    )

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    owns_client = client is None

    if owns_client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=45000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    try:
        result = answer_with_evidence(
            effective_question,
            evidence,
            client=client,
            model=model,
            language=language,
        )
    finally:
        if owns_client:
            client.close()

    answer = result["answer"]

    if incomplete:
        answer += (
            "\n\nNote: This answer is based on selected indexed "
            "excerpts, not a complete syllabus."
            if language == "en"
            else "\n\nनोट: यो उत्तर छानिएका इन्डेक्स गरिएका "
                 "अंशमा आधारित छ, पूर्ण पाठ्यक्रममा होइन।"
        )

    return {
        "answer": answer,
        "citations": result["citations"],
        "retrieved_chunks": len(evidence),
    }
