"""Turn SkillSathi tool results into student-facing answers."""

import os
from pathlib import Path

from app.agent.executor import execute_question
from app.rag.engine import answer_with_evidence


def ask_agent(
    question: str,
    *,
    language: str = "ne",
    top_k: int = 4,
    client=None,
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
                "section": "Course option",
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

        return {
            "answer": "\n".join(lines),
            "citations": citations,
            "retrieved_chunks": 0,
        }

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
            question.strip(),
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
