"""Execute SkillSathi tool plans against indexed website evidence."""

from app.agent.router import decide_route
from app.rag.engine import retrieve_evidence
from app.tools.courses import (
    compare_courses,
    get_course_details,
    list_courses,
)


def execute_question(
    question: str,
    *,
    top_k: int = 4,
) -> dict:
    """Route a student question and execute one supported tool.

    Returns structured tool data, not an LLM-generated answer.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if len(question) > 2000:
        raise ValueError("Question exceeds 2000 characters.")

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        raise ValueError("top_k must be an integer between 1 and 10.")

    question = question.strip()

    # list_courses opens and closes its local Qdrant client.
    # Subsequent tools run only after that client is released.
    catalog = list_courses()
    plan = decide_route(question, catalog)

    tool = plan["tool"]
    arguments = plan["arguments"]

    if tool == "list_courses":
        data = {
            "courses": catalog,
            "scope": "indexed_courses_only",
        }

    elif tool == "get_course_details":
        details = get_course_details(
            arguments["course_url"],
            max_chunks=8,
        )

        data = {
            "found": details is not None,
            "course": details,
        }

    elif tool == "compare_courses":
        data = compare_courses(
            arguments["first_url"],
            arguments["second_url"],
            max_chunks_per_course=4,
        )

    elif tool == "knowledge_search":
        from app.agent.router import split_user_questions

        parts = split_user_questions(question)

        if len(parts) == 1:
            evidence = retrieve_evidence(question, limit=top_k)
        else:
            evidence = []
            seen = set()

            for part in parts:
                results = retrieve_evidence(
                    part,
                    limit=min(top_k, 3),
                )

                for item in results:
                    key = (
                        item.get("source_url"),
                        item.get("section"),
                        item.get("text"),
                    )

                    if key not in seen:
                        seen.add(key)
                        evidence.append(item)

        data = {
            "evidence": evidence,
            "retrieved_chunks": len(evidence),
        }

    elif tool == "clarify":
        data = {
            "reason": arguments["reason"],
            "matched_courses": arguments["matched_courses"],
        }

    else:
        raise ValueError(f"Unsupported tool route: {tool}")

    return {
        "tool": tool,
        "arguments": arguments,
        "data": data,
    }
