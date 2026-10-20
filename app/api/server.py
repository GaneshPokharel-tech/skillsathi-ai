"""HTTP API for the SkillSathi AI chatbot."""

import logging
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.rag.engine import ask_skillsathi

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SkillSathi AI API",
    version="0.1.0",
)

# Local Qdrant uses filesystem storage. Serialize requests within
# this process until we migrate to a concurrent Qdrant server.
_chat_lock = Lock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, strict=True, ge=1, le=10)
    language: Literal["ne", "en"] = "ne"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be blank.")
        return question


class Citation(BaseModel):
    marker: str
    section: str
    source_url: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: int = Field(ge=0)


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    """Serve the same-origin student chat interface."""
    return FileResponse(
        Path(__file__).with_name("chat.html"),
        media_type="text/html",
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; does not verify Qdrant or Gemini."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a student question using website-grounded RAG."""
    try:
        with _chat_lock:
            result = ask_skillsathi(
                request.question,
                top_k=request.top_k,
                language=request.language,
            )
        return ChatResponse.model_validate(result)

    except Exception as exc:
        # Log only exception type, not API keys or student content.
        logger.error(
            "Chat processing failed: %s",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Chat service temporarily unavailable.",
        ) from None
