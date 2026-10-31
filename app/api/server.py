"""HTTP API for the SkillSathi AI chatbot."""

import logging
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pathlib import Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.agent.service import ask_agent
from contextlib import asynccontextmanager
import asyncio
from app.ingestion.auto_sync_scheduler import AutoSyncScheduler

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(application: FastAPI):
    scheduler = AutoSyncScheduler(chat_lock=_chat_lock)
    scheduler.start()

    try:
        yield
    finally:
        await asyncio.to_thread(scheduler.stop)


app = FastAPI(
    title="SkillSathi AI API",
    version="0.1.0",
    lifespan=lifespan,
)

# Local Qdrant uses filesystem storage. Serialize requests within
# this process until we migrate to a concurrent Qdrant server.
_chat_lock = Lock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, strict=True, ge=1, le=10)
    language: Literal["ne", "en"] = "ne"
    pending_course_urls: list[str] | None = Field(
        default=None, min_length=2, max_length=5
    )

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
    pending_course_urls: list[str] | None = None


@app.get("/brand-icon", include_in_schema=False)
def brand_icon() -> FileResponse:
    """Serve the locally stored SkillShikshya site icon."""
    return FileResponse(
        Path(__file__).with_name("skillshikshya-icon.png"),
        media_type="image/png",
    )


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


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a student question using website-grounded RAG."""
    try:
        with _chat_lock:
            options = {
                "top_k": request.top_k,
                "language": request.language,
            }

            if request.pending_course_urls is not None:
                options["pending_course_urls"] = (
                    request.pending_course_urls
                )

            result = ask_agent(
                request.question,
                **options,
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




@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Handle validated chat messages over one persistent connection."""
    from urllib.parse import urlsplit

    origin = websocket.headers.get("origin")

    if origin is not None:
        try:
            parsed = urlsplit(origin)
            allowed = (
                parsed.scheme in {"http", "https"}
                and parsed.netloc.lower()
                == websocket.headers.get("host", "").lower()
            )
        except ValueError:
            allowed = False

        if not allowed:
            await websocket.close(code=1008)
            return

    await websocket.accept()

    invalid_request = {
        "type": "error",
        "code": "invalid_request",
        "message": "Invalid chat request.",
    }

    service_unavailable = {
        "type": "error",
        "code": "service_unavailable",
        "message": "Chat service temporarily unavailable.",
    }

    while True:
        try:
            payload = await websocket.receive_json()

        except WebSocketDisconnect:
            return

        except (ValueError, UnicodeError):
            await websocket.send_json(invalid_request)
            continue

        try:
            request = ChatRequest.model_validate(payload)

        except ValidationError:
            await websocket.send_json(invalid_request)
            continue

        try:
            # Reuse the HTTP handler, including its Qdrant lock,
            # validation contract, and sanitized error handling.
            response = await asyncio.to_thread(chat, request)

        except HTTPException:
            await websocket.send_json(service_unavailable)
            continue

        except Exception as exc:
            logger.error(
                "WebSocket chat failed: %s",
                type(exc).__name__,
            )
            await websocket.send_json(service_unavailable)
            continue

        await websocket.send_json({
            "type": "answer",
            "data": response.model_dump(exclude_none=True),
        })


@app.get("/widget.js", include_in_schema=False)
def chatbot_widget_script():
    """Serve the embeddable chatbot launcher."""
    from pathlib import Path
    from fastapi.responses import FileResponse

    return FileResponse(
        Path(__file__).with_name("widget.js"),
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )
