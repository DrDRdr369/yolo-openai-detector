"""POST /v1/chat/completions — OpenAI compatibility facade.

Contract: docs/api-contract.md section 4.
- Accept the standard OpenAI vision message shape; tolerate extra fields.
- Extract exactly ONE base64 image from image_url content parts.
- Multiple/zero images, remote URLs, or streaming → 400 (fail closed).
- Return a standard chat.completion envelope; assistant message.content is a
  JSON string of the detection payload (clients can json.loads it).
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_api_key
from ..config import Settings, get_settings
from ..dependencies import get_engine
from ..inference.engine import DetectionEngine
from ..schemas.detections import Box, Detection
from ..schemas.openai import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
)
from ..service import run_detection

router = APIRouter()

_ERR = "invalid_request_error"


def _bad(msg: str) -> HTTPException:
    return HTTPException(400, detail={"error": {"type": _ERR, "message": msg}})


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    _: None = Depends(require_api_key),
    engine: DetectionEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> ChatCompletionResponse:
    """Adapt an OpenAI vision chat request into a detection call and return the envelope."""
    if body.stream:
        raise _bad("Streaming is not supported.")

    image_urls: list[str] = []
    for msg in body.messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "image_url" and part.image_url is not None:
                    image_urls.append(part.image_url.url)

    if len(image_urls) == 0:
        raise _bad("Exactly one image is required; none found.")
    if len(image_urls) > 1:
        raise _bad(f"Exactly one image is required; {len(image_urls)} found.")

    # ImageDecodeError for remote URLs and invalid images propagates to the 400 handler.
    image, raw, _decode_ms, _infer_ms = run_detection(
        engine=engine,
        image_b64=image_urls[0],
        max_bytes=settings.max_image_bytes,
        max_pixels=settings.max_image_pixels,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.iou_threshold,
    )
    del image  # not needed for the chat response

    # Serialise through Pydantic to coerce numpy scalars and match the native endpoint format.
    detections_out = [
        Detection(
            class_id=d["class_id"],
            label=d["label"],
            confidence=d["confidence"],
            box=Box(**d["box"]),
        ).model_dump()
        for d in raw
    ]
    content_str = json.dumps({"detections": detections_out})
    model_id = body.model or settings.model_id

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model_id,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=AssistantMessage(role="assistant", content=content_str),
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
