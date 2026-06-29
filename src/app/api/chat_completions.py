"""POST /v1/chat/completions — OpenAI compatibility facade.

Implements: PR-3.

Contract: docs/api-contract.md section 4.
- Accept the standard OpenAI vision message shape.
- Extract exactly ONE base64 image from the first image_url content part.
- Multiple/zero images, remote URLs, or streaming -> 400 (fail closed).
- Return a standard chat.completion envelope whose assistant message content is a JSON
  STRING of the detection payload (so clients can json.loads it). usage is zero-filled.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions() -> dict:
    """Adapt an OpenAI vision chat request into a detection call.

    PR-3: parse the message, reuse the detection pipeline, and wrap the result in a
    chat.completion envelope. Reject unsupported features per the contract.
    """
    raise NotImplementedError("PR-3: implement chat-completions facade.")
