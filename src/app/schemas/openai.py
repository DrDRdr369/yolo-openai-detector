"""OpenAI-compatible chat-completion schemas (facade).

Implements: PR-3.

Mirror docs/api-contract.md section 4. Only the subset needed to accept a vision request
and emit a chat.completion envelope. Unsupported fields (stream, etc.) are rejected at the
handler, not silently accepted.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    """PR-3: model (str | None), messages (list), optional fields.

    The image is carried inside messages[].content[].image_url.url as base64/data: URL.
    """


class ChatCompletionResponse(BaseModel):
    """PR-3: standard chat.completion envelope with zero-filled usage."""
