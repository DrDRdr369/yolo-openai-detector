"""OpenAI-compatible chat-completion schemas (facade).

Mirror docs/api-contract.md section 4. Accepts the vision message shape; unknown/extra
OpenAI fields on request models are silently ignored via ConfigDict(extra="ignore").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Tolerant(BaseModel):
    """Base for request-side models: extra fields from the OpenAI SDK are silently dropped."""

    model_config = ConfigDict(extra="ignore")


class ImageUrlContent(_Tolerant):
    url: str


class ContentPart(_Tolerant):
    type: str
    image_url: ImageUrlContent | None = None
    text: str | None = None


class Message(_Tolerant):
    role: str
    content: str | list[ContentPart] = ""


class ChatCompletionRequest(_Tolerant):
    model: str | None = None
    messages: list[Message]
    stream: bool = False


# ---------------------------------------------------------------------------
# Response models (we build these ourselves — no need for extra="ignore")
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int
    finish_reason: str
    message: AssistantMessage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
