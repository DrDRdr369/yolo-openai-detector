"""Bearer-token authentication dependency.

Rules (see AGENTS.md section 3):
- Single fixed key from settings (GATEWAY_API_KEY).
- Compare in constant time (hmac.compare_digest), never `==`.
- Missing/invalid key -> HTTP 401 with an OpenAI-style error body.
- Never log or echo the key.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from .config import get_settings

_AUTH_ERROR = {
    "error": {
        "type": "authentication_error",
        "message": "Invalid or missing API key.",
    }
}


async def require_api_key(request: Request) -> None:
    """FastAPI dependency — authorizes a request or raises HTTP 401."""
    auth = request.headers.get("Authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise HTTPException(status_code=401, detail=_AUTH_ERROR)
    token = parts[1]
    expected = get_settings().gateway_api_key
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail=_AUTH_ERROR)
