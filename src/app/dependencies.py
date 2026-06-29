"""Shared FastAPI dependencies.

Kept separate from main.py so routers can import them without creating circular
imports (main.py imports routers, routers import dependencies).
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from .inference.engine import DetectionEngine

_503_DETAIL = {
    "error": {
        "type": "server_error",
        "message": "Detection model is not loaded. Check MODEL_PATH and restart the service.",
    }
}


def get_engine(request: Request) -> DetectionEngine:
    """FastAPI dependency — return the loaded engine or raise HTTP 503.

    The engine is stored in ``app.state.engine`` by the lifespan hook in
    ``main.py``.  Override this dependency in tests via
    ``app.dependency_overrides[get_engine] = lambda: fake``.
    """
    # Use getattr so tests that skip the lifespan (ASGI transport doesn't run it)
    # still return 503 rather than AttributeError when engine is never set.
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail=_503_DETAIL)
    return engine
