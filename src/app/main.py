"""FastAPI application entry point.

Routing and startup only — keep this module thin.
PR-2 mounts POST /v1/detections; PR-3 mounts POST /v1/chat/completions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.models_route import router as models_router
from .config import get_settings


@asynccontextmanager
async def _lifespan(app: FastAPI):
    get_settings()  # fails closed at startup if GATEWAY_API_KEY is unset
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="YOLO Vision Gateway", lifespan=_lifespan)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(models_router)
    return app


app = create_app()
