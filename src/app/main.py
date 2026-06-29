"""FastAPI application entry point.

Routing and startup only — keep this module thin.
PR-3 mounts POST /v1/chat/completions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.chat_completions import router as chat_completions_router
from .api.detections import router as detections_router
from .api.models_route import router as models_router
from .config import get_settings
from .errors import register_exception_handlers
from .inference.engine import DetectionEngine, ModelLoadError
from .logging_config import RequestLoggingMiddleware, configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()  # fails closed at startup if GATEWAY_API_KEY is unset
    configure_logging(settings.log_level)

    # Load detection model; if absent/corrupt, keep serving auth + health (fail-soft for model).
    try:
        app.state.engine = DetectionEngine(settings.model_path, settings.onnx_provider)
        logger.info("Detection model loaded from %s", settings.model_path)
    except ModelLoadError as exc:
        logger.warning(
            "Detection model could not be loaded (%s). "
            "/v1/detections will return 503 until restarted with a valid MODEL_PATH.",
            exc,
        )
        app.state.engine = None

    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="YOLO Vision Gateway", lifespan=_lifespan)

    register_exception_handlers(app)
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(models_router)
    app.include_router(detections_router)
    app.include_router(chat_completions_router)
    return app


app = create_app()
