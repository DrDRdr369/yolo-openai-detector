"""Structured per-request logging middleware and log-level configuration.

SECURITY INVARIANT — this module MUST NOT log:
  - The GATEWAY_API_KEY value or Authorization header
  - Raw image bytes or base64 image data
  - Request body content

Only metadata derived from the completed response is recorded:
method, path, status, latency_ms, and (for detection endpoints)
detection_count, image_width, image_height, model_id — all stored
in request.state.detection_log by the route handlers.
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .config import get_settings

_logger = logging.getLogger("app.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured log line per request; also guards oversized bodies."""

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()

        content_length_str = request.headers.get("content-length")
        if content_length_str:
            try:
                content_length = int(content_length_str)
            except ValueError:
                content_length = 0
            if content_length > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Request body too large.",
                        }
                    },
                )

        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Route handlers store safe metadata in request.state.detection_log.
        # This dict MUST contain only metadata — never secrets or image bytes.
        meta: dict = getattr(request.state, "detection_log", {})

        _logger.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                **meta,
            },
        )
        return response


def configure_logging(level: str) -> None:
    """Set level on the app logger hierarchy. Idempotent; safe to call in tests."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("app").setLevel(numeric)
