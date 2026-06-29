"""Centralized exception → HTTP response mapping.

Registers FastAPI/Starlette exception handlers that emit the OpenAI-style error
body from docs/api-contract.md §5 for every failure mode.  Called once in
create_app(); PR-3 reuses this without any changes.

Error body shape (all statuses)::

    {"error": {"type": "<type>", "message": "<human-readable>"}}
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .imaging.decode import ImageDecodeError


def _body(type_: str, message: str) -> dict:
    return {"error": {"type": type_, "message": message}}


async def _http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Re-emit HTTPExceptions in OpenAI error-body format.

    Our code always sets ``detail`` to an ``{"error": {...}}`` dict; fall back
    to a generic ``server_error`` for anything unexpected (e.g. 404 from FastAPI).
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = detail
    else:
        body = _body("server_error", str(detail) if detail else "An unexpected error occurred.")
    return JSONResponse(status_code=exc.status_code, content=body)


async def _validation_exc_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map Pydantic RequestValidationError → 400 invalid_request_error."""
    errors = exc.errors()
    parts = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors]
    message = "; ".join(parts)
    return JSONResponse(status_code=400, content=_body("invalid_request_error", message))


async def _image_decode_handler(request: Request, exc: ImageDecodeError) -> JSONResponse:
    """Map ImageDecodeError → 400 invalid_request_error."""
    return JSONResponse(status_code=400, content=_body("invalid_request_error", str(exc)))


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all project exception handlers to *app*."""
    app.add_exception_handler(HTTPException, _http_exc_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exc_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ImageDecodeError, _image_decode_handler)  # type: ignore[arg-type]
