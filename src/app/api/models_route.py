"""GET /v1/models — authenticated readiness probe + model list.

Returns 200 + the loaded model when the engine is ready.
Returns 503 when the model is not loaded (same fail-closed contract as /v1/detections).
Returns 401 when the API key is missing or wrong.

Response shape: see docs/api-contract.md section 2.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..config import get_settings
from ..dependencies import get_engine
from ..inference.engine import DetectionEngine

router = APIRouter()


@router.get("/v1/models")
async def list_models(
    _: None = Depends(require_api_key),
    engine: DetectionEngine = Depends(get_engine),
) -> dict:
    """Return the loaded model in OpenAI `{object: list, data: [...]}` shape.

    Raises HTTP 503 (via ``get_engine``) when the model is not loaded, making
    this endpoint a genuine **authenticated readiness** probe — not just a config echo.
    """
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "owned_by": "local",
                "created": 0,
            }
        ],
    }
