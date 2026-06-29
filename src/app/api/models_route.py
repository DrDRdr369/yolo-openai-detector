"""GET /v1/models — list the loaded detection model in OpenAI list shape.

Response shape: see docs/api-contract.md section 2.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..config import get_settings

router = APIRouter()


@router.get("/v1/models")
async def list_models(_: None = Depends(require_api_key)) -> dict:
    """Return the configured model in OpenAI `{object: list, data: [...]}` shape."""
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
