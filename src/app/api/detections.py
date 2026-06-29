"""POST /v1/detections — native, typed, stateless detection endpoint.

Implements: PR-2.

Request/response/error contracts: docs/api-contract.md sections 3 and 5.
- One base64 image only (raw base64 or data: URL). Remote URLs -> 400.
- Enforce input limits BEFORE decode.
- Fully stateless: identical input -> identical output.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/v1/detections")
async def detect() -> dict:
    """Run detection on one attached base64 image and return typed detections.

    PR-2: validate the DetectRequest, decode the image (imaging.decode), run inference
    (inference.engine), and return a DetectResponse. Map all failure modes to the error
    model in docs/api-contract.md section 5.
    """
    raise NotImplementedError("PR-2: implement native detection endpoint.")
