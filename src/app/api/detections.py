"""POST /v1/detections — native, typed, stateless detection endpoint.

Request/response/error contracts: docs/api-contract.md sections 3 and 5.
- One base64 image (raw base64 or data: URL). Remote URLs -> 400.
- Input limits enforced BEFORE raster decode.
- Identical input always produces identical output (stateless, deterministic).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..auth import require_api_key
from ..config import Settings, get_settings
from ..dependencies import get_engine
from ..inference.engine import DetectionEngine
from ..schemas.detections import (
    Box,
    Detection,
    DetectRequest,
    DetectResponse,
    ImageSize,
    TimingMs,
)
from ..service import run_detection

router = APIRouter()


@router.post("/v1/detections")
async def detect(
    request: Request,
    body: DetectRequest,
    _: None = Depends(require_api_key),
    engine: DetectionEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> DetectResponse:
    """Run detection on one attached base64 image and return typed detections."""
    conf = body.conf_threshold if body.conf_threshold is not None else settings.conf_threshold
    iou = body.iou_threshold if body.iou_threshold is not None else settings.iou_threshold

    image, raw, decode_ms, inference_ms = run_detection(
        engine=engine,
        image_b64=body.image,
        max_bytes=settings.max_image_bytes,
        max_pixels=settings.max_image_pixels,
        conf_threshold=conf,
        iou_threshold=iou,
        classes=body.classes,
    )

    orig_h, orig_w = image.shape[:2]

    detections = [
        Detection(
            class_id=d["class_id"],
            label=d["label"],
            confidence=d["confidence"],
            box=Box(**d["box"]),
        )
        for d in raw
    ]

    request.state.detection_log = {
        "detection_count": len(detections),
        "image_width": orig_w,
        "image_height": orig_h,
        "model_id": settings.model_id,
    }

    return DetectResponse(
        model=settings.model_id,
        image=ImageSize(width=orig_w, height=orig_h),
        detections=detections,
        timing_ms=TimingMs(decode=decode_ms, inference=inference_ms),
    )
