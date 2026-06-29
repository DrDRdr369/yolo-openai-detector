"""POST /v1/detections — native, typed, stateless detection endpoint.

Request/response/error contracts: docs/api-contract.md sections 3 and 5.
- One base64 image (raw base64 or data: URL). Remote URLs -> 400.
- Input limits enforced BEFORE raster decode.
- Identical input always produces identical output (stateless, deterministic).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..config import Settings, get_settings
from ..dependencies import get_engine
from ..imaging.decode import decode_base64_image
from ..inference.engine import DetectionEngine
from ..schemas.detections import (
    Box,
    Detection,
    DetectRequest,
    DetectResponse,
    ImageSize,
    TimingMs,
)

router = APIRouter()


@router.post("/v1/detections")
async def detect(
    body: DetectRequest,
    _: None = Depends(require_api_key),
    engine: DetectionEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> DetectResponse:
    """Run detection on one attached base64 image and return typed detections."""
    conf = body.conf_threshold if body.conf_threshold is not None else settings.conf_threshold
    iou = body.iou_threshold if body.iou_threshold is not None else settings.iou_threshold

    # Decode and validate image (ImageDecodeError → 400 via error handler)
    t0 = time.perf_counter()
    image = decode_base64_image(body.image, settings.max_image_bytes, settings.max_image_pixels)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    orig_h, orig_w = image.shape[:2]

    # Inference
    t1 = time.perf_counter()
    raw = engine.infer(image, conf, iou, classes=body.classes)
    inference_ms = (time.perf_counter() - t1) * 1000.0

    detections = [
        Detection(
            class_id=d["class_id"],
            label=d["label"],
            confidence=d["confidence"],
            box=Box(**d["box"]),
        )
        for d in raw
    ]

    return DetectResponse(
        model=settings.model_id,
        image=ImageSize(width=orig_w, height=orig_h),
        detections=detections,
        timing_ms=TimingMs(decode=decode_ms, inference=inference_ms),
    )
