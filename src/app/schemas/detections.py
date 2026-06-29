"""Native endpoint schemas.

Implements: PR-2.

Mirror docs/api-contract.md section 3 exactly.
"""

from __future__ import annotations

from pydantic import BaseModel


class Box(BaseModel):
    """Absolute pixel coordinates, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    label: str
    confidence: float
    box: Box


class DetectRequest(BaseModel):
    """PR-2: define fields per contract.

    Expected fields: model (str | None), image (str, base64/data: URL, required),
    conf_threshold (float | None), iou_threshold (float | None), classes (list[int] | None).
    """


class DetectResponse(BaseModel):
    """PR-2: define fields per contract.

    Expected fields: model, image (width/height), detections (list[Detection]),
    timing_ms (decode/inference).
    """
