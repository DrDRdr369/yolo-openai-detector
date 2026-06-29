"""Native endpoint schemas — mirrors docs/api-contract.md section 3 exactly."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


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


class ImageSize(BaseModel):
    width: int
    height: int


class TimingMs(BaseModel):
    decode: float
    inference: float


class DetectRequest(BaseModel):
    model: str | None = None
    image: str  # required — raw base64 or data: URL
    conf_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    classes: list[int] | None = None

    @field_validator("classes")
    @classmethod
    def _classes_non_negative(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(c < 0 for c in v):
            raise ValueError("class IDs must be non-negative integers")
        return v


class DetectResponse(BaseModel):
    model: str
    image: ImageSize
    detections: list[Detection]
    timing_ms: TimingMs
