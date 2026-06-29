"""Shared detection pipeline: decode image → run inference → return raw results."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .imaging.decode import decode_base64_image
from .inference.engine import DetectionEngine


def run_detection(
    engine: DetectionEngine,
    image_b64: str,
    max_bytes: int,
    max_pixels: int,
    conf_threshold: float,
    iou_threshold: float,
    classes: list[int] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    """Decode *image_b64* and run *engine* inference; return (image, raw, decode_ms, infer_ms).

    Raises ImageDecodeError on invalid input — callers let it propagate to the 400 handler.
    """
    t0 = time.perf_counter()
    image = decode_base64_image(image_b64, max_bytes, max_pixels)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    raw = engine.infer(image, conf_threshold, iou_threshold, classes=classes)
    inference_ms = (time.perf_counter() - t1) * 1000.0

    return image, raw, decode_ms, inference_ms
