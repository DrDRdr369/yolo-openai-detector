"""ONNX Runtime detection engine (CPU).

Implements: PR-1.

Loads an exported YOLO ONNX model and runs detection. The training framework
(ultralytics/torch) must NOT be imported here — serving uses onnxruntime only.
"""

from __future__ import annotations

import numpy as np


class DetectionEngine:
    """Wraps an onnxruntime InferenceSession for a YOLO detection model."""

    def __init__(self, model_path: str, provider: str = "cpu") -> None:
        """PR-1: create the InferenceSession with the CPU (or OpenVINO) provider.

        Raise a clear error if the model cannot be loaded; callers map this to HTTP 503.
        """
        raise NotImplementedError("PR-1: init onnxruntime session (CPU/OpenVINO).")

    def infer(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int] | None = None,
    ) -> list[dict]:
        """Run detection and return a list of detection dicts.

        PR-1: preprocess (letterbox), run the session, postprocess (decode + NMS), filter by
        confidence/class, and return boxes in absolute pixel coordinates.
        """
        raise NotImplementedError("PR-1: run inference and return detections.")
