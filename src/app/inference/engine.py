"""ONNX Runtime detection engine (CPU).

The training framework (ultralytics / torch) is NOT imported here.
The serving path uses onnxruntime only (AGENTS.md §3, stack law).
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from .labels import load_labels
from .postprocess import decode_predictions
from .preprocess import letterbox

_logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the ONNX model cannot be loaded. Callers map this to HTTP 503."""


_PROVIDER_MAP: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
}


class DetectionEngine:
    """Wraps an onnxruntime InferenceSession for a YOLO detection model."""

    def __init__(self, model_path: str, provider: str = "cpu") -> None:
        """Create the InferenceSession.

        Parameters
        ----------
        model_path : str
            Path to the exported ONNX detection model.
        provider : str
            ``"cpu"`` (default) or ``"openvino"``.  Falls back to CPU if the
            requested provider is unavailable.

        Raises
        ------
        ModelLoadError
            If the model file cannot be opened or is not a valid ONNX graph.
        """
        providers = _PROVIDER_MAP.get(provider.lower(), _PROVIDER_MAP["cpu"])

        if provider.lower() == "openvino":
            available = ort.get_available_providers()
            if "OpenVINOExecutionProvider" not in available:
                _logger.warning(
                    "ONNX_PROVIDER=openvino requested but OpenVINOExecutionProvider is not "
                    "installed (available: %s); session will use CPU. "
                    "Install onnxruntime-openvino to enable OpenVINO acceleration.",
                    available,
                )

        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load detection model from {model_path!r}: {exc}"
            ) from exc

        self._labels = load_labels(self._session)
        self._input_name: str = self._session.get_inputs()[0].name

        # Determine the spatial size the model expects (default 640)
        input_shape = self._session.get_inputs()[0].shape
        self._input_size: int = int(input_shape[2]) if len(input_shape) >= 3 else 640

    def infer(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int] | None = None,
    ) -> list[dict]:
        """Run detection and return a list of detection dicts.

        Parameters
        ----------
        image : np.ndarray
            HWC RGB uint8 array (from :func:`~app.imaging.decode.decode_base64_image`).
        conf_threshold, iou_threshold : float
            Confidence and NMS IoU thresholds.
        classes : list[int] | None
            Optional allowlist of class IDs; ``None`` returns all classes.

        Returns
        -------
        list[dict]
            Each dict follows the schema in ``docs/api-contract.md`` §3::

                {
                    "class_id": int,
                    "label": str,
                    "confidence": float,
                    "box": {"x1": float, "y1": float, "x2": float, "y2": float},
                }
            Boxes are absolute pixel coordinates in the *original* image space.
        """
        orig_h, orig_w = image.shape[:2]
        tensor, scale, (pad_w, pad_h) = letterbox(image, self._input_size)

        raw_outputs = self._session.run(None, {self._input_name: tensor})
        raw_output = raw_outputs[0]

        boxes, confs, class_ids = decode_predictions(
            raw_output,
            conf_threshold,
            iou_threshold,
            scale,
            pad_w,
            pad_h,
            orig_h,
            orig_w,
        )

        detections: list[dict] = []
        for box, conf, cls_id in zip(boxes, confs, class_ids, strict=False):
            cid = int(cls_id)
            if classes is not None and cid not in classes:
                continue
            detections.append(
                {
                    "class_id": cid,
                    "label": self._labels.get(cid, str(cid)),
                    "confidence": float(conf),
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                }
            )

        return detections
