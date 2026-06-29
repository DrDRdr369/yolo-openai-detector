"""Class-id → label map for the detection model.

Strategy: read class names from ONNX model metadata (key ``"names"``, JSON dict
``{"0": "person", ...}``), which Ultralytics writes at export time. Fall back to the
built-in COCO-80 list if the metadata is absent or unparseable.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Standard COCO-80 detection class names (index == class_id).
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

_COCO_MAP: dict[int, str] = {i: name for i, name in enumerate(COCO_CLASSES)}


def load_labels(session) -> dict[int, str]:  # session: onnxruntime.InferenceSession
    """Return a class-id → label mapping for *session*.

    Reads ``"names"`` from the ONNX model's custom metadata map (Ultralytics format).
    Falls back to :data:`COCO_CLASSES` if the metadata is absent or cannot be parsed.
    """
    try:
        meta = session.get_modelmeta()
        names_raw = meta.custom_metadata_map.get("names", "")
        if names_raw:
            parsed = json.loads(names_raw)
            labels = {int(k): str(v) for k, v in parsed.items()}
            logger.debug("Loaded %d class labels from ONNX model metadata.", len(labels))
            return labels
    except Exception as exc:
        logger.warning("Could not read class names from model metadata: %s. Using COCO-80.", exc)

    logger.debug("Using built-in COCO-80 class label list.")
    return dict(_COCO_MAP)
