"""YOLO output decoding + non-maximum suppression.

Expected raw model output shape: (1, 4 + num_classes, num_predictions).
For YOLO11n / YOLOv8n with COCO-80: (1, 84, 8400).
Box values are in the letterboxed image coordinate space (absolute pixels, not
normalized), in cx/cy/w/h format. Class scores are post-sigmoid (0–1).
"""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Return indices of boxes kept after IoU-based non-maximum suppression.

    Parameters
    ----------
    boxes : np.ndarray
        Shape (N, 4) — xyxy absolute pixel coordinates.
    scores : np.ndarray
        Shape (N,) — confidence scores in descending order will be processed.
    iou_threshold : float
        Boxes with IoU > this threshold relative to a kept box are suppressed.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    kept: list[int] = []

    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return kept


def decode_predictions(
    raw_output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    scale: float,
    pad_w: int,
    pad_h: int,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLO11/v8 ONNX output into original-image absolute-pixel coordinates.

    Parameters
    ----------
    raw_output : np.ndarray
        Shape (1, 4+C, N) from the ONNX session.
    conf_threshold, iou_threshold : float
        Detection and NMS thresholds.
    scale : float
        Letterbox scale factor (from :func:`~app.inference.preprocess.letterbox`).
    pad_w, pad_h : int
        Left/top padding added by letterbox.
    orig_h, orig_w : int
        Original image dimensions before letterboxing.

    Returns
    -------
    boxes_xyxy : np.ndarray  shape (M, 4)  — absolute pixels in original image space
    confidences : np.ndarray  shape (M,)
    class_ids : np.ndarray   shape (M,)   — int
    """
    # Squeeze batch and transpose: (1, 84, 8400) → (8400, 84)
    out = raw_output[0].T  # (N, 4+C)

    boxes_xywh = out[:, :4]
    class_scores = out[:, 4:]  # (N, C)

    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence before NMS
    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty, empty.astype(np.int32)

    # cx/cy/w/h → x1/y1/x2/y2 in letterbox pixel space
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    # Per-class NMS
    kept_indices: list[int] = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_idx = np.where(cls_mask)[0]
        kept = nms(boxes_xyxy[cls_mask], confidences[cls_mask], iou_threshold)
        kept_indices.extend(cls_idx[kept].tolist())

    if not kept_indices:
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty, empty.astype(np.int32)

    boxes_xyxy = boxes_xyxy[kept_indices].copy()
    confidences = confidences[kept_indices]
    class_ids = class_ids[kept_indices].astype(np.int32)

    # Invert letterbox: remove padding then divide by scale → original pixel coords
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / scale

    # Clip to original image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, float(orig_w))
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, float(orig_h))

    return boxes_xyxy, confidences, class_ids
