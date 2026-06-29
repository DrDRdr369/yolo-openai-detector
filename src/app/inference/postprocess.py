"""YOLO output decoding + non-maximum suppression.

Implements: PR-1.
"""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Return indices kept after non-maximum suppression.

    PR-1: implement IoU-based NMS (pure numpy is fine for CPU).
    """
    raise NotImplementedError("PR-1: implement NMS.")
