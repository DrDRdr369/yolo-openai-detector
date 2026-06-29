"""Image preprocessing for YOLO (letterbox resize, normalization, layout).

Implements: PR-1.
"""

from __future__ import annotations

import numpy as np


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio using padding.

    PR-1: return (preprocessed_tensor, scale, (pad_w, pad_h)) for postprocess to invert.
    """
    raise NotImplementedError("PR-1: implement letterbox preprocessing.")
