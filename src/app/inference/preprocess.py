"""Image preprocessing for YOLO (letterbox resize, normalization, layout).

Normalization: pixel values divided by 255.0 → [0, 1] float32.
Layout: HWC RGB numpy array → NCHW float32 tensor (batch=1).
Padding fill value: 114 (YOLO convention).
"""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage


def letterbox(
    image: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize an HWC RGB image to a square canvas while preserving aspect ratio.

    Returns
    -------
    tensor : np.ndarray
        NCHW float32 array of shape (1, 3, size, size), values in [0, 1].
    scale : float
        Scale factor applied to the original image.
    (pad_w, pad_h) : tuple[int, int]
        Pixels of padding added to the *left* and *top* sides respectively.
        Postprocessing subtracts these before dividing by scale to recover
        original-image coordinates.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w = round(w * scale)
    new_h = round(h * scale)

    pil_img = PILImage.fromarray(image)
    pil_img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
    resized = np.array(pil_img)

    # Fill letterbox canvas with YOLO padding value
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # HWC → CHW → NCHW, normalize
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # (1, 3, size, size)

    return tensor, scale, (pad_w, pad_h)
