"""Decode a base64 / data: URL image into a numpy array, enforcing input limits.

Rules (AGENTS.md section 3):
- Accept raw base64 or `data:image/...;base64,...`. Reject remote URLs (-> 400).
- Enforce max bytes before decode and max pixels before full raster decode.
- Never persist the image; never log raw bytes.
"""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np
from PIL import Image as PILImage

# Disable PIL's built-in decompression-bomb guard; we enforce our own configurable
# limit (max_pixels) before the full raster decode happens.
PILImage.MAX_IMAGE_PIXELS = None


class ImageDecodeError(ValueError):
    """Raised for all image input validation failures. Callers map this to HTTP 400."""


def decode_base64_image(
    image: str,
    max_bytes: int,
    max_pixels: int,
) -> np.ndarray:
    """Decode an attached base64 image to an RGB numpy array.

    Accepts raw base64 or a ``data:image/...;base64,...`` URL.
    Raises :exc:`ImageDecodeError` for any invalid, oversized, or remote-URL input.
    """
    # Reject remote URLs (AGENTS.md §3.4 — no SSRF surface)
    if image.lstrip().startswith(("http://", "https://")):
        raise ImageDecodeError(
            "Remote image URLs are not accepted. Attach the image as base64 or a data: URL."
        )

    # Strip optional data: URL prefix
    b64_data = image
    if image.startswith("data:"):
        parts = image.split(",", 1)
        if len(parts) != 2 or not parts[1]:
            raise ImageDecodeError("Malformed data: URL — expected 'data:<type>;base64,<data>'.")
        b64_data = parts[1]

    # Decode base64
    try:
        raw_bytes = base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError(f"Invalid base64 encoding: {exc}") from exc

    # Enforce byte limit BEFORE any raster decode
    if len(raw_bytes) > max_bytes:
        raise ImageDecodeError(
            f"Image size {len(raw_bytes)} bytes exceeds the {max_bytes}-byte limit."
        )

    # Open image header only (lazy — no full raster decode yet)
    try:
        pil_img = PILImage.open(io.BytesIO(raw_bytes))
        w, h = pil_img.size
    except Exception as exc:
        raise ImageDecodeError(f"Cannot open image: {exc}") from exc

    # Enforce pixel limit before full decode to cap memory / CPU
    if w * h > max_pixels:
        raise ImageDecodeError(
            f"Image resolution {w}x{h} ({w * h} pixels) exceeds the {max_pixels}-pixel limit."
        )

    # Full raster decode
    try:
        pil_img = pil_img.convert("RGB")
    except Exception as exc:
        raise ImageDecodeError(f"Cannot decode image: {exc}") from exc

    return np.array(pil_img)
