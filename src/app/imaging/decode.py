"""Decode a base64 / data: URL image into a numpy array, enforcing input limits.

Implements: PR-1.

Rules (AGENTS.md section 3):
- Accept raw base64 or `data:image/...;base64,...`. Reject remote URLs (-> 400).
- Enforce max bytes and max pixels BEFORE/at decode to bound CPU and memory.
- Never persist the image; never log raw bytes.
"""

from __future__ import annotations

import numpy as np


def decode_base64_image(
    image: str,
    max_bytes: int,
    max_pixels: int,
) -> np.ndarray:
    """Decode an attached base64 image to an RGB numpy array.

    PR-1: strip an optional data: URL prefix, base64-decode, enforce max_bytes, open with
    Pillow, enforce max_pixels, convert to RGB ndarray. Raise a validation error (mapped to
    HTTP 400) on any malformed/oversized/remote input.
    """
    raise NotImplementedError("PR-1: implement base64 image decode with limits.")
