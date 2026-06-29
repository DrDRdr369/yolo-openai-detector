"""Unit tests for src/app/imaging/decode.py — no model required."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image as PILImage  # noqa: F401 (used via PILImage.new)

from app.imaging.decode import ImageDecodeError, decode_base64_image

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_MAX_BYTES = 10_000_000
_MAX_PIXELS = 4096 * 4096


def _make_png_b64(width: int = 64, height: int = 64, color: tuple = (70, 130, 180)) -> str:
    """Return a base64-encoded PNG of the given size and solid color."""
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_data_url(b64: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{b64}"


VALID_B64 = _make_png_b64()
VALID_DATA_URL = _make_data_url(VALID_B64)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_raw_base64_returns_rgb_array():
    """Raw base64 string → RGB ndarray with correct shape."""
    arr = decode_base64_image(VALID_B64, _MAX_BYTES, _MAX_PIXELS)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3  # RGB


def test_data_url_returns_rgb_array():
    """data: URL with base64 payload → same result as raw base64."""
    arr = decode_base64_image(VALID_DATA_URL, _MAX_BYTES, _MAX_PIXELS)
    assert arr.shape == (64, 64, 3)


def test_returned_array_has_correct_dimensions():
    """Returned array dimensions match the source image."""
    b64 = _make_png_b64(width=100, height=200)
    arr = decode_base64_image(b64, _MAX_BYTES, _MAX_PIXELS)
    assert arr.shape == (200, 100, 3)


# ---------------------------------------------------------------------------
# Fail-closed: remote URLs
# ---------------------------------------------------------------------------


def test_http_url_raises():
    """http:// URL must be rejected immediately."""
    with pytest.raises(ImageDecodeError, match="Remote image"):
        decode_base64_image("http://example.com/img.jpg", _MAX_BYTES, _MAX_PIXELS)


def test_https_url_raises():
    """https:// URL must be rejected immediately."""
    with pytest.raises(ImageDecodeError, match="Remote image"):
        decode_base64_image("https://example.com/img.jpg", _MAX_BYTES, _MAX_PIXELS)


# ---------------------------------------------------------------------------
# Fail-closed: oversized input
# ---------------------------------------------------------------------------


def test_oversized_bytes_raises():
    """Image whose decoded byte length exceeds max_bytes must raise."""
    b64 = _make_png_b64()
    with pytest.raises(ImageDecodeError, match="exceeds the"):
        decode_base64_image(b64, max_bytes=10, max_pixels=_MAX_PIXELS)


def test_oversized_pixels_raises():
    """Image whose pixel count exceeds max_pixels must raise before full decode."""
    b64 = _make_png_b64(width=64, height=64)  # 4096 pixels
    with pytest.raises(ImageDecodeError, match="pixel"):
        decode_base64_image(b64, max_bytes=_MAX_BYTES, max_pixels=100)


# ---------------------------------------------------------------------------
# Fail-closed: corrupt / invalid input
# ---------------------------------------------------------------------------


def test_corrupt_base64_raises():
    """Random bytes that are not an image must raise ImageDecodeError."""
    junk_b64 = base64.b64encode(b"this is not an image file at all!!!").decode()
    with pytest.raises(ImageDecodeError):
        decode_base64_image(junk_b64, _MAX_BYTES, _MAX_PIXELS)


def test_invalid_base64_encoding_raises():
    """Non-base64 string must raise ImageDecodeError."""
    with pytest.raises(ImageDecodeError):
        decode_base64_image("not-valid-base64!!!", _MAX_BYTES, _MAX_PIXELS)


def test_malformed_data_url_raises():
    """data: URL without a comma must raise."""
    with pytest.raises(ImageDecodeError, match="Malformed"):
        decode_base64_image("data:image/png;base64", _MAX_BYTES, _MAX_PIXELS)
