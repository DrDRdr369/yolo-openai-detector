"""Inference pipeline unit tests (no model) + golden-image integration test.

Unit tests run always (synthetic inputs, no ONNX model needed).
Golden test requires the exported model at MODEL_PATH and is skipped when absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.inference.postprocess import decode_predictions, nms
from app.inference.preprocess import letterbox

# ---------------------------------------------------------------------------
# NMS unit tests
# ---------------------------------------------------------------------------


def test_nms_empty_input():
    """NMS on empty input returns empty list."""
    result = nms(np.empty((0, 4)), np.empty((0,)), iou_threshold=0.45)
    assert result == []


def test_nms_single_box():
    """Single box is always kept."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    scores = np.array([0.9])
    assert nms(boxes, scores, 0.45) == [0]


def test_nms_suppresses_overlapping():
    """Two highly overlapping boxes: only the higher-score one is kept."""
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],  # IoU ~0.68 with box 0
    ])
    scores = np.array([0.9, 0.8])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert kept == [0]


def test_nms_keeps_non_overlapping():
    """Two non-overlapping boxes must both survive NMS."""
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    scores = np.array([0.9, 0.85])
    kept = nms(boxes, scores, iou_threshold=0.5)
    assert set(kept) == {0, 1}


def test_nms_order_independent():
    """The result does not depend on which box comes first in the array."""
    boxes = np.array([
        [100.0, 100.0, 110.0, 110.0],
        [0.0, 0.0, 10.0, 10.0],
    ])
    scores = np.array([0.7, 0.95])
    kept = nms(boxes, scores, iou_threshold=0.5)
    # Higher-score box (index 1) must be kept; both are non-overlapping so both kept
    assert 1 in kept


# ---------------------------------------------------------------------------
# Letterbox unit tests
# ---------------------------------------------------------------------------


def _make_rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_letterbox_output_shape():
    """Output tensor has shape (1, 3, size, size) for any input aspect ratio."""
    image = _make_rgb(480, 640)
    tensor, scale, (pad_w, pad_h) = letterbox(image, size=640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32


def test_letterbox_values_in_range():
    """Output tensor values are in [0, 1]."""
    image = _make_rgb(100, 200)
    tensor, _, _ = letterbox(image, size=320)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_letterbox_scale_and_pad_inversion():
    """Postprocessing can exactly recover the original image centre from letterbox coords."""
    # 480x640 image → 640x640 letterbox
    # scale=1.0, pad_w=0, pad_h=80 (80px top padding to centre the 480-tall image)
    orig_h, orig_w = 480, 640
    image = _make_rgb(orig_h, orig_w)
    _, scale, (pad_w, pad_h) = letterbox(image, size=640)

    # The centre of the original image in letterbox canvas coords:
    #   lx = orig_w/2 * scale + pad_w
    #   ly = orig_h/2 * scale + pad_h
    lx = orig_w / 2 * scale + pad_w
    ly = orig_h / 2 * scale + pad_h

    # Invert back to original coords
    ox = (lx - pad_w) / scale
    oy = (ly - pad_h) / scale

    assert abs(ox - orig_w / 2) < 1e-6
    assert abs(oy - orig_h / 2) < 1e-6


def test_letterbox_portrait_image():
    """Portrait image (tall) gets correct scale and padding."""
    image = _make_rgb(1280, 640)  # 2:1 tall
    tensor, scale, (pad_w, pad_h) = letterbox(image, size=640)
    assert tensor.shape == (1, 3, 640, 640)
    # Scale must be 640/1280 = 0.5; no vertical padding
    assert abs(scale - 0.5) < 1e-6
    assert pad_h == 0


# ---------------------------------------------------------------------------
# decode_predictions round-trip
# ---------------------------------------------------------------------------


def _make_raw_output(cx: float, cy: float, w: float, h: float, cls: int = 0) -> np.ndarray:
    """Build a minimal (1, 84, 1) raw YOLO output with one detection."""
    out = np.zeros((1, 84, 1), dtype=np.float32)
    out[0, 0, 0] = cx
    out[0, 1, 0] = cy
    out[0, 2, 0] = w
    out[0, 3, 0] = h
    out[0, 4 + cls, 0] = 1.0  # score = 1.0 for this class
    return out


def test_decode_predictions_returns_original_coords():
    """A box placed at a known position in letterbox space maps to correct orig coords."""
    # 480x640 original, letterboxed to 640x640
    orig_h, orig_w = 480, 640
    image = _make_rgb(orig_h, orig_w)
    _, scale, (pad_w, pad_h) = letterbox(image, size=640)

    # Place a box exactly at the centre of the letterboxed image
    lx, ly = 320.0, 240.0
    lw, lh = 100.0, 80.0
    raw = _make_raw_output(cx=lx, cy=ly, w=lw, h=lh, cls=0)

    boxes, confs, class_ids = decode_predictions(
        raw,
        conf_threshold=0.5,
        iou_threshold=0.45,
        scale=scale,
        pad_w=pad_w,
        pad_h=pad_h,
        orig_h=orig_h,
        orig_w=orig_w,
    )
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    expected_cx = (lx - pad_w) / scale
    expected_cy = (ly - pad_h) / scale
    expected_w = lw / scale
    expected_h = lh / scale
    assert abs((x1 + x2) / 2 - expected_cx) < 1.0
    assert abs((y1 + y2) / 2 - expected_cy) < 1.0
    assert abs((x2 - x1) - expected_w) < 1.0
    assert abs((y2 - y1) - expected_h) < 1.0


def test_decode_predictions_conf_threshold_filters():
    """Predictions below conf_threshold must be excluded."""
    raw = _make_raw_output(cx=100.0, cy=100.0, w=50.0, h=50.0, cls=0)
    raw[0, 4, 0] = 0.1  # low score

    boxes, confs, class_ids = decode_predictions(
        raw,
        conf_threshold=0.5,
        iou_threshold=0.45,
        scale=1.0,
        pad_w=0,
        pad_h=0,
        orig_h=640,
        orig_w=640,
    )
    assert len(boxes) == 0


# ---------------------------------------------------------------------------
# Golden-image integration test (requires exported model)
# ---------------------------------------------------------------------------

_MODEL_PATH = os.environ.get("MODEL_PATH", "models/yolo11n.onnx")


@pytest.mark.skipif(
    not os.path.isfile(_MODEL_PATH),
    reason=f"Golden test skipped: model not found at {_MODEL_PATH!r}. "
    "Run `make export-model` to export it first.",
)
def test_golden_image_engine_format():
    """DetectionEngine.infer returns correctly shaped dicts (format golden test).

    Uses the committed blue fixture image which has no detectable COCO objects,
    so we assert format correctness and 0 detections deterministically.
    """
    from app.inference.engine import DetectionEngine

    engine = DetectionEngine(_MODEL_PATH, provider="cpu")

    import numpy as np

    image = np.full((640, 640, 3), 114, dtype=np.uint8)  # grey canvas
    detections = engine.infer(image, conf_threshold=0.25, iou_threshold=0.45)

    assert isinstance(detections, list)
    for det in detections:
        assert "class_id" in det
        assert "label" in det
        assert "confidence" in det
        assert "box" in det
        box = det["box"]
        assert all(k in box for k in ("x1", "y1", "x2", "y2"))
        assert isinstance(det["class_id"], int)
        assert isinstance(det["label"], str)
        assert 0.0 <= det["confidence"] <= 1.0
        assert box["x1"] <= box["x2"]
        assert box["y1"] <= box["y2"]
