"""POST /v1/detections endpoint tests.

Unit tests (always run): inject FakeDetectionEngine via dependency override —
no real ONNX model needed.

Integration test (optional): skipped when the exported model is absent.
"""

from __future__ import annotations

import base64
import io
import os

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage

from app.config import get_settings

_MODEL_PATH = os.environ.get("MODEL_PATH", "models/yolo11n.onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_b64(width: int = 64, height: int = 64, color: tuple = (70, 130, 180)) -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _auth(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


async def test_detect_happy_path(detection_client, api_key, fake_engine):
    """Valid base64 image + valid key → 200 with well-formed DetectResponse."""
    payload = {"image": _png_b64()}
    response = await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    assert response.status_code == 200
    body = response.json()

    # Top-level fields
    assert "model" in body
    assert "image" in body
    assert "detections" in body
    assert "timing_ms" in body

    # Image dimensions
    assert body["image"]["width"] == 64
    assert body["image"]["height"] == 64

    # Timing recorded
    assert body["timing_ms"]["decode"] >= 0.0
    assert body["timing_ms"]["inference"] >= 0.0

    # Detections forwarded from fake engine
    assert len(body["detections"]) == 1
    det = body["detections"][0]
    assert det["class_id"] == 0
    assert det["label"] == "person"
    assert abs(det["confidence"] - 0.91) < 1e-6
    assert det["box"] == {"x1": 100.0, "y1": 50.0, "x2": 220.0, "y2": 400.0}


async def test_threshold_overrides_forwarded(detection_client, api_key, fake_engine):
    """Custom thresholds in request are forwarded to engine.infer exactly."""
    payload = {"image": _png_b64(), "conf_threshold": 0.7, "iou_threshold": 0.3}
    response = await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    assert response.status_code == 200
    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert abs(call["conf_threshold"] - 0.7) < 1e-9
    assert abs(call["iou_threshold"] - 0.3) < 1e-9


async def test_classes_filter_forwarded(detection_client, api_key, fake_engine):
    """classes list in request is forwarded to engine.infer unchanged."""
    payload = {"image": _png_b64(), "classes": [0, 2]}
    response = await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    assert response.status_code == 200
    assert fake_engine.calls[0]["classes"] == [0, 2]


async def test_no_classes_passes_none(detection_client, api_key, fake_engine):
    """Omitting classes in request forwards None (return all classes)."""
    payload = {"image": _png_b64()}
    await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    assert fake_engine.calls[0]["classes"] is None


async def test_stateless_identical_requests(detection_client, api_key):
    """Identical requests always produce identical responses (stateless)."""
    payload = {"image": _png_b64()}
    r1 = await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    r2 = await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    assert r1.status_code == 200
    assert r1.json()["detections"] == r2.json()["detections"]


async def test_default_thresholds_from_settings(detection_client, api_key, fake_engine):
    """When thresholds are omitted, settings defaults are passed to engine."""
    from app.config import get_settings

    settings = get_settings()
    payload = {"image": _png_b64()}
    await detection_client.post("/v1/detections", json=payload, headers=_auth(api_key))
    call = fake_engine.calls[0]
    assert abs(call["conf_threshold"] - settings.conf_threshold) < 1e-9
    assert abs(call["iou_threshold"] - settings.iou_threshold) < 1e-9


# ---------------------------------------------------------------------------
# Fail-closed: auth
# ---------------------------------------------------------------------------


async def test_missing_key_returns_401(detection_client):
    """No Authorization header → 401 authentication_error."""
    response = await detection_client.post("/v1/detections", json={"image": _png_b64()})
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


async def test_wrong_key_returns_401(detection_client):
    """Wrong bearer token → 401 authentication_error."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": _png_b64()},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


# ---------------------------------------------------------------------------
# Fail-closed: image validation (400)
# ---------------------------------------------------------------------------


async def test_bad_base64_returns_400(detection_client, api_key):
    """Non-base64 string in image field → 400 invalid_request_error."""
    response = await detection_client.post(
        "/v1/detections", json={"image": "not-valid-base64!!!"}, headers=_auth(api_key)
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_non_image_bytes_returns_400(detection_client, api_key):
    """Valid base64 but not an image → 400."""
    junk_b64 = base64.b64encode(b"this is definitely not an image file").decode()
    response = await detection_client.post(
        "/v1/detections", json={"image": junk_b64}, headers=_auth(api_key)
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_oversized_image_returns_400(detection_client, api_key, monkeypatch):
    """Image exceeding max_bytes limit → 400."""
    # Temporarily lower the limit so our small PNG exceeds it
    monkeypatch.setenv("MAX_IMAGE_BYTES", "10")
    get_settings.cache_clear()
    response = await detection_client.post(
        "/v1/detections", json={"image": _png_b64()}, headers=_auth(api_key)
    )
    get_settings.cache_clear()
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_remote_url_returns_400(detection_client, api_key):
    """http:// URL in image field → 400 (no SSRF)."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": "http://example.com/img.jpg"},
        headers=_auth(api_key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_https_url_returns_400(detection_client, api_key):
    """https:// URL in image field → 400 (no SSRF)."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": "https://example.com/img.jpg"},
        headers=_auth(api_key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_missing_image_field_returns_400(detection_client, api_key):
    """Request body without required image field → 400 invalid_request_error."""
    response = await detection_client.post(
        "/v1/detections", json={"model": "yolo11n"}, headers=_auth(api_key)
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Fail-closed: parameter validation (400)
# ---------------------------------------------------------------------------


async def test_conf_threshold_out_of_range_returns_400(detection_client, api_key):
    """conf_threshold > 1.0 → 400 invalid_request_error."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": _png_b64(), "conf_threshold": 2.0},
        headers=_auth(api_key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_iou_threshold_negative_returns_400(detection_client, api_key):
    """iou_threshold < 0 → 400 invalid_request_error."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": _png_b64(), "iou_threshold": -0.1},
        headers=_auth(api_key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_negative_class_id_returns_400(detection_client, api_key):
    """Negative value in classes list → 400 invalid_request_error."""
    response = await detection_client.post(
        "/v1/detections",
        json={"image": _png_b64(), "classes": [-1]},
        headers=_auth(api_key),
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Fail-closed: engine unavailable (503)
# ---------------------------------------------------------------------------


async def test_engine_unavailable_returns_503(monkeypatch, api_key):
    """When engine cannot be loaded, /v1/detections returns 503 server_error."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from fastapi import HTTPException

    from app.dependencies import get_engine
    from app.main import create_app

    app_instance = create_app()

    def _unavailable():
        raise HTTPException(
            status_code=503,
            detail={"error": {"type": "server_error", "message": "Model not loaded."}},
        )

    app_instance.dependency_overrides[get_engine] = _unavailable

    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as c:
        response = await c.post(
            "/v1/detections", json={"image": _png_b64()}, headers=_auth(api_key)
        )

    get_settings.cache_clear()
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"


# ---------------------------------------------------------------------------
# Real-model integration test (skipped when model absent)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.isfile(_MODEL_PATH),
    reason=f"Integration test skipped: model not found at {_MODEL_PATH!r}. "
    "Run `make test-integration` to export the model and run this test.",
)
async def test_golden_image_detection(api_key, monkeypatch, valid_image_b64):
    """End-to-end detection with the real ONNX model — format and schema correctness."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from app.main import create_app

    app_instance = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as c:
        response = await c.post(
            "/v1/detections", json={"image": valid_image_b64}, headers=_auth(api_key)
        )

    get_settings.cache_clear()
    assert response.status_code == 200
    body = response.json()
    assert "model" in body
    assert body["image"]["width"] == 64
    assert body["image"]["height"] == 64
    assert isinstance(body["detections"], list)
    for det in body["detections"]:
        assert all(k in det for k in ("class_id", "label", "confidence", "box"))
        assert 0.0 <= det["confidence"] <= 1.0
        assert det["box"]["x1"] <= det["box"]["x2"]
        assert det["box"]["y1"] <= det["box"]["y2"]
