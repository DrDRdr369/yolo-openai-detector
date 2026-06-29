"""Logging middleware and OpenVINO fallback tests.

Key security invariants tested:
  - Log records contain only safe metadata (method, path, status, latency_ms,
    detection_count, image dimensions, model_id).
  - No log record ever contains the API key, the Authorization header value,
    or any image bytes.
"""

from __future__ import annotations

import logging

import pytest

from app.config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Structured fields in log records
# ---------------------------------------------------------------------------


async def test_request_log_contains_metadata_fields(
    detection_client, api_key, valid_image_b64, caplog
):
    """app.request logger emits method, path, status, latency_ms on each request."""
    with caplog.at_level(logging.INFO, logger="app.request"):
        resp = await detection_client.post(
            "/v1/detections",
            json={"image": valid_image_b64},
            headers=_auth(api_key),
        )
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) >= 1, "No log records emitted by app.request logger"

    r = records[0]
    assert getattr(r, "method", None) == "POST"
    assert getattr(r, "path", None) == "/v1/detections"
    assert getattr(r, "status", None) == 200
    assert isinstance(getattr(r, "latency_ms", None), float)
    assert r.latency_ms >= 0.0


async def test_detection_log_contains_detection_metadata(
    detection_client, api_key, valid_image_b64, caplog
):
    """Detection endpoint stores count and image dimensions in the log record."""
    with caplog.at_level(logging.INFO, logger="app.request"):
        resp = await detection_client.post(
            "/v1/detections",
            json={"image": valid_image_b64},
            headers=_auth(api_key),
        )
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) >= 1

    r = records[0]
    assert getattr(r, "detection_count", None) == 1  # FakeDetectionEngine: 1 canned detection
    assert getattr(r, "image_width", None) == 64   # 64x64 test fixture
    assert getattr(r, "image_height", None) == 64
    assert getattr(r, "model_id", None) is not None


async def test_chat_log_contains_detection_metadata(
    detection_client, api_key, valid_image_b64, caplog
):
    """Chat completions endpoint also stores detection metadata in the log record."""
    with caplog.at_level(logging.INFO, logger="app.request"):
        resp = await detection_client.post(
            "/v1/chat/completions",
            json={
                "model": "yolo11n",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{valid_image_b64}"
                                },
                            }
                        ],
                    }
                ],
            },
            headers=_auth(api_key),
        )
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "app.request"]
    assert len(records) >= 1
    r = records[0]
    assert getattr(r, "detection_count", None) == 1
    assert getattr(r, "image_width", None) == 64
    assert getattr(r, "image_height", None) == 64


# ---------------------------------------------------------------------------
# Security: log scrub — the critical invariant
# ---------------------------------------------------------------------------


async def test_log_never_contains_api_key(detection_client, api_key, valid_image_b64, caplog):
    """SECURITY: API key must never appear in any log record message or extra attrs."""
    with caplog.at_level(logging.DEBUG):
        await detection_client.post(
            "/v1/detections",
            json={"image": valid_image_b64},
            headers=_auth(api_key),
        )

    for record in caplog.records:
        msg = record.getMessage()
        assert api_key not in msg, f"API key leaked in log message: {msg!r}"
        for val in vars(record).values():
            assert api_key not in str(val), "API key leaked in log record attribute"


async def test_log_never_contains_image_bytes(
    detection_client, api_key, valid_image_b64, caplog
):
    """SECURITY: Image bytes (base64) must never appear in any log record."""
    # First 32 chars of base64 are long enough to be a false-positive-free fingerprint.
    image_fingerprint = valid_image_b64[:32]

    with caplog.at_level(logging.DEBUG):
        await detection_client.post(
            "/v1/detections",
            json={"image": valid_image_b64},
            headers=_auth(api_key),
        )

    for record in caplog.records:
        msg = record.getMessage()
        assert image_fingerprint not in msg, "Image bytes leaked in log message"
        for val in vars(record).values():
            assert image_fingerprint not in str(val), "Image bytes leaked in log record attr"


async def test_log_never_contains_authorization_header(
    detection_client, api_key, valid_image_b64, caplog
):
    """SECURITY: Authorization header value must never appear in any log record."""
    auth_value = f"Bearer {api_key}"

    with caplog.at_level(logging.DEBUG):
        await detection_client.post(
            "/v1/detections",
            json={"image": valid_image_b64},
            headers={"Authorization": auth_value},
        )

    for record in caplog.records:
        msg = record.getMessage()
        assert auth_value not in msg, "Authorization value leaked in log message"
        assert api_key not in msg, "API key leaked via Authorization check"


# ---------------------------------------------------------------------------
# Body size guard
# ---------------------------------------------------------------------------


async def test_oversized_body_returns_400(
    detection_client, api_key, monkeypatch
):
    """Content-Length exceeding max_request_body_bytes → 400 before body parse."""
    monkeypatch.setenv("MAX_REQUEST_BODY_BYTES", "10")
    get_settings.cache_clear()

    # Any real JSON body will be > 10 bytes; httpx sets Content-Length automatically.
    resp = await detection_client.post(
        "/v1/detections",
        json={"image": "dGVzdA=="},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    get_settings.cache_clear()

    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "too large" in resp.json()["error"]["message"].lower()


# ---------------------------------------------------------------------------
# OpenVINO fallback
# ---------------------------------------------------------------------------


def test_openvino_unavailable_logs_warning(caplog, monkeypatch, tmp_path):
    """When OpenVINO is requested but unavailable, a WARNING is logged before CPU fallback."""
    import onnxruntime as ort

    from app.inference.engine import DetectionEngine, ModelLoadError

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with caplog.at_level(logging.WARNING, logger="app.inference.engine"):
        with pytest.raises(ModelLoadError):
            # Model file is absent → ModelLoadError after the provider check.
            # The WARNING about OpenVINO unavailability must fire BEFORE session creation.
            DetectionEngine(str(tmp_path / "no_model.onnx"), provider="openvino")

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "OpenVINO" in m or "openvino" in m.lower() for m in warning_messages
    ), f"Expected OpenVINO warning. Got: {warning_messages}"


def test_cpu_provider_emits_no_openvino_warning(caplog, monkeypatch, tmp_path):
    """With ONNX_PROVIDER=cpu, no OpenVINO warning is logged even if OpenVINO is absent."""
    import onnxruntime as ort

    from app.inference.engine import DetectionEngine, ModelLoadError

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with caplog.at_level(logging.WARNING, logger="app.inference.engine"):
        with pytest.raises(ModelLoadError):
            DetectionEngine(str(tmp_path / "no_model.onnx"), provider="cpu")

    openvino_warnings = [
        r
        for r in caplog.records
        if "OpenVINO" in r.getMessage() or "openvino" in r.getMessage().lower()
    ]
    unexpected = [r.getMessage() for r in openvino_warnings]
    assert len(openvino_warnings) == 0, (
        f"Unexpected OpenVINO warning with cpu provider: {unexpected}"
    )
