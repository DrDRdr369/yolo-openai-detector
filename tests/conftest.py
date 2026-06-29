"""Shared pytest fixtures.

Provides an httpx ASGI test client with a fake GATEWAY_API_KEY (never a real secret).
The lru_cache on get_settings is cleared around each client fixture so settings
cannot leak between tests.
"""

from __future__ import annotations

import base64
import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image as PILImage

from app.config import get_settings

# ---------------------------------------------------------------------------
# Key / image helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def api_key() -> str:
    """An obviously-fake key — never a real secret."""
    return "test-key-not-a-real-secret"


@pytest.fixture
def valid_image_b64() -> str:
    """Small valid PNG image as raw base64 — reusable across test modules."""
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(70, 130, 180)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Unauthenticated / basic client (PR-0 tests)
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(monkeypatch, api_key: str):
    """httpx.AsyncClient wired to the ASGI app with test env configured."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from app.main import create_app

    app_instance = create_app()
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as c:
        yield c

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fake engine (PR-2 / PR-3 tests — no real model needed)
# ---------------------------------------------------------------------------


class FakeDetectionEngine:
    """A deterministic stand-in for DetectionEngine in endpoint tests.

    Records every call to ``infer()`` in ``self.calls`` so tests can assert
    that thresholds and class filters were forwarded correctly.
    """

    DEFAULT_DETECTIONS: list[dict] = [
        {
            "class_id": 0,
            "label": "person",
            "confidence": 0.91,
            "box": {"x1": 100.0, "y1": 50.0, "x2": 220.0, "y2": 400.0},
        }
    ]

    def __init__(self, detections: list[dict] | None = None) -> None:
        self._detections = detections if detections is not None else list(self.DEFAULT_DETECTIONS)
        self.calls: list[dict] = []

    def infer(
        self,
        image,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int] | None = None,
    ) -> list[dict]:
        self.calls.append(
            {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold, "classes": classes}
        )
        return self._detections


@pytest.fixture
def fake_engine() -> FakeDetectionEngine:
    """Default fake engine returning one canned detection."""
    return FakeDetectionEngine()


@pytest.fixture
async def detection_client(monkeypatch, api_key: str, fake_engine: FakeDetectionEngine):
    """Client with a fake DetectionEngine injected — no real model needed."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from app.dependencies import get_engine
    from app.main import create_app

    app_instance = create_app()
    app_instance.dependency_overrides[get_engine] = lambda: fake_engine

    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as c:
        yield c

    get_settings.cache_clear()
