"""Auth and routing tests.

Covers: /healthz (unauthenticated); /v1/models with valid/missing/wrong key;
/v1/models as authenticated readiness probe (200 when loaded, 503 when not);
app refuses to start without GATEWAY_API_KEY; constant-time key comparison.
"""

from __future__ import annotations

import hmac
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.config import Settings, get_settings


async def test_healthz_no_auth(client):
    """/healthz returns 200 without any Authorization header."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_valid_key_allows_models(detection_client, api_key):
    """/v1/models with correct bearer key and loaded engine returns 200 + list shape."""
    response = await detection_client.get(
        "/v1/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert len(body["data"]) == 1
    model = body["data"][0]
    assert model["object"] == "model"
    assert model["owned_by"] == "local"
    assert "id" in model


async def test_models_returns_503_when_engine_unavailable(monkeypatch, api_key):
    """/v1/models returns 503 when the detection engine is not loaded (readiness semantics)."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from app.main import create_app

    app_instance = create_app()
    # No dependency override → lifespan sets engine=None (model file absent in test env)
    async with AsyncClient(
        transport=ASGITransport(app=app_instance), base_url="http://test"
    ) as c:
        resp = await c.get("/v1/models", headers={"Authorization": f"Bearer {api_key}"})

    get_settings.cache_clear()
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "server_error"


async def test_missing_auth_header_returns_401(client):
    """/v1/models with no Authorization header returns 401 with OpenAI error body."""
    response = await client.get("/v1/models")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["type"] == "authentication_error"


async def test_wrong_key_returns_401(client):
    """/v1/models with an incorrect bearer token returns 401 with OpenAI error body."""
    response = await client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["type"] == "authentication_error"


async def test_malformed_auth_header_returns_401(client):
    """Authorization header without 'Bearer ' prefix returns 401."""
    response = await client.get("/v1/models", headers={"Authorization": "Token something"})
    assert response.status_code == 401


def test_settings_fail_without_key(monkeypatch):
    """App must refuse to start without GATEWAY_API_KEY — raises ValidationError specifically."""
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        # Disable env_file so no .env file on disk can satisfy the requirement
        Settings(_env_file=None)  # type: ignore[call-arg]
    get_settings.cache_clear()


async def test_key_check_uses_constant_time_compare(detection_client, api_key):
    """The auth path calls hmac.compare_digest, not plain ==."""
    with patch("app.auth.hmac.compare_digest", wraps=hmac.compare_digest) as mock_cd:
        response = await detection_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {api_key}"}
        )
        assert response.status_code == 200
        assert mock_cd.called, "hmac.compare_digest was not called — not constant-time"
