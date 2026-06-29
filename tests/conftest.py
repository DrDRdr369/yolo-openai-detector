"""Shared pytest fixtures.

Provides an httpx ASGI test client with a fake GATEWAY_API_KEY (never a real secret).
The lru_cache on get_settings is cleared around each client fixture so settings
cannot leak between tests.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings


@pytest.fixture
def api_key() -> str:
    """An obviously-fake key — never a real secret."""
    return "test-key-not-a-real-secret"


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
