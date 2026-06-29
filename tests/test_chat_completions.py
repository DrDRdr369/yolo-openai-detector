"""OpenAI compatibility facade tests.

Cover: stock openai SDK acceptance; streaming → 400; zero/multi-image → 400;
remote URL → 400; auth → 401; engine unavailable → 503; extra OpenAI fields
tolerated; cross-endpoint detection consistency.
"""

from __future__ import annotations

import json

import httpx
import openai
from httpx import ASGITransport, AsyncClient

from app.config import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _vision_payload(image_b64: str, *, stream: bool = False) -> dict:
    return {
        "model": "yolo11n",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "detect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Happy-path structure
# ---------------------------------------------------------------------------


async def test_happy_path_structure(detection_client, api_key, valid_image_b64):
    """Valid vision request returns a well-formed chat.completion envelope."""
    resp = await detection_client.post(
        "/v1/chat/completions",
        json=_vision_payload(valid_image_b64),
        headers=_auth(api_key),
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int) and body["created"] > 0
    assert isinstance(body["model"], str)
    assert len(body["choices"]) == 1

    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"

    assert body["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    content = json.loads(choice["message"]["content"])
    assert "detections" in content
    det = content["detections"][0]
    assert det["class_id"] == 0
    assert det["label"] == "person"
    assert abs(det["confidence"] - 0.91) < 1e-6
    assert det["box"] == {"x1": 100.0, "y1": 50.0, "x2": 220.0, "y2": 400.0}


async def test_model_echoed_from_request(detection_client, api_key, valid_image_b64):
    """model field in response echoes the request model."""
    resp = await detection_client.post(
        "/v1/chat/completions",
        json=_vision_payload(valid_image_b64),
        headers=_auth(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "yolo11n"


async def test_model_falls_back_to_settings(detection_client, api_key, valid_image_b64):
    """When model is omitted, settings.model_id is used."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{valid_image_b64}"},
                    }
                ],
            }
        ]
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == get_settings().model_id


# ---------------------------------------------------------------------------
# Fail-closed: streaming
# ---------------------------------------------------------------------------


async def test_streaming_returns_400(detection_client, api_key, valid_image_b64):
    """stream: true → 400 invalid_request_error."""
    resp = await detection_client.post(
        "/v1/chat/completions",
        json=_vision_payload(valid_image_b64, stream=True),
        headers=_auth(api_key),
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Fail-closed: image count
# ---------------------------------------------------------------------------


async def test_zero_images_returns_400(detection_client, api_key):
    """No image_url content parts → 400."""
    payload = {
        "model": "yolo11n",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "detect"}]}],
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_text_only_string_content_returns_400(detection_client, api_key):
    """String content with no image parts → 400."""
    payload = {
        "model": "yolo11n",
        "messages": [{"role": "user", "content": "detect objects in this image"}],
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_multiple_images_returns_400(detection_client, api_key, valid_image_b64):
    """Two image_url parts → 400."""
    data_url = f"data:image/png;base64,{valid_image_b64}"
    payload = {
        "model": "yolo11n",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Fail-closed: remote URL (SSRF)
# ---------------------------------------------------------------------------


async def test_remote_http_url_returns_400(detection_client, api_key):
    """http:// image URL → 400 (no SSRF)."""
    payload = {
        "model": "yolo11n",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
                ],
            }
        ],
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


async def test_remote_https_url_returns_400(detection_client, api_key):
    """https:// image URL → 400 (no SSRF)."""
    payload = {
        "model": "yolo11n",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ],
            }
        ],
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Fail-closed: auth
# ---------------------------------------------------------------------------


async def test_missing_auth_returns_401(detection_client):
    """No Authorization header → 401 authentication_error."""
    resp = await detection_client.post(
        "/v1/chat/completions", json={"messages": []}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


async def test_wrong_key_returns_401(detection_client, valid_image_b64):
    """Wrong bearer token → 401 authentication_error."""
    resp = await detection_client.post(
        "/v1/chat/completions",
        json=_vision_payload(valid_image_b64),
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


# ---------------------------------------------------------------------------
# Fail-closed: engine unavailable
# ---------------------------------------------------------------------------


async def test_engine_unavailable_returns_503(monkeypatch, api_key, valid_image_b64):
    """503 is returned when the detection engine is not loaded."""
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
        resp = await c.post(
            "/v1/chat/completions",
            json=_vision_payload(valid_image_b64),
            headers=_auth(api_key),
        )

    get_settings.cache_clear()
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "server_error"


# ---------------------------------------------------------------------------
# Extra OpenAI fields tolerated
# ---------------------------------------------------------------------------


async def test_extra_openai_fields_tolerated(detection_client, api_key, valid_image_b64):
    """Extra OpenAI fields (temperature, max_tokens, n, top_p) are silently ignored."""
    payload = {
        **_vision_payload(valid_image_b64),
        "temperature": 0.7,
        "max_tokens": 1024,
        "n": 1,
        "top_p": 1.0,
    }
    resp = await detection_client.post(
        "/v1/chat/completions", json=payload, headers=_auth(api_key)
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Cross-endpoint consistency
# ---------------------------------------------------------------------------


async def test_cross_endpoint_consistency(detection_client, api_key, valid_image_b64):
    """Both endpoints share the same detection service: identical images → identical detections."""
    r_native = await detection_client.post(
        "/v1/detections",
        json={"image": valid_image_b64},
        headers=_auth(api_key),
    )
    r_chat = await detection_client.post(
        "/v1/chat/completions",
        json=_vision_payload(valid_image_b64),
        headers=_auth(api_key),
    )
    assert r_native.status_code == 200
    assert r_chat.status_code == 200

    native_dets = r_native.json()["detections"]
    chat_dets = json.loads(r_chat.json()["choices"][0]["message"]["content"])["detections"]
    assert native_dets == chat_dets


# ---------------------------------------------------------------------------
# Stock OpenAI SDK acceptance test
# ---------------------------------------------------------------------------


async def test_openai_sdk_vision_request_succeeds(
    monkeypatch, api_key, fake_engine, valid_image_b64
):
    """A real openai.AsyncOpenAI client succeeds with only base_url and api_key changed."""
    monkeypatch.setenv("GATEWAY_API_KEY", api_key)
    get_settings.cache_clear()

    from app.dependencies import get_engine
    from app.main import create_app

    app_instance = create_app()
    app_instance.dependency_overrides[get_engine] = lambda: fake_engine

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_instance),
        base_url="http://test",
    ) as http_client:
        oa_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="http://test/v1",
            http_client=http_client,
        )
        response = await oa_client.chat.completions.create(
            model="yolo11n",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "detect"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{valid_image_b64}",
                            },
                        },
                    ],
                }
            ],
        )

    get_settings.cache_clear()

    assert response.object == "chat.completion"
    assert response.id.startswith("chatcmpl-")
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "stop"

    content = json.loads(response.choices[0].message.content)
    assert "detections" in content
    assert len(content["detections"]) == 1
    det = content["detections"][0]
    assert det["label"] == "person"
    assert abs(det["confidence"] - 0.91) < 1e-6
