# API Contract & OpenAI Compatibility Matrix

**Status:** Discovery output (v0.2). This is the contract the execution agent implements. Changes to this file are strategic decisions, not executor improvisation.

**v0.2 scope:** stateless single-shot object detection. No sessions, no tracking, no segmentation, no background jobs. Image is supplied **as base64 only**.

Base URL convention: clients set `base_url = http://<host>:8000/v1` and `api_key = <GATEWAY_API_KEY>`.

Auth: `Authorization: Bearer <GATEWAY_API_KEY>` on every request. Missing/invalid → `401`.

---

## 1. Compatibility matrix

| OpenAI concept | Supported? | Behavior in this gateway |
|---|---|---|
| `Authorization: Bearer <key>` | ✅ | Single fixed key, constant-time compare, fail closed. |
| `base_url` override | ✅ | Point the stock SDK at `/v1`. |
| `GET /v1/models` | ✅ | Lists the loaded detection model in OpenAI list shape. |
| `POST /v1/chat/completions` (vision, base64 image) | ✅ (adapted) | Accepts one base64 image in the message; returns detections as JSON in the assistant message. Non-vision chat features unsupported → fail closed. |
| Image via base64 / `data:` URL | ✅ | The only accepted image form. |
| Image via remote `http(s)` URL | ❌ | Rejected — no server-side fetch (SSRF-free). |
| Streaming (`stream: true`) | ❌ | `400` with a clear message. |
| `POST /v1/embeddings`, `/v1/audio`, etc. | ❌ | `404` / `400` — out of scope. |
| Native typed endpoint `POST /v1/detections` | ✅ (gateway extension) | First-class detection schema for power users. |
| Tracking / `session_id` / segmentation masks | ❌ | Out of scope by design. |

---

## 2. `GET /v1/models`

Returns the loaded model so SDK readiness checks pass.

```json
{
  "object": "list",
  "data": [
    { "id": "yolo11n", "object": "model", "owned_by": "local", "created": 0 }
  ]
}
```

---

## 3. Native endpoint — `POST /v1/detections`

The clean, typed interface. Recommended for clients that can write a small amount of custom code.

### Request
```json
{
  "model": "yolo11n",
  "image": "<base64 string OR data: URL>",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45,
  "classes": [0, 2]
}
```
- `image` — required. **One base64 image only** (`data:image/...;base64,...` or raw base64). No remote URLs.
- `conf_threshold`, `iou_threshold` — optional, defaults from config.
- `classes` — optional class-ID allowlist filter.

### Response
```json
{
  "model": "yolo11n",
  "image": { "width": 1280, "height": 720 },
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.91,
      "box": { "x1": 100.0, "y1": 50.0, "x2": 220.0, "y2": 400.0 }
    }
  ],
  "timing_ms": { "decode": 4.1, "inference": 88.3 }
}
```
- `box` is absolute pixel coordinates, top-left origin.
- Stateless: the same image always yields the same response.

---

## 4. Compatibility facade — `POST /v1/chat/completions`

Lets the **stock OpenAI SDK** work unchanged. The image rides in the vision message format as base64; detections come back as JSON text in the assistant message.

### Request (standard OpenAI vision shape)
```json
{
  "model": "yolo11n",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "detect" },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ]
}
```
- The image is taken from the first `image_url` content part and **must be a base64 / `data:` URL** (a remote `http(s)` URL → `400`).
- Exactly one image is accepted; multiple or zero images → `400`.

### Response (standard chat-completion envelope)
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1751234567,
  "model": "yolo11n",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "{\"detections\":[{\"label\":\"person\",\"confidence\":0.91,\"box\":{...}}]}"
      }
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```
- `created` is a real Unix timestamp (seconds since epoch), not a static placeholder.
- `message.content` is a JSON **string** containing the same detection payload as the native endpoint (so clients can `json.loads` it).
- `usage` is zero-filled (no token billing concept here) but present for SDK compatibility.

---

## 5. Error model (fail closed)

| Condition | HTTP | Body shape (OpenAI-style error) |
|---|---|---|
| Missing/invalid key | `401` | `{"error": {"type": "authentication_error", "message": "..."}}` |
| Missing/oversized/corrupt image, bad base64, bad params | `400` | `{"error": {"type": "invalid_request_error", "message": "..."}}` |
| Remote image URL supplied | `400` | `{"error": {"type": "invalid_request_error", "message": "base64 image required"}}` |
| Unsupported feature (streaming, embeddings, multi/zero image) | `400`/`404` | `{"error": {"type": "invalid_request_error", "message": "..."}}` |
| Model not loaded | `503` | `{"error": {"type": "server_error", "message": "..."}}` |

Errors never include the API key or raw image bytes.

---

## 6. Config surface (env vars)

| Var | Meaning | Default |
|---|---|---|
| `GATEWAY_API_KEY` | The single fixed key (required) | — (fail to start if unset) |
| `MODEL_PATH` | Path to exported ONNX detection model | bundled model |
| `MODEL_ID` | Name reported by `/v1/models` | `yolo11n` |
| `CONF_THRESHOLD` | Default confidence threshold | `0.25` |
| `IOU_THRESHOLD` | Default NMS IoU threshold | `0.45` |
| `MAX_IMAGE_BYTES` | Reject larger decoded images | e.g. `10_000_000` |
| `MAX_IMAGE_PIXELS` | Reject larger resolutions | e.g. `4096*4096` |
| `ONNX_PROVIDER` | `cpu` or `openvino` | `cpu` |
