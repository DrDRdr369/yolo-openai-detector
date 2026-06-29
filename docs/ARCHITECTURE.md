# YOLO Vision Gateway — Architecture Note

**Status:** Discovery output (v0.2, strategic draft)
**Role of this document:** Durable record of the strategic discovery decisions. Written *before* the execution agent implements code, so the executor builds inside constraints rather than improvising the product. This is a strategic proposal — the human lead (domain authority) may reject or redirect any decision.

**v0.2 scope change:** tracking, sessions, and segmentation removed. The service is **stateless single-shot object detection** only.

---

## 1. Product category

This is **not** "wrap YOLO in a web server." It is a:

> **Self-hosted, CPU-only object-detection gateway that emulates the OpenAI API surface, authenticates with a single fixed key, runs YOLO detection on one attached image per request, and fails closed on bad auth or bad input.**

The reference pattern is the SLAIF OpenAI-compatible gateway: compatibility-first, key-gated, fail-closed. The difference is that the "upstream provider" is replaced by a **local CPU model**, and all multi-tenant machinery (quotas, accounting, sessions) is intentionally dropped.

### What this buys the user
- Existing OpenAI SDK clients point at a new `base_url` + key and get detections back — minimal client code change.
- Runs on commodity hardware with **no GPU**.
- One static key. **No database, no background workers, no session state** — fully stateless.

---

## 2. Core behavior (deliberately minimal)

- **One image per request**, supplied **as base64** (the way the OpenAI vision API attaches images — a `data:` URL or raw base64 string).
- Run YOLO **object detection** → bounding boxes, class labels, confidence scores.
- Return results synchronously. **No tracking, no IDs, no segmentation masks, no async jobs.**
- Every request is **fully independent**; the process holds no per-client state.

What this removes versus the earlier draft: no `session_id`, no ByteTrack, no in-memory tracker store, no TTL eviction, no remote image fetching, no video/streams.

---

## 3. Component overview

```
                 ┌──────────────────────────────────────────────┐
 OpenAI SDK ───► │  FastAPI app (Uvicorn, async)                 │
 / HTTP client   │                                              │
                 │  ┌─────────────┐   ┌───────────────────────┐ │
                 │  │ Auth dep    │   │ Routers               │ │
                 │  │ (Bearer,    │──►│  /v1/models           │ │
                 │  │ fail-closed)│   │  /v1/chat/completions │ │  ← compat facade
                 │  └─────────────┘   │  /v1/detections       │ │  ← native typed
                 │                    └──────────┬────────────┘ │
                 │                               ▼              │
                 │   ┌──────────────┐   ┌───────────────────┐  │
                 │   │ base64 decode│──►│ Inference service │  │
                 │   │ (Pillow)     │   │  YOLO via ONNX    │  │
                 │   │ + limits     │   │  Runtime (CPU)    │  │
                 │   └──────────────┘   └─────────┬─────────┘  │
                 │                                ▼            │
                 │                     detections (boxes,      │
                 │                     labels, scores)         │
                 └──────────────────────────────────────────────┘
```

Request flow: authenticate (fail closed) → decode + validate the single base64 image → run YOLO detection → format response (native schema or chat-completions envelope). No shared state touched.

---

## 4. Stack and rationale

All choices favor **boring, CPU-friendly, well-supported** components.

| Concern | Choice | Why |
|---|---|---|
| Web framework | **FastAPI + Uvicorn** (async) | Async I/O, automatic OpenAPI/Pydantic validation, matches the OpenAI-gateway reference pattern. |
| Request/response schemas | **Pydantic v2** | Typed contract for both endpoints; rejects malformed input early (fail closed). |
| Detection model | **YOLO (Ultralytics), small variant** — e.g. `yolo11n`/`yolo11s` or `yolov8n`/`v8s` | Smallest models that keep acceptable CPU latency. Configurable. Detection weights only — no segmentation head. |
| Inference runtime | **ONNX Runtime (CPU)**, with **OpenVINO execution provider** optional on Intel CPUs | Export YOLO → ONNX once; ONNX Runtime is markedly faster on CPU than eager PyTorch and ships a small dependency footprint. |
| Image decode | **Pillow** | Decode the base64/`data:` URL image to a numpy array. |
| Auth | **Single static bearer key** from env (`GATEWAY_API_KEY`) | Matches "fixed api key". Constant-time compare, fail closed. |
| Config | **Pydantic Settings / env vars** | 12-factor; key, model path, thresholds, input limits configurable. |
| Packaging | **Docker + docker-compose**, single service | Reproducible deploy on any GPU-less box. |
| Tests | **pytest, pytest-asyncio, httpx ASGI client** | Async route tests, deterministic golden-image fixtures. |

### Why ONNX Runtime over raw PyTorch
On a GPU-less machine, throughput and cold-start memory dominate. ONNX Runtime with the CPU execution provider (and optional OpenVINO/oneDNN) gives lower latency and a smaller install than the full PyTorch stack, which is wasted weight without a GPU. The YOLO model is exported to ONNX at setup time; the runtime never needs the training framework.

---

## 5. Deployment sketch

- Single container image: app + ONNX Runtime + the exported detection model (baked in or mounted as a volume).
- `docker-compose.yml`: one `gateway` service, exposing `:8000`, env-configured key, model path, thresholds, and input limits.
- **No database, no Redis, no background worker, no scheduler.** Fully stateless.
- **Liveness probe (unauthenticated):** `GET /healthz` — always 200 while the process is alive; no key required.
- **Readiness probe (authenticated):** `GET /v1/models` with `Authorization: Bearer <key>` — returns 200 + model list when the detection model is loaded; returns 503 when the model failed to load; returns 401 for a missing/wrong key. Wire your load-balancer or orchestrator accordingly.
- Because there is no shared state, the service **scales horizontally trivially** — run as many identical workers/replicas behind a load balancer as needed.

---

## 6. Trust boundary and security posture

- The **only** secret is `GATEWAY_API_KEY`. No upstream provider keys exist (the model is local).
- **Fail closed:** missing/invalid key → 401; missing/invalid/oversized image → 400; model not loaded → 503. Never return a "best guess" on bad input.
- **Base64 input only** — no server-side URL fetching, which removes SSRF risk and any need for outbound network calls.
- Input limits: max decoded bytes, max resolution, allowed image formats — enforced before decode to bound CPU and memory.
- No image persistence. Images are processed in memory and discarded. Logs record metadata (size, latency, detection count), never raw image bytes or the key.
- Per the runtime doctrine, the execution agent builds and tests inside a disposable VM; durable truth is the repository.

---

## 7. Known limitations and reference latency

1. **CPU latency.** Detection latency per image depends on model size and image resolution.
2. **Chat-completions facade is a thin adapter,** not a general LLM. It only returns detection JSON for one attached image; unsupported chat features fail closed with a clear error.
3. **Detection only.** No tracking, no segmentation, no classification-beyond-detection. By design.
4. **No quotas / no per-user accounting.** Single fixed key by design.

### Reference hardware latency table

Measured end-to-end per-image latency (HTTP round-trip, ASGI transport, CPU only).
Values marked **TODO** must be measured on the operator's reference hardware before
publishing performance claims. Do not substitute fabricated numbers.

| Model | Input size | Provider | CPU / hardware | p50 latency | p95 latency |
|---|---|---|---|---|---|
| yolo11n | 640 × 640 | CPU | TODO: measure | TODO | TODO |
| yolo11s | 640 × 640 | CPU | TODO: measure | TODO | TODO |
| yolo11n | 640 × 640 | OpenVINO | TODO: measure (Intel only) | TODO | TODO |

**How to measure:** export the model (`make export-model`), run the server, then use the
test suite's ASGI client in a tight loop (`timeit`) or a dedicated benchmark script against
a fixed test image. Record CPU model, core count, and RAM. Add the results to this table
before any public performance claim.
