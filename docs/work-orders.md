# Work-Order Sequence (First PR Slices)

**Status:** Discovery output (v0.2). Each slice is PR-sized: one branch, one reviewable diff, tests included. The strategic layer issues these one at a time to the execution agent and reviews evidence before authorizing the next. Do not batch them.

**v0.2 scope:** stateless single-shot detection, base64 image only. The tracking slice from the earlier draft is removed.

Order matters: each builds on the verified state of the previous. **Current verified state: empty repository.**

---

## PR-0 — Project skeleton & constitution intake
**Goal:** A runnable FastAPI app with config loading and auth. No inference yet.
**Build:** repo layout, `pyproject`/requirements pinned to stack law, FastAPI app, `GATEWAY_API_KEY` config (fail to start if unset), Bearer auth dependency (constant-time), `GET /healthz`, `GET /v1/models` returning a static stub, Dockerfile + docker-compose. Commit `AGENTS.md`, `ARCHITECTURE.md`, `docs/api-contract.md`.
**Acceptance:** app starts only with key set; `/healthz` 200; `/v1/models` 200 with key, 401 without.
**Tests:** auth pass/fail (401), constant-time path, app-fails-without-key.
**Non-goals:** no model, no detection.

---

## PR-1 — CPU inference core (stateless detection)
**Goal:** Load an ONNX YOLO detection model and run detection on a decoded image in-process. Library-level, no HTTP yet.
**Build:** model export script (Ultralytics → ONNX, committed as a documented step), ONNX Runtime CPU session wrapper, pre/post-processing (letterbox, NMS), class-label map. Base64/`data:` URL decode (Pillow) with input limits enforced before decode.
**Acceptance:** given a golden image (as base64), returns expected classes above threshold, deterministically.
**Tests:** golden-image detection; oversized/corrupt/bad-base64 image rejected; NMS correctness on a synthetic case.
**Non-goals:** no HTTP endpoint yet, no chat facade.

---

## PR-2 — Native endpoint `POST /v1/detections`
**Goal:** Expose stateless detection over HTTP with the native typed schema.
**Build:** Pydantic request/response models per `api-contract.md` §3, wire to inference core, full error model (§5), reject remote URLs, input-limit enforcement before work.
**Acceptance:** valid base64 request → typed detections; bad input → 400; remote URL → 400; no key → 401; model not loaded → 503.
**Tests:** happy path; each fail-closed branch; `classes` filter; threshold overrides; statelessness (identical request → identical response).
**Non-goals:** chat facade.

---

## PR-3 — Compatibility facade `POST /v1/chat/completions`
**Goal:** Stock OpenAI SDK works unchanged; detections returned as JSON in the assistant message.
**Build:** parse vision message (extract single base64 `image_url`), build chat-completion envelope (§4), zero-filled `usage`. Unsupported features (streaming, multi/zero image, remote URL) → fail closed.
**Acceptance:** a request built with the real OpenAI Python client (base_url+key swapped, image attached as base64) succeeds and returns parseable detection JSON.
**Tests:** stock-SDK integration test; streaming/embeddings → 400; multi-image → 400; remote URL → 400.
**Non-goals:** streaming support.

---

## PR-4 — Hardening, docs, release honesty
**Goal:** Production-honest RC-beta, not "production certified."
**Build:** README with quickstart + curl + OpenAI-SDK examples; reference-hardware latency table per model size; finalize input limits; structured logging (metadata only, no image bytes, no key); `docs/limitations.md` (CPU latency, detection-only, facade scope). Optional OpenVINO provider flag.
**Acceptance:** fresh `docker compose up` works from README alone; logs leak nothing sensitive.
**Tests:** end-to-end smoke via compose; log-scrub assertion.
**Non-goals:** everything in the v1 non-goal list below.

---

## Explicit non-goals for v1 (guardrails — do not build)

- No tracking, object IDs, sessions, or `session_id`.
- No segmentation / masks. Detection boxes only.
- No background jobs, workers, queues, or schedulers.
- No user DB, per-user keys, quotas, billing, or usage accounting.
- No GPU / CUDA / TensorRT.
- No remote image URL fetching — base64 attached image only.
- No video / frame sequences / RTSP / webcam. One image per request.
- No general chat/LLM behavior on the facade.
- No streaming responses.
- No web UI / admin dashboard.
- No training / fine-tuning / model-management endpoints.

---

## Strategic review checkpoints (per PR, before merge)

For each returned PR, the strategic layer asks: Did it match the work order exactly? Are the tests meaningful and unskipped? Is it still fail-closed? Any secret/image-byte leakage in logs? Any new dependency outside stack law? Anything documented but not implemented? Only after clean answers does the human authorize merge and the next slice is issued.
