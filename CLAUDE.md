# AGENTS.md — Project Constitution

**Project:** YOLO Vision Gateway (CPU-only, OpenAI-compatible object detection)
**Audience:** The execution agent (coding agent). This file is project law. Every work order inherits these rules. When a work order and this file conflict, **stop and ask the strategic layer** — do not improvise.

---

## 1. Mission (do not redefine)

Build a self-hosted, **CPU-only** object-detection gateway that:
- exposes an **OpenAI-compatible** HTTP surface authenticated by a **single fixed API key**;
- runs **YOLO object detection** locally on **one image per request**;
- accepts the image **as base64** (the way the OpenAI vision API attaches it — a `data:` URL or raw base64 string);
- is **fully stateless** — no sessions, no tracking, no background jobs, no database;
- **fails closed** on bad auth, bad input, or unloaded model.

You (the executor) implement bounded slices of this. You do **not** change the product category, the API contract, or the non-goals without an explicit work order.

---

## 2. Stack law (use these; do not substitute silently)

- Language: **Python 3.11+**
- Web: **FastAPI** + **Uvicorn**, async handlers
- Schemas/validation: **Pydantic v2**
- Inference runtime: **ONNX Runtime (CPU)**; OpenVINO execution provider is an *optional* opt-in. **Do not add a CUDA/GPU dependency.**
- Model: **Ultralytics YOLO** detection variant, **exported to ONNX**. The training framework must not be a runtime dependency of the serving path. **No segmentation weights.**
- Image decode: **Pillow**.
- Config: env vars via **Pydantic Settings**.
- Packaging: **Docker + docker-compose**, single service.
- Tests: **pytest**, **pytest-asyncio**, **httpx** ASGI client.

If a slice genuinely needs a tool not listed here, propose it in your report and wait — do not add heavy dependencies on your own initiative.

---

## 3. Security rules (non-negotiable)

1. **Fail closed.** Invalid/missing key → `401`. Invalid/missing/oversized image → `400`. Model not loaded → `503`. Never infer on, or guess around, invalid input.
2. **Single secret.** The only secret is `GATEWAY_API_KEY`, read from the environment. Never hardcode it, never log it, never echo it in errors.
3. **Constant-time key comparison.** Use `hmac.compare_digest` (or equivalent), not `==`.
4. **Base64 input only.** Decode the attached image from base64 / `data:` URL. **Do not fetch remote URLs** (no SSRF surface, no outbound calls).
5. **No GPU/cloud/provider keys.** There is no upstream provider. Do not add provider SDKs or credentials.
6. **Input limits before work.** Enforce max decoded bytes, max resolution, allowed formats *before* decoding/inference to bound CPU/memory.
7. **No image persistence.** Process in memory, discard. Logs may record metadata (size, latency, count) but **never raw image bytes**.
8. **No real secrets in tests or fixtures.** Test keys must be obviously fake.

---

## 4. Non-goals for v1 (do not build these)

- **No tracking, no object IDs, no `session_id`, no session state.**
- **No segmentation / masks.** Detection boxes only.
- **No background jobs**, workers, queues, or schedulers (no Celery/Redis).
- No user database, no per-user keys, no quotas, no billing/accounting.
- No GPU support, no CUDA, no TensorRT.
- No remote image URL fetching; **base64 attached image only**.
- No video / frame sequences / RTSP / webcam. **One image per request.**
- No general LLM/chat behavior. The chat-completions endpoint returns detection JSON only.
- No streaming responses.
- No web UI / admin dashboard.
- No training, fine-tuning, or model-management endpoints.

If a task seems to require one of these, it is out of scope — report it, do not implement it.

---

## 5. Testing expectations (tests are the language of trust)

- Every PR adds or updates tests for the behavior it changes.
- Required coverage areas as they are implemented:
  - **Auth:** valid key passes; missing/wrong key → 401; constant-time path exercised.
  - **Input validation:** oversized image → 400; bad/unsupported format → 400; corrupt base64 → 400; missing image → 400.
  - **Detection:** a golden test image yields expected classes above threshold (deterministic).
  - **Statelessness:** identical requests yield identical results; no cross-request state.
  - **Compatibility:** a request built with the stock OpenAI client (base_url + key swapped, image attached as base64) succeeds against `/v1/chat/completions`.
  - **Fail-closed:** model-not-loaded path → 503; unsupported chat features (streaming, multi/zero image) → 400.
- **No skipped tests presented as passing.** Skips must be reported explicitly with a reason.
- Tests must be deterministic: pin the model, seed any randomness, use fixed fixtures.

---

## 6. Definition of done (per PR)

A slice is done only when:
1. Code implements exactly the work-order scope (no silent scope expansion).
2. New/updated tests exist and pass; full suite passes locally.
3. No new dependency outside stack law (or it is flagged and justified).
4. Docs touched if behavior/contract changed (`docs/api-contract.md`, README).
5. Security rules upheld (re-read Section 3).
6. Only related files are committed; clean working tree otherwise.
7. A branch + PR is opened (never commit to `main` directly; never self-merge).
8. The executor report (Section 8) is filled in.

---

## 7. PR workflow

- One PR-sized task per branch. Branch naming: `pr/<slice-short-name>`.
- The remote repository, PR, and CI are the source of truth — not the local VM.
- Do not merge your own PR. The strategic layer reviews evidence; the human authorizes merge/release.
- Keep diffs reviewable; if a slice grows large, stop and request re-slicing.

---

## 8. Required executor report (paste into every PR)

```
## What I was asked to do
<the work-order goal, restated>

## What I changed
<files + one-line reason each>

## Tests
<commands run, pass/fail counts, what each new test proves>
<explicitly list any skipped tests and why>

## Dependencies
<anything installed; flag anything outside stack law>

## Security check
<which fail-closed paths were verified; confirm no secrets/image bytes logged>

## Out of scope / not done
<anything I deliberately did not do, and why>

## What I'm least confident about
<honest uncertainty for the strategic reviewer to probe>
```

---

## 9. Anti-pilot rule

Do low-level setup (dependency installs, model export, fixtures) **yourself inside the execution VM**. Do not ask the human to run commands or install packages on your behalf. Report what you changed so the VM is rebuildable.
