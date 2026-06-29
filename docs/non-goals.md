# Non-Goals (v1)

These are hard boundaries. They exist to keep the system small, auditable, and fast on
CPU. The execution agent **must not** implement any of these without an explicit,
new strategic decision recorded in `AGENTS.md` and `docs/work-orders.md`. If a task seems
to require one of these, it is out of scope — report it, do not build it.

## Capability boundaries
- **No object tracking.** No object IDs, no `session_id`, no cross-request association.
- **No sessions / no per-client state.** The service is fully stateless.
- **No segmentation / instance masks.** Bounding-box detection only.
- **No classification-only or pose/keypoint modes.** Detection only.
- **No video, frame sequences, RTSP, or webcam ingestion.** Exactly one image per request.

## Input boundaries
- **Base64 attached image only** (raw base64 or `data:` URL), as the OpenAI vision API
  attaches it.
- **No server-side remote URL fetching.** A remote `http(s)` image URL is rejected with
  `400`. This removes SSRF risk and any outbound network dependency.

## Runtime / infra boundaries
- **No GPU.** No CUDA, no TensorRT, no GPU-only dependencies.
- **No background jobs.** No Celery, no task queues, no schedulers, no async workers
  beyond the request/response cycle.
- **No database.** No persistence layer of any kind.
- **No Redis or other external state store.**

## Product boundaries
- **No multi-tenancy.** Single fixed key only — no per-user keys, quotas, billing, or
  usage accounting.
- **No general chat / LLM behavior.** The `/v1/chat/completions` endpoint is a thin
  adapter that returns detection JSON only.
- **No streaming responses.**
- **No web UI / admin dashboard.**
- **No training, fine-tuning, or model-management endpoints.**

## Why these are non-goals
Each excluded feature adds state, attack surface, CPU cost, or operational burden that
conflicts with the core value: a tiny, stateless, fail-closed, OpenAI-compatible detector
that runs anywhere without a GPU. Several of these (tracking, video, sessions) may be
legitimate *future* products, but they are deliberately out of v1.
