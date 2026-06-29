# Known Limitations (release-facing, honest)

Release honesty is a project rule (see `AGENTS.md`). This file states what the system does
**not** guarantee, so no document or report overclaims maturity.

## Current state
- **Scaffold / pre-implementation.** Source modules are stubs raising `NotImplementedError`.
  The service is not yet functional. Implementation follows `docs/work-orders.md`.

## Functional limitations (by design)
- **Detection only.** No tracking, no segmentation, no IDs. See `docs/non-goals.md`.
- **One base64 image per request.** No batching, no video, no remote URLs.
- **Chat-completions facade is a thin adapter,** not a general LLM. Unsupported chat
  features (streaming, multiple/zero images, non-image prompts expecting text answers)
  fail closed with `400`.

## Performance limitations
- **CPU latency.** Per-image latency scales with model size and image resolution. On
  GPU-less hardware, large images or large models may be slow. Operators should document
  measured latency per model size on their reference hardware before promising throughput.
- **No request batching** in v1, so concurrency is bounded by worker count and per-image
  CPU time.

## Operational limitations
- **No persistence.** Nothing is stored; there is no audit log of images (by design —
  images are processed in memory and discarded).
- **Single secret.** Access control is a single shared key. Anyone with the key has full
  access. Rotate via `docs/runbook.md`.

## Release language
Until the work-order slices are complete and verified, this project may be described as
**"in development / scaffold"** — never as production-ready. After implementation and the
hardening slice, it may be described as **RC-beta for the implemented scope**, not as
production-certified, security-audited, or penetration-tested.
