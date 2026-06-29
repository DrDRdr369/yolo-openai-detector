# Known Limitations

Release honesty is a project rule (see `AGENTS.md`). This file states what the system does
**not** guarantee, so no document or report overclaims maturity.

## Release status

**RC-beta (implemented scope).**
PRs 0–4 implement the full v1 scope: bearer auth, CPU ONNX inference, native
`POST /v1/detections`, OpenAI `POST /v1/chat/completions` facade, structured
no-leak logging, and input hardening. The service is **not** production-certified,
security-audited, or penetration-tested. Do not deploy to an internet-exposed
endpoint without an independent security review.

## Functional limitations (by design)

- **Detection only.** No tracking, no segmentation, no object IDs. See `docs/non-goals.md`.
- **One base64 image per request.** No batching, no video, no remote URL fetching.
- **Chat-completions facade is a thin adapter,** not a general LLM. Unsupported chat
  features (streaming, multiple/zero images, text-only prompts expecting text answers)
  fail closed with `400`.
- **Single shared key.** Anyone with `GATEWAY_API_KEY` has full access. No per-user
  quotas or per-key audit trail. Rotation is a single-value swap (see `docs/runbook.md`).

## Performance limitations

- **CPU latency.** Per-image inference time scales with model size and image resolution.
  See the latency table in `docs/ARCHITECTURE.md §7` for measured or reference values.
- **No request batching** in v1; concurrency is bounded by worker count and per-image CPU time.
- **No horizontal GPU scaling.** CPU-only by design; add more identical CPU replicas for throughput.

## Input limits

Enforced before decode on both endpoints (fail closed):

| Limit | Default | Config var |
|---|---|---|
| Max decoded image bytes | 10 MB | `MAX_IMAGE_BYTES` |
| Max image pixels (w × h) | 16 777 216 (4096²) | `MAX_IMAGE_PIXELS` |
| Max JSON request body | 25 MB | `MAX_REQUEST_BODY_BYTES` |

The JSON body limit is checked via `Content-Length` header (defense-in-depth);
chunked transfer without `Content-Length` bypasses this check but is still bounded
by the image decode limits above.

## Operational limitations

- **No image persistence.** Images are processed in memory and discarded immediately.
  Logs record only metadata (latency, count, dimensions) — never image bytes or the key.
- **No audit log of who called what.** The single-key model means all requests are
  indistinguishable by caller.
