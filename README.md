# YOLO Vision Gateway

A **self-hosted, CPU-only object-detection gateway** that exposes an **OpenAI-compatible**
HTTP API, authenticated with a **single fixed API key**. It runs a YOLO detection model
locally on **one base64-attached image per request** and returns bounding boxes, class
labels, and confidence scores. It is **fully stateless** — no tracking, no segmentation,
no sessions, no background jobs, no database.

> Drop-in shape: point the stock OpenAI SDK at this server's `base_url`, attach an image,
> and get detections back as JSON.

---

## Why this exists

Many users need practical object detection but only have **GPU-less** hardware, and they
already speak the OpenAI API. This gateway lets existing OpenAI clients get local CPU
detections with a one-line `base_url` + key change, while keeping a single static key and a
tiny, auditable surface.

This repository was initialized from a **strategic discovery** process. The governing
documents are not decoration — they are project law:

| Document | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Project constitution.** Stack law, security rules, non-goals, definition of done, executor report format. |
| [`CLAUDE.md`](CLAUDE.md) | Verbatim mirror of `AGENTS.md` for Claude Code. `AGENTS.md` is canonical — keep them in sync. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Product category, component design, stack rationale, deployment, known limits. |
| [`docs/api-contract.md`](docs/api-contract.md) | OpenAI compatibility matrix + request/response schemas + error model. |
| [`docs/non-goals.md`](docs/non-goals.md) | The hard boundaries. What this project will **not** do. |
| [`docs/work-orders.md`](docs/work-orders.md) | PR-sized implementation sequence for the execution agent. |
| [`docs/limitations.md`](docs/limitations.md) | Honest, release-facing limitations. |
| [`docs/runbook.md`](docs/runbook.md) | Operator runbook (key rotation, deploy, troubleshooting). |

---

## Project status

**RC-beta (implemented scope).** PRs 0–5 implement the full v1 feature set: bearer auth,
CPU ONNX inference, `POST /v1/detections`, `POST /v1/chat/completions` (stock OpenAI SDK
compatible), structured no-leak logging, input hardening, and CI with real-model integration
tests. The service passes its own test suite and is suitable for internal evaluation.

This is **not** production-certified, security-audited, or penetration-tested.
Do not expose to an untrusted network without an independent security review.
See [`docs/limitations.md`](docs/limitations.md) for the full honest limitation list.

---

## Architecture at a glance

```
OpenAI SDK / HTTP client
        │  Authorization: Bearer <GATEWAY_API_KEY>
        ▼
FastAPI (Uvicorn, async)
  ├── auth dependency (constant-time, fail closed)
  ├── GET  /v1/models
  ├── POST /v1/detections          ← native typed endpoint
  └── POST /v1/chat/completions     ← OpenAI compatibility facade
        │
        ▼
base64 decode (Pillow) + input limits
        │
        ▼
YOLO detection via ONNX Runtime (CPU)
        │
        ▼
detections: [{class_id, label, confidence, box}]
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# edit .env and set GATEWAY_API_KEY

# 2. Run with Docker
docker compose up --build

# 3. Call it with the stock OpenAI SDK
python - <<'PY'
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_FIXED_KEY")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()

resp = client.chat.completions.create(
    model="yolo11n",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "detect"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ],
    }],
)
print(resp.choices[0].message.content)  # JSON string of detections
PY
```

Native endpoint via `curl`:

```bash
curl -s http://localhost:8000/v1/detections \
  -H "Authorization: Bearer YOUR_FIXED_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"yolo11n","image":"data:image/jpeg;base64,..."}'
```

---

## Local development

```bash
make install          # install runtime + dev dependencies
make export-model     # export a YOLO detection model to ONNX into models/
make run              # run the dev server (uvicorn, reload)
make test             # run pytest (hermetic suite — no model required)
make test-integration # export model if absent, then run real-model integration tests
make lint             # run ruff
```

CI runs two jobs on every push and PR: fast hermetic tests (no model), then real-model
integration tests with the ONNX model cached across runs. See [`docs/runbook.md`](docs/runbook.md)
for details.

See the [`Makefile`](Makefile) for the underlying commands.

---

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) and the
config table in [`docs/api-contract.md`](docs/api-contract.md). The only secret is
`GATEWAY_API_KEY`; the app **fails to start** if it is unset.

---

## License

No license file is included yet — **add a `LICENSE` before any public release.** Until then,
treat the contents as all-rights-reserved by the author/owner.
