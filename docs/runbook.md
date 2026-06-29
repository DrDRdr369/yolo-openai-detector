# Operator Runbook

Operational procedures for running YOLO Vision Gateway. Keep this current as the system is
implemented.

## Deploy
1. Provision a Linux host with Docker (no GPU required).
2. `cp .env.example .env` and set `GATEWAY_API_KEY` to a strong random value.
3. Ensure a detection model exists at `MODEL_PATH` (see "Model export" below) or is baked
   into the image.
4. `docker compose up -d --build`.
5. Verify: `curl -f http://localhost:8000/healthz` and
   `curl -H "Authorization: Bearer $GATEWAY_API_KEY" http://localhost:8000/v1/models`.

## Model export
The serving path uses an ONNX model; the training framework is **not** a runtime
dependency. Export once (in a dev/build environment) with:

```bash
make export-model            # wraps scripts/export_model.py
# or
python scripts/export_model.py --model yolo11n --out models/yolo11n.onnx
```

Commit the export *procedure*, not necessarily the weights. Large weights may be mounted
as a volume or fetched at build time.

## Key rotation
1. Generate a new key.
2. Update `GATEWAY_API_KEY` in the environment / secret store.
3. Restart the service (`docker compose up -d`).
4. Distribute the new key to clients. There is no key list — rotation is a single value
   swap, so coordinate the cutover.

## Leaked key response
1. Rotate immediately (above).
2. Because there is no per-key audit trail, treat all prior traffic as potentially
   attributable to the leaked key.
3. If network-level logs exist (reverse proxy), review for anomalous source IPs.

## Troubleshooting
| Symptom | Likely cause | Action |
|---|---|---|
| App exits on startup | `GATEWAY_API_KEY` unset | Set the env var; the app fails closed by design. |
| `503` on requests | Model not loaded | Check `MODEL_PATH` and that the ONNX file exists/readable. |
| `401` on every call | Wrong/missing bearer token | Verify the `Authorization: Bearer <key>` header. |
| `400` "base64 image required" | Client sent a remote URL | Attach the image as base64 / `data:` URL. |
| High latency | Large model or image on CPU | Use a smaller model (`yolo11n`), cap `MAX_IMAGE_PIXELS`, scale replicas. |

## Scaling
The service is stateless, so scale horizontally: run multiple identical replicas behind a
load balancer. No sticky routing or shared state is required.

## Logging
Logs contain request metadata only (size, latency, detection count). They must **never**
contain raw image bytes or the API key. Verify this after any logging change.
