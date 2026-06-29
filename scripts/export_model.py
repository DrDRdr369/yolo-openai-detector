"""Export a YOLO detection model to ONNX for CPU serving.

Run OFFLINE / at build time only. Requires the ``[export]`` optional dependencies:

    pip install -e ".[export]"

These must NEVER be imported by the serving path (src/app/).

Usage::

    python scripts/export_model.py --model yolo11n --out models/yolo11n.onnx
    # or via Make:
    make export-model MODEL_ID=yolo11n

The exported ONNX is opset 12, compatible with the pinned onnxruntime version.
Class names are embedded in the ONNX model's custom metadata (key ``"names"``)
so the serving engine can read them without re-importing ultralytics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO detection model to ONNX.")
    parser.add_argument("--model", default="yolo11n", help="Ultralytics model name or .pt path")
    parser.add_argument("--out", default="models/yolo11n.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    args = parser.parse_args()

    # Import ultralytics only here — never in src/app/
    from ultralytics import YOLO  # noqa: PLC0415

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Verify this is a detection model (no segmentation head)
    task = getattr(model, "task", None)
    if task and task != "detect":
        raise SystemExit(
            f"Model task is {task!r}, expected 'detect'. "
            "Export a detection-only model (e.g. yolo11n, yolov8n)."
        )

    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset=12) → {out_path}")
    model.export(format="onnx", imgsz=args.imgsz, opset=12, dynamic=False)

    # Ultralytics writes the file alongside the weights; move to --out if needed
    default_export = Path(args.model).with_suffix(".onnx")
    if not default_export.exists():
        # Try common locations ultralytics uses
        candidates = list(Path(".").glob(f"**/{Path(args.model).stem}.onnx"))
        if candidates:
            default_export = candidates[0]

    if default_export.exists() and default_export.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        default_export.rename(out_path)

    # Print class-name map for verification
    names: dict = model.names  # type: ignore[assignment]
    print(f"Output: {out_path}")
    print(f"Classes ({len(names)}): {json.dumps(names, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
