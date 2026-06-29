"""Export a YOLO detection model to ONNX for CPU serving.

Implements: PR-1 (procedure committed at init; weights are produced offline).

Run OFFLINE / at build time only. Requires the `export` optional dependencies
(`pip install -e ".[export]"`) which include ultralytics + onnx. These must NEVER be
imported by the serving path.

Usage:
    python scripts/export_model.py --model yolo11n --out models/yolo11n.onnx
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO detection model to ONNX.")
    parser.add_argument("--model", default="yolo11n", help="Ultralytics model name or .pt path")
    parser.add_argument("--out", default="models/yolo11n.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.parse_args()

    # PR-1: load the Ultralytics detection model and export to ONNX (opset suitable for
    # onnxruntime), writing to --out. Keep it a DETECTION model (no segmentation head).
    raise NotImplementedError("PR-1: implement Ultralytics -> ONNX export.")


if __name__ == "__main__":
    main()
