# scripts/

Offline / build-time utilities. Not part of the serving runtime.

- `export_model.py` — export a YOLO **detection** model to ONNX into `models/`.
  Requires `pip install -e ".[export]"`. The serving path never imports ultralytics/torch.
