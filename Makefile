.PHONY: install run test lint export-model docker-build docker-up test-integration

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload --app-dir src --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

export-model:
	pip install -e ".[export]"
	python scripts/export_model.py --model $(MODEL_ID) --out models/$(MODEL_ID).onnx

docker-build:
	docker compose build

docker-up:
	docker compose up -d

MODEL_ID ?= yolo11n

test-integration:
	$(MAKE) export-model MODEL_ID=$(MODEL_ID)
	MODEL_PATH=models/$(MODEL_ID).onnx pytest -m integration -v
