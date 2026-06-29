"""Application configuration (env-driven).

All configuration comes from environment variables (12-factor). The only secret is
GATEWAY_API_KEY; the application MUST fail to start if it is unset (fail closed).

See docs/api-contract.md section 6 for the full config surface.
"""

from __future__ import annotations

import functools

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # REQUIRED secret — no default. Absence aborts startup.
    gateway_api_key: str

    # Model
    model_path: str = "models/yolo11n.onnx"
    model_id: str = "yolo11n"

    # Detection defaults
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45

    # Input limits (enforced BEFORE decode)
    max_image_bytes: int = 10_000_000
    max_image_pixels: int = 4096 * 4096

    # ONNX Runtime execution provider: "cpu" | "openvino"
    onnx_provider: str = "cpu"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"

    # Max JSON request body (defense-in-depth before image decode limits)
    max_request_body_bytes: int = 25_000_000

    @field_validator("gateway_api_key")
    @classmethod
    def _key_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "GATEWAY_API_KEY must be a non-empty string; "
                "the app refuses to start without it (fail closed)."
            )
        return v


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings. Raises ValidationError at startup if GATEWAY_API_KEY is unset."""
    return Settings()
