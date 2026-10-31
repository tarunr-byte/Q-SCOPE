"""Application configuration for Q-SCOPE in production and local environments.

Configuration is loaded from environment variables and an optional project-local ``.env``.
Supports automatic detection of Railway.com DATABASE_URL and PORT.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int, minimum: int) -> int:
    """Parse an integer environment variable with a fail-fast error."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float | None = None) -> float:
    """Parse a bounded finite float from the environment."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bound}")
    return value


def _project_path(name: str, default: str) -> Path:
    """Resolve configured checkpoint paths relative to the repository root."""
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_model_map(raw: str) -> dict[str, str]:
    """Parse the optional JSON map of agent names to provider model IDs."""
    try:
        data: Any = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("NVIDIA_MODELS_JSON must contain valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("NVIDIA_MODELS_JSON must be a JSON object")
    result: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("NVIDIA_MODELS_JSON keys and values must be strings")
        key, value = key.strip(), value.strip()
        if key and value:
            result[key] = value
    return result


def _load_origins(raw: str) -> tuple[str, ...]:
    """Parse CORS origins without enabling a wildcard with credentials."""
    origins = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if not origins:
        return ("*",)
    return origins


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, validated configuration shared across application modules."""

    project_root: Path
    app_env: str
    api_access_token: str
    cors_origins: tuple[str, ...]
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_timeout_seconds: float
    nvidia_models: dict[str, str]
    max_waveform_samples: int
    max_phase_uncertainty_rad: float
    hcqa_checkpoint: Path
    agent7_checkpoint: Path
    database_url: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Build settings after validating all values that affect runtime safety."""
        model_map = _load_model_map(os.getenv("NVIDIA_MODELS_JSON", "{}"))
        origins = _load_origins(
            os.getenv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:8000",
            )
        )
        base_url = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip().rstrip("/")
        if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("NVIDIA_BASE_URL must use HTTPS (HTTP is allowed only for localhost)")

        # Database URL resolution (Railway PostgreSQL or SQLite)
        db_url = os.getenv("DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'qscope.db').as_posix()}").strip()
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)

        return cls(
            project_root=PROJECT_ROOT,
            app_env=os.getenv("APP_ENV", "development").strip().lower(),
            api_access_token=os.getenv("API_ACCESS_TOKEN", "").strip(),
            cors_origins=origins,
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
            nvidia_base_url=base_url,
            nvidia_timeout_seconds=_env_float("NVIDIA_TIMEOUT_SECONDS", 20.0, 0.5, 180.0),
            nvidia_models=model_map,
            max_waveform_samples=_env_int("MAX_WAVEFORM_SAMPLES", 65536, 16),
            max_phase_uncertainty_rad=_env_float(
                "MAX_PHASE_UNCERTAINTY_RAD", 0.10, 1e-9, 3.141592653589793
            ),
            hcqa_checkpoint=_project_path("HCQA_CHECKPOINT", "models/hcqa_dqn.npz"),
            agent7_checkpoint=_project_path("AGENT7_CHECKPOINT", "models/agent7_multiview.pt"),
            database_url=db_url,
        )

    @property
    def auth_required(self) -> bool:
        """Production mode refuses to silently run without API authentication only when token is set."""
        return bool(self.api_access_token)


settings = Settings.from_environment()

# Compatibility constants
NVIDIA_API_KEY = settings.nvidia_api_key
NVIDIA_BASE_URL = settings.nvidia_base_url
NVIDIA_TIMEOUT_SECONDS = settings.nvidia_timeout_seconds
NVIDIA_MODELS = settings.nvidia_models
MAX_WAVEFORM_SAMPLES = settings.max_waveform_samples
MAX_PHASE_UNCERTAINTY_RAD = settings.max_phase_uncertainty_rad
HCQA_CHECKPOINT = settings.hcqa_checkpoint
AGENT7_CHECKPOINT = settings.agent7_checkpoint
API_ACCESS_TOKEN = settings.api_access_token
CORS_ORIGINS = list(settings.cors_origins)
DATABASE_URL = settings.database_url
