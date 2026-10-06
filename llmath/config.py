"""Settings and the model registry, read from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'llmath.db').as_posix()}")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
API_URL = os.getenv("LLMATH_API_URL", "http://127.0.0.1:8000")
DATA_DIR = ROOT / "data"
PILOT_FILE = DATA_DIR / "pilot.jsonl"


@dataclass(frozen=True)
class ModelSpec:
    key: str                    # short id stored in the database and shown in the UI
    provider: str               # "ollama" (local) or "litellm" (cloud APIs)
    model: str                  # model name as the provider expects it
    label: str                  # display name
    think: bool | None = None   # Ollama reasoning switch; None = model default
    env_key: str | None = None  # environment variable holding the API key
    max_tokens: int = 4096


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec("qwen3-8b", "ollama", "qwen3:8b", "Qwen3 8B (yerel)", think=False),
        ModelSpec("qwen3-8b-think", "ollama", "qwen3:8b", "Qwen3 8B, düşünme açık (yerel)",
                  think=True, max_tokens=8192),
        ModelSpec("llama3.1-8b", "ollama", "llama3.1:8b", "Llama 3.1 8B (yerel)"),
        ModelSpec("gemini-flash-lite", "litellm", "gemini/gemini-3.5-flash-lite",
                  "Gemini 3.5 Flash-Lite", env_key="GEMINI_API_KEY"),
        ModelSpec("gemini-flash", "litellm", "gemini/gemini-3.8-flash",
                  "Gemini 3.8 Flash", env_key="GEMINI_API_KEY"),
    ]
}


def has_api_key(spec: ModelSpec) -> bool:
    return spec.env_key is None or bool(os.getenv(spec.env_key))
