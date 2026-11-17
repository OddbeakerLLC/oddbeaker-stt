"""Configuration, resolved from environment variables or explicit values.

Every backend is configurable; nothing here imports heavy optional deps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # --- STT backend -------------------------------------------------------
    stt_backend: str = field(default_factory=lambda: _env("ODDBEAKER_STT_BACKEND", "faster-whisper"))
    # faster-whisper model size or path (tiny/base/small/... or local dir)
    stt_model: str = field(default_factory=lambda: _env("ODDBEAKER_STT_MODEL", "base"))
    stt_device: str = field(default_factory=lambda: _env("ODDBEAKER_STT_DEVICE", "cpu"))
    stt_compute_type: str = field(default_factory=lambda: _env("ODDBEAKER_STT_COMPUTE", "int8"))
    stt_language: str | None = field(default_factory=lambda: os.environ.get("ODDBEAKER_STT_LANGUAGE"))

    # --- LLM cleanup backend ----------------------------------------------
    llm_backend: str = field(default_factory=lambda: _env("ODDBEAKER_STT_LLM_BACKEND", "ollama"))
    llm_base_url: str = field(default_factory=lambda: _env("ODDBEAKER_STT_LLM_BASE_URL", "http://127.0.0.1:11434"))
    llm_model: str = field(default_factory=lambda: _env("ODDBEAKER_STT_LLM_MODEL", "qwen3.8:27b-32k"))
    llm_api_key: str | None = field(default_factory=lambda: os.environ.get("ODDBEAKER_STT_LLM_API_KEY"))
    llm_timeout: float = float(_env("ODDBEAKER_STT_LLM_TIMEOUT", "120"))

    # --- Turn-end detection (streaming) ------------------------------------
    # Silence (in seconds) after which an utterance is considered finished.
    turn_silence_seconds: float = float(_env("ODDBEAKER_STT_TURN_SILENCE", "1.2"))
    # RMS below which a frame counts as silence.
    silence_rms: float = float(_env("ODDBEAKER_STT_SILENCE_RMS", "0.01"))

    # --- HTTP service ------------------------------------------------------
    host: str = field(default_factory=lambda: _env("ODDBEAKER_STT_HOST", "127.0.0.1"))
    port: int = int(_env("ODDBEAKER_STT_PORT", "9202"))

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy with the given fields replaced (None values ignored)."""
        import dataclasses

        changes = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **changes)
