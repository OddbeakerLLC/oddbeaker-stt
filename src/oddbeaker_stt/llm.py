"""Pluggable LLM backends for the intent-cleanup pass.

Any chat-completions-style endpoint works: Ollama, llama.cpp server, vLLM,
OpenAI-compatible APIs, ...
"""

from __future__ import annotations

from typing import Protocol

import httpx


class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's reply text for one system+user exchange."""
        ...


class NullLLM:
    """Dev/test backend: echoes the user prompt back (no network)."""

    name = "null"

    def complete(self, system: str, user: str) -> str:
        return user


class OllamaLLM:
    """Chat completion via a local Ollama daemon (/api/chat)."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatLLM:
    """Chat completion via any OpenAI-compatible /v1/chat/completions endpoint."""

    name = "openai-compat"

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload,
                       headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def load_llm(settings) -> LLMBackend:
    """Build the configured LLM cleanup backend."""
    kind = settings.llm_backend
    if kind == "null":
        return NullLLM()
    if kind == "ollama":
        return OllamaLLM(settings.llm_base_url, settings.llm_model, settings.llm_timeout)
    if kind in ("openai", "openai-compat", "llamacpp"):
        return OpenAICompatLLM(settings.llm_base_url, settings.llm_model,
                               settings.llm_api_key, settings.llm_timeout)
    raise ValueError(f"unknown llm_backend: {kind!r}")