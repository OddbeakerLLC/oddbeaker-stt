# oddbeaker-stt — Sitemap (local only; gitignored)
_Last updated: 2026-11-17_

## Overview
Intent-aware STT module: verbatim transcription + LLM cleanup pass that outputs what the speaker meant. Designed for embedding in BizAgent and any third-party app (library + HTTP service).

## Structure
- `src/oddbeaker_stt/` — `stt.py` (pluggable STT backends; faster-whisper default, lazy import), `llm.py` (pluggable LLM backends: ollama / openai-compat / null), `cleanup.py` (intent-cleanup prompt + JSON parse + verbatim fallback), `session.py` (streaming session, energy-based turn-end detection), `core.py` (one-shot API, `IntentResult` with verbatim+cleaned), `service.py` (FastAPI `/transcribe` + CLI), `config.py` (env-driven Settings).
- `tests/` — cleanup parsing, null-backend e2e, HTTP roundtrip, turn-end detection.
- README.md — architecture + integration guide (public).

## Key Integrations
- STT: faster-whisper (optional extra `[stt]`), CPU int8 default.
- LLM: local Ollama at 127.0.0.1:11434 (default model qwen3.8:27b-32k) or any OpenAI-compatible endpoint.
- Test audio can be synthesized via oddbeaker-tts service on 127.0.0.1:9201.

## Active Work
- 2026-11-17: scaffolded repo, API design, tests, e2e prototype (TTS-generated speech → whisper → Ollama cleanup). Awaiting hub design check before deep implementation.

## Known Issues
- Turn-end detection is energy-based only (no VAD model yet) — noisy environments may need webrtcvad/silero.
- Linear-resample audio decode is approximate; fine for speech prototype.
