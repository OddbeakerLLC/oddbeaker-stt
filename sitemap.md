# oddbeaker-stt — Sitemap (local only; gitignored)
_Last updated: 2026-11-17_

## Overview
Intent-aware STT module: verbatim transcription + LLM cleanup pass that outputs what the speaker meant. Designed for embedding in BizAgent and any third-party app. **The FastAPI HTTP service (127.0.0.1:9202) is the canonical integration surface** (locked decision); the Python library import path is secondary.

## Structure
- `src/oddbeaker_stt/` — `stt.py` (pluggable STT backends; faster-whisper default, lazy import, per-config cached so the service loads the model once), `llm.py` (pluggable LLM backends: ollama / openai-compat / null), `cleanup.py` (intent-cleanup prompt + JSON parse + verbatim fallback on backend error or parse failure), `turn.py` (`TurnDetector` protocol + `SilenceWindowDetector` v1 default; VAD slots in here), `session.py` (streaming session using a pluggable detector), `core.py` (one-shot API, `IntentResult` with verbatim+cleaned), `service.py` (canonical FastAPI surface: `/health`, `/transcribe`, `/sessions*` + CLI), `config.py` (env-driven Settings).
- `tests/` — cleanup parsing/fallbacks (incl. LLM-down fallback), null-backend e2e, HTTP contract tests (one-shot + full session flow, error codes 400/413/500), turn detection incl. custom detector injection.
- README.md — HTTP-first integration guide (curl + JS examples), architecture, config.

## Key Integrations
- STT: faster-whisper (optional extra `[stt]`), CPU int8 default.
- LLM: local Ollama at 127.0.0.1:11434 (default model qwen3.8:27b-32k) or any OpenAI-compatible endpoint.
- Test audio can be synthesized via oddbeaker-tts service on 127.0.0.1:9201.
- BizAgent (JS/shell) integrates over HTTP only.

## Active Work
- v1 feature-complete (2026-11-17): API contract finalized (verbatim + cleaned + corrections[{said, meant}]; timestamps deferred, shape left open for span offsets), silence-window-only turn detection with pluggable detector, streaming sessions over HTTP, hardened error paths, 29 tests green, real-backend e2e verified (TTS speech → whisper → Ollama, one-shot + session).

## Known Issues
- Turn-end detection is energy-based only (no VAD model yet) — noisy environments may need webrtcvad/silero; detector protocol is ready for it.
- Linear-resample audio decode is approximate; fine for speech.
- HTTP session store is in-memory (single-process only); fine for the planned single-host deployment.

## Remote
- `origin`: `ssh://git@github.com/OddbeakerLLC/oddbeaker-stt.git` — **canonical branch: `main`** (GitHub default; master deleted 2026-11-18). Local branch `main` tracks `origin/main`.
