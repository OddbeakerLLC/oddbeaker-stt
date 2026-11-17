# oddbeaker-stt

Intent-aware speech-to-text. It outputs what the speaker **meant**, not just
what it heard.

Pipeline: **capture → verbatim STT → LLM cleanup pass → clean statement of
intent.** The cleanup pass resolves self-corrections ("oops, I mean…"),
mid-sentence direction changes, false starts and filler. The verbatim
transcript is always preserved alongside the cleaned output — both are exposed
on every result.

```
mic / file ──► STT backend ──► verbatim transcript ──► LLM cleanup ──► cleaned intent
               (pluggable)         (always kept)         (pluggable)      + corrections[]
```

## Why cleanup runs on the full transcript

Corrections can span the whole utterance: *"set a timer for five— no wait, ten
minutes, and also um, remind me to stretch"*. Partial-window cleanup would
resolve "five" before the speaker retracts it. So the design is:

- **Streaming:** audio is buffered per turn; turn-end is detected by a silence
  window after speech (energy-based, configurable). Only then is the full turn
  transcribed and cleaned.
- **One-shot (file):** the whole file is one turn.

## Architecture

| Module | Role |
|---|---|
| `oddbeaker_stt.stt` | Pluggable STT backends (`FasterWhisperSTT`, `NullSTT` for tests). Audio in = PCM16 mono 16 kHz bytes. |
| `oddbeaker_stt.llm` | Pluggable LLM backends (`OllamaLLM`, `OpenAICompatLLM`, `NullLLM`). |
| `oddbeaker_stt.cleanup` | The intent-cleanup pass: strict system prompt, JSON reply (`cleaned`, `corrections[]`, `notes`), graceful fallback to verbatim on parse failure. |
| `oddbeaker_stt.session` | Streaming session with turn-end detection. |
| `oddbeaker_stt.core` | One-shot API (`transcribe_file`, `transcribe_bytes`) and `IntentResult`. |
| `oddbeaker_stt.service` | Thin FastAPI adapter + CLI. The library works without FastAPI. |

`IntentResult` always carries both outputs:

```python
result.verbatim    # what was heard
result.cleaned     # what was meant
result.corrections # [{"said": "five minutes", "meant": "ten minutes"}]
```

## Installation

```bash
pip install -e .            # library + HTTP service (no STT engine yet)
pip install -e ".[stt]"     # + faster-whisper (the default STT backend)
```

## Integration guide

### Library (embed in any Python app)

```python
import oddbeaker_stt

# One-shot: audio file -> verbatim + cleaned intent
result = oddbeaker_stt.transcribe_file("clip.wav")
print(result.verbatim)   # "set a timer for five minutes, no wait, ten minutes"
print(result.cleaned)    # "Set a timer for ten minutes."

# Streaming: feed mic chunks, detect turn end, then finish the turn
session = oddbeaker_stt.Session()          # uses Settings() from env
for chunk in mic_stream():                 # PCM16 mono 16 kHz bytes
    session.feed(chunk)
    if session.turn_ended():
        result = session.finish()          # IntentResult or None
        break
```

### HTTP service (language-agnostic apps)

```bash
oddbeaker-stt serve --host 127.0.0.1 --port 9202
```

```bash
curl -F file=@clip.wav http://127.0.0.1:9202/transcribe
```

```json
{
  "verbatim": "set a timer for five minutes, no wait, ten minutes",
  "cleaned": "Set a timer for ten minutes.",
  "corrections": [{"said": "five minutes", "meant": "ten minutes"}],
  "notes": "", "language": "en", "duration_s": 4.2,
  "stt_backend": "faster-whisper", "llm_backend": "ollama"
}
```

Endpoints: `POST /transcribe` (multipart file), `GET /health`.

### CLI

```bash
oddbeaker-stt transcribe clip.wav          # prints verbatim + cleaned
oddbeaker-stt transcribe clip.wav --json   # full JSON
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ODDBEAKER_STT_BACKEND` | `faster-whisper` | `faster-whisper` or `null` (dev) |
| `ODDBEAKER_STT_MODEL` | `base` | Whisper model size or local path |
| `ODDBEAKER_STT_DEVICE` | `cpu` | `cpu` / `cuda` |
| `ODDBEAKER_STT_COMPUTE` | `int8` | CTranslate2 compute type |
| `ODDBEAKER_STT_LANGUAGE` | auto-detect | Force a language code |
| `ODDBEAKER_STT_LLM_BACKEND` | `ollama` | `ollama`, `openai-compat`, `null` |
| `ODDBEAKER_STT_LLM_BASE_URL` | `http://127.0.0.1:11434` | LLM endpoint base |
| `ODDBEAKER_STT_LLM_MODEL` | `qwen3.8:27b-32k` | Cleanup model |
| `ODDBEAKER_STT_LLM_API_KEY` | – | For OpenAI-compatible endpoints |
| `ODDBEAKER_STT_TURN_SILENCE` | `1.2` | Silence seconds that end a turn |
| `ODDBEAKER_STT_SILENCE_RMS` | `0.01` | RMS threshold for "silence" |
| `ODDBEAKER_STT_HOST` / `ODDBEAKER_STT_PORT` | `127.0.0.1` / `9202` | HTTP service bind |

## Bringing your own backends

Both boundaries are `Protocol`s — implement `transcribe(pcm16, sample_rate)`
or `complete(system, user)` and pass the object directly:

```python
from oddbeaker_stt.core import transcribe_bytes
from oddbeaker_stt.session import Session

session = Session(stt=my_custom_stt)          # STTBackend protocol
result = transcribe_bytes(pcm, settings, ...) # or inject via Settings
```

Minimal dependencies: fastapi/uvicorn (service only), numpy, soundfile, httpx.
The STT engine (faster-whisper) is an optional extra; the LLM is any local or
remote chat endpoint.

## Design notes for embedding (BizAgent)

- The library layer has **no** FastAPI import; embedders pay only for what they use.
- `Session` is single-turn by design; the host app owns the mic loop and
  barge-in policy. `turn_ended()` is a poll, not a callback, so it fits any
  event loop.
- Cleanup never invents content: the system prompt forbids answering or
  extending the request, and a malformed LLM reply falls back to verbatim
  rather than failing.
- Both outputs are first-class, so a host can show the raw transcript for
  debugging/audit while acting on the cleaned intent.

## Status

Alpha. Working end-to-end prototype (file in → verbatim + intent out) with
faster-whisper + Ollama. Next: mic capture helper, word timestamps for
correction alignment, tests against real audio fixtures.
