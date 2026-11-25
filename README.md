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

## Integration contract (v1)

**The HTTP service is the canonical integration surface.** BizAgent and
external consumers talk HTTP; the Python library import path remains available
as a secondary surface for Python embedders. Locked v1 decisions:

- **Turn-end detection is silence-window-only** in v1. The detector is a pluggable protocol (`oddbeaker_stt.turn.TurnDetector`), so a real VAD (silero / webrtcvad) can slot in later without API churn.
- **No word timestamps in v1.** The `corrections: [{said, meant}]` shape is deliberately open so span offsets can be added later without breaking consumers.

### Result shape (every endpoint that returns a result)

```json
{
  "verbatim": "set a timer for five minutes, no wait, ten minutes",
  "cleaned": "Set a timer for ten minutes.",
  "corrections": [{"said": "five minutes", "meant": "ten minutes"}],
  "notes": "",
  "language": "en",
  "duration_s": 4.2,
  "stt_backend": "faster-whisper",
  "llm_backend": "ollama"
}
```

`verbatim` and `cleaned` are always present. `corrections` records each
self-correction the cleanup pass resolved. On any cleanup failure (LLM down,
malformed reply) the pipeline **never fails**: `cleaned` falls back to
`verbatim` and `notes` says why.

## HTTP API

Run it:

```bash
oddbeaker-stt serve                 # 127.0.0.1:9202 by default
```

### `GET /health`

`{"status": "ok", "version": "0.2.0", "stt_backend": "faster-whisper", "llm_backend": "ollama"}`

### `POST /transcribe` — one-shot

Multipart upload of any soundfile-supported audio (wav/flac/mp3/ogg…):

```bash
curl -F file=@clip.wav http://127.0.0.1:9202/transcribe
```

Errors: `400` corrupt/unsupported audio, `413` upload over
`ODDBEAKER_STT_MAX_UPLOAD_MB` (default 25), `500` backend crash. All error
bodies are `{"error": "..."}`.

JavaScript (BizAgent-style consumer):

```js
const form = new FormData();
form.append("file", new Blob([wavBytes], { type: "audio/wav" }), "clip.wav");
const r = await fetch("http://127.0.0.1:9202/transcribe", { method: "POST", body: form });
const { verbatim, cleaned, corrections } = await r.json();
```

### Streaming sessions — turn-end detection server-side

For mic-style clients that stream chunks and want the server to decide when
the turn is over:

| Endpoint | Purpose |
|---|---|
| `POST /sessions?sample_rate=16000` | Create a session → `{"session_id", "sample_rate"}` |
| `POST /sessions/{id}/audio` | Raw PCM16 mono body (one chunk) → `{"has_speech", "turn_ended", "buffered_seconds"}` |
| `GET /sessions/{id}` | Poll status without feeding audio |
| `POST /sessions/{id}/finish` | Transcribe + clean the buffered turn → result JSON, or `204` if no speech |
| `DELETE /sessions/{id}` | Abandon the session (idempotent, `204`) |

Sessions are in-memory, capped (`ODDBEAKER_STT_SESSION_MAX`, default 100) and
reaped after an idle TTL (`ODDBEAKER_STT_SESSION_TTL`, default 3600 s). Feed
chunks at the sample rate you created the session with; chunks must be even
byte counts (PCM16). Cleanup always runs on the **full buffered turn**, never
on partials — corrections may span the whole utterance.

```js
// BizAgent-style loop
const { session_id } = await post("/sessions?sample_rate=16000");
for (;;) {
  const chunk = await mic.readPCM16();            // Int16 mono bytes
  const s = await post(`/sessions/${session_id}/audio`, chunk);
  if (s.turn_ended) {
    const result = await post(`/sessions/${session_id}/finish`);
    break;
  }
}
```

## Library (secondary surface, Python embedders)

```python
import oddbeaker_stt

result = oddbeaker_stt.transcribe_file("clip.wav")   # one-shot
result.verbatim; result.cleaned; result.corrections

session = oddbeaker_stt.Session()                    # streaming
for chunk in mic_stream():                           # PCM16 mono 16 kHz bytes
    session.feed(chunk)
    if session.turn_ended():
        result = session.finish()                    # IntentResult or None
```

Custom turn detector (VAD-ready):

```python
from oddbeaker_stt import Session, TurnDetector

class MyVAD:                       # implement feed/has_speech/turn_ended/reset
    ...

session = Session(detector=MyVAD())
```

## CLI

```bash
oddbeaker-stt transcribe clip.wav          # prints verbatim + cleaned
oddbeaker-stt transcribe clip.wav --json   # full JSON
```

## Architecture

| Module | Role |
|---|---|
| `oddbeaker_stt.stt` | Pluggable STT backends (`FasterWhisperSTT`, `NullSTT` for tests). Audio in = PCM16 mono 16 kHz bytes. Backends are cached per config so the service loads the whisper model once. |
| `oddbeaker_stt.llm` | Pluggable LLM backends (`OllamaLLM`, `OpenAICompatLLM`, `NullLLM`). |
| `oddbeaker_stt.cleanup` | The intent-cleanup pass: strict system prompt, JSON reply (`cleaned`, `corrections[]`, `notes`), graceful fallback to verbatim on backend error or parse failure. |
| `oddbeaker_stt.turn` | Turn-end detection: `TurnDetector` protocol + `SilenceWindowDetector` (v1 default). VAD slots in here. |
| `oddbeaker_stt.session` | Streaming session using a pluggable detector. |
| `oddbeaker_stt.core` | One-shot API (`transcribe_file`, `transcribe_bytes`) and `IntentResult`. |
| `oddbeaker_stt.service` | The canonical FastAPI HTTP surface + CLI. |

## Installation

```bash
pip install -e .            # library + HTTP service (no STT engine yet)
pip install -e ".[stt]"     # + faster-whisper (the default STT backend)
```

`python-multipart` is a **hard runtime dependency** (not optional): the
`POST /transcribe` endpoint consumes multipart uploads, so it is installed
automatically with the base package.

**Canonical branch is `main`.** The old `master` branch was deleted
(2026-11-18); clone the default branch or `git clone --branch main`. Any doc
or script still referencing `master` is stale.

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
| `ODDBEAKER_STT_LLM_TIMEOUT` | `120` | LLM call timeout (s) |
| `ODDBEAKER_STT_TURN_SILENCE` | `1.2` | Silence seconds that end a turn |
| `ODDBEAKER_STT_SILENCE_RMS` | `0.01` | RMS threshold for "silence" |
| `ODDBEAKER_STT_HOST` / `ODDBEAKER_STT_PORT` | `127.0.0.1` / `9202` | HTTP service bind |
| `ODDBEAKER_STT_MAX_UPLOAD_MB` | `25` | `/transcribe` upload limit |
| `ODDBEAKER_STT_SESSION_TTL` | `3600` | Idle seconds before a session is reaped |
| `ODDBEAKER_STT_SESSION_MAX` | `100` | Max concurrent sessions (oldest evicted) |

## Design notes

- The library layer has **no** FastAPI import; embedders pay only for what they use.
- `Session` is single-turn by design; the host app owns the mic loop and barge-in policy. `turn_ended()` is a poll, not a callback, so it fits any event loop — and the same logic is exposed over HTTP via `/sessions`.
- Cleanup never invents content: the system prompt forbids answering or extending the request, and any LLM failure (network, malformed JSON, non-object reply) falls back to verbatim rather than failing the pipeline.
- Both outputs are first-class, so a host can show the raw transcript for debugging/audit while acting on the cleaned intent.
- Blocking work (whisper, LLM) runs in the worker threadpool; the event loop stays responsive under concurrent requests.

## Status

**v1 feature-complete.** HTTP contract finalized (`/transcribe` + streaming
`/sessions`), silence-window turn detection (VAD pluggable), corrections shape
open for span offsets later. Verified end-to-end with real backends
(faster-whisper base/CPU + Ollama qwen3.8) using TTS-generated speech: one-shot
and session flows both produce correct verbatim + cleaned output with the
correction captured. Known limitations: energy-based turn detection struggles
in noisy environments (swap in a VAD later); linear-resample audio decode is
approximate; word timestamps deferred out of v1.
