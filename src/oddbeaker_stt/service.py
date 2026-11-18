"""HTTP service + CLI — the canonical integration surface.

BizAgent and third-party consumers talk to oddbeaker-stt over HTTP; the
Python library import path stays available as a secondary surface. The
service is a thin adapter over the library and adds:

- ``POST /transcribe``           one-shot: audio file -> verbatim + cleaned + corrections
- ``POST /sessions``             start a streaming session (turn-end detection server-side)
- ``POST /sessions/{id}/audio``  feed PCM16 mono chunks
- ``GET  /sessions/{id}``        poll status (has_speech / turn_ended / buffered_seconds)
- ``POST /sessions/{id}/finish`` transcribe + clean the buffered turn
- ``DELETE /sessions/{id}``      abandon a session
- ``GET  /health``               liveness + configured backends

Everything blocking runs in the threadpool so the event loop stays responsive.
"""

import argparse
import json
import sys
import threading
import time
import uuid

import anyio
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings

VERSION = "0.2.0"


def create_app(settings: Settings | None = None):
    from .core import IntentResult
    from .session import Session
    from .stt import decode_audio_file, load_stt

    settings = settings or Settings()
    app = FastAPI(title="oddbeaker-stt", version=VERSION)

    # --- shared, lazily-loaded backends (whisper model loads once) ---------
    _stt = load_stt(settings)

    def _run_transcribe(data: bytes) -> dict:
        from .cleanup import IntentCleaner
        from .llm import load_llm

        pcm16 = decode_audio_file(data)
        transcript = _stt.transcribe(pcm16, 16000)
        cleaner = IntentCleaner(load_llm(settings))
        cleanup = cleaner.clean(transcript.text)
        return IntentResult.from_parts(
            transcript, cleanup, _stt.name, cleaner.llm.name
        ).to_dict()

    # --- one-shot -----------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": VERSION,
            "stt_backend": _stt.name,
            "llm_backend": settings.llm_backend,
        }

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"error": f"upload exceeds {settings.max_upload_mb} MB"},
            )
        try:
            return await anyio.to_thread.run_sync(_run_transcribe, data)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - backend errors are server faults
            return JSONResponse(status_code=500, content={"error": str(exc)})

    # --- streaming sessions ---------------------------------------------------
    class _Entry:
        __slots__ = ("session", "last_seen")

        def __init__(self, session):
            self.session = session
            self.last_seen = time.monotonic()

    sessions: dict[str, _Entry] = {}
    sessions_lock = threading.Lock()

    def _sweep() -> None:
        """Drop idle sessions past the TTL; enforce the cap (oldest first)."""
        now = time.monotonic()
        dead = [k for k, e in sessions.items()
                if now - e.last_seen > settings.session_ttl_seconds]
        for k in dead:
            del sessions[k]
        while len(sessions) > settings.session_max:
            oldest = min(sessions, key=lambda k: sessions[k].last_seen)
            del sessions[oldest]

    def _get_session(session_id: str):
        with sessions_lock:
            entry = sessions.get(session_id)
            if entry is not None:
                entry.last_seen = time.monotonic()
        if entry is None:
            raise HTTPException(status_code=404, detail="unknown session id")
        return entry.session

    @app.post("/sessions")
    def create_session(sample_rate: int = Query(16000, gt=8000, le=192000)):
        session_id = uuid.uuid4().hex
        with sessions_lock:
            sessions[session_id] = _Entry(
                Session(settings, stt=_stt, sample_rate=sample_rate))
            _sweep()  # after insert, so the cap includes the new session
        return {"session_id": session_id, "sample_rate": sample_rate}

    @app.post("/sessions/{session_id}/audio")
    async def feed_audio(session_id: str, request: Request):
        chunk = await request.body()
        if not chunk:
            raise HTTPException(status_code=400, detail="empty audio chunk")
        if len(chunk) % 2:
            raise HTTPException(status_code=400,
                                detail="PCM16 chunk must have an even byte count")
        session = _get_session(session_id)
        await anyio.to_thread.run_sync(session.feed, chunk)
        return {
            "has_speech": session.has_speech(),
            "turn_ended": session.turn_ended(),
            "buffered_seconds": session.buffered_seconds(),
        }

    @app.get("/sessions/{session_id}")
    def session_status(session_id: str):
        session = _get_session(session_id)
        return {
            "has_speech": session.has_speech(),
            "turn_ended": session.turn_ended(),
            "buffered_seconds": session.buffered_seconds(),
        }

    @app.post("/sessions/{session_id}/finish")
    async def finish_session(session_id: str):
        session = _get_session(session_id)
        result = await anyio.to_thread.run_sync(session.finish)
        if result is None:
            return Response(status_code=204)
        return result.to_dict()

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        with sessions_lock:
            sessions.pop(session_id, None)
        return Response(status_code=204)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oddbeaker-stt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("transcribe", help="transcribe an audio file")
    p_t.add_argument("path", help="audio file (wav/flac/mp3/ogg, any soundfile format)")
    p_t.add_argument("--json", action="store_true", help="full JSON output")

    p_s = sub.add_parser("serve", help="run the HTTP service")
    p_s.add_argument("--host", default=None)
    p_s.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "transcribe":
        from .core import transcribe_file

        result = transcribe_file(args.path)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"verbatim: {result.verbatim}")
            print(f"cleaned:  {result.cleaned}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        settings = Settings().with_overrides(host=args.host, port=args.port)
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())