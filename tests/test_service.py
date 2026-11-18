"""HTTP service contract tests — the canonical integration surface."""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from oddbeaker_stt.config import Settings
from oddbeaker_stt.service import VERSION, create_app


def _settings(**kw):
    return Settings(stt_backend="null", llm_backend="null", **kw)


def _wav_bytes(seconds=0.2, rate=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(rate * seconds), dtype=np.float32), rate, format="WAV")
    return buf.getvalue()


def _client(**kw):
    return TestClient(create_app(_settings(**kw)))


def test_health_reports_version_and_backends():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == VERSION
    assert body["stt_backend"] == "null"
    assert body["llm_backend"] == "null"


def test_transcribe_returns_full_contract():
    client = _client()
    r = client.post("/transcribe", files={"file": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    # Locked v1 contract: verbatim + cleaned + corrections[{said, meant}].
    assert set(body) >= {"verbatim", "cleaned", "corrections", "notes",
                         "language", "duration_s", "stt_backend", "llm_backend"}
    assert body["verbatim"] == "this is a null transcription"
    assert body["cleaned"] == body["verbatim"]
    assert body["corrections"] == []


def test_transcribe_rejects_corrupt_audio_with_400():
    r = _client().post("/transcribe",
                       files={"file": ("a.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400
    assert "error" in r.json()


def test_transcribe_rejects_oversize_upload_with_413():
    client = _client(max_upload_mb=0.001)  # ~1 KB limit
    r = client.post("/transcribe",
                    files={"file": ("a.wav", _wav_bytes(seconds=1), "audio/wav")})
    assert r.status_code == 413


def test_transcribe_unknown_backend_fails_fast():
    # Unknown backend config is a startup error, not a per-request error.
    with pytest.raises(ValueError):
        create_app(Settings(stt_backend="nope", llm_backend="null"))


def test_transcribe_backend_crash_returns_500(monkeypatch):
    import oddbeaker_stt.stt as stt_mod

    class BrokenSTT:
        name = "broken"

        def transcribe(self, pcm16, sample_rate=16000):
            raise RuntimeError("engine exploded")

    monkeypatch.setattr(stt_mod, "load_stt", lambda settings: BrokenSTT())
    client = TestClient(create_app(Settings(stt_backend="null", llm_backend="null")),
                        raise_server_exceptions=False)
    r = client.post("/transcribe", files={"file": ("a.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 500
    assert "engine exploded" in r.json()["error"]


def test_full_session_flow_over_http():
    client = _client(turn_silence_seconds=0.3)
    sid = client.post("/sessions").json()["session_id"]

    # status before any audio
    s = client.get(f"/sessions/{sid}").json()
    assert s == {"has_speech": False, "turn_ended": False, "buffered_seconds": 0.0}

    # feed speech (16 kHz PCM16 tone)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype(np.int16).tobytes()
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    s = client.post(f"/sessions/{sid}/audio", content=tone).json()
    assert s["has_speech"] and not s["turn_ended"]
    for _ in range(10):  # 0.05 s per chunk; cross the 0.3 s silence window
        s = client.post(f"/sessions/{sid}/audio", content=silence).json()
        if s["turn_ended"]:
            break
    assert s["turn_ended"]

    # finish -> full result, session buffers reset but stay usable
    r = client.post(f"/sessions/{sid}/finish")
    assert r.status_code == 200
    body = r.json()
    assert body["verbatim"] == "this is a null transcription"
    assert client.get(f"/sessions/{sid}").json()["buffered_seconds"] == 0.0

    # finish with no speech -> 204
    r = client.post(f"/sessions/{sid}/finish")
    assert r.status_code == 204

    # delete -> subsequent use is 404
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_session_rejects_bad_chunks_and_unknown_ids():
    client = _client()
    assert client.post("/sessions/nonexistent/audio", content=b"\x00\x00").status_code == 404
    assert client.post("/sessions/nonexistent/finish").status_code == 404
    sid = client.post("/sessions").json()["session_id"]
    assert client.post(f"/sessions/{sid}/audio", content=b"").status_code == 400
    assert client.post(f"/sessions/{sid}/audio", content=b"\x00").status_code == 400  # odd


def test_session_custom_sample_rate():
    client = _client()
    body = client.post("/sessions?sample_rate=48000").json()
    assert body["sample_rate"] == 48000
    assert client.delete(f"/sessions/{body['session_id']}").status_code == 204


def test_session_cap_evicts_oldest():
    client = _client(session_max=2)
    ids = [client.post("/sessions").json()["session_id"] for _ in range(3)]
    assert client.get(f"/sessions/{ids[0]}").status_code == 404   # oldest evicted
    assert client.get(f"/sessions/{ids[2]}").status_code == 200


def test_openapi_documents_the_contract():
    client = _client()
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/transcribe", "/sessions", "/health",
              "/sessions/{session_id}", "/sessions/{session_id}/audio",
              "/sessions/{session_id}/finish"):
        assert p in paths