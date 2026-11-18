import json

import pytest
from fastapi.testclient import TestClient

from oddbeaker_stt.cleanup import IntentCleaner
from oddbeaker_stt.config import Settings
from oddbeaker_stt.core import transcribe_bytes
from oddbeaker_stt.llm import NullLLM
from oddbeaker_stt.service import create_app
from oddbeaker_stt.stt import NullSTT


class ScriptedLLM:
    """LLM stub returning a fixed cleanup JSON."""

    name = "scripted"

    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        return self.reply


REPLY = json.dumps({
    "cleaned": "Set a timer for ten minutes.",
    "corrections": [{"said": "five minutes", "meant": "ten minutes"}],
    "notes": "",
})


def test_cleanup_parses_llm_json():
    cleaner = IntentCleaner(ScriptedLLM(f"```json\n{REPLY}\n```"))
    result = cleaner.clean("set a timer for five minutes, no wait, um, ten minutes")
    assert result.cleaned == "Set a timer for ten minutes."
    assert result.corrections[0].meant == "ten minutes"


def test_cleanup_falls_back_on_bad_reply():
    cleaner = IntentCleaner(ScriptedLLM("not json at all"))
    result = cleaner.clean("hello world")
    assert result.cleaned == "hello world"
    assert "parse failed" in result.notes


def test_cleanup_empty_transcript():
    cleaner = IntentCleaner(ScriptedLLM(REPLY))
    assert cleaner.clean("").cleaned == ""


def test_transcribe_bytes_end_to_end_with_null_backends():
    settings = Settings(stt_backend="null", llm_backend="null")
    result = transcribe_bytes(b"\x00\x00" * 1600, settings)
    assert result.verbatim == "this is a null transcription"
    assert result.cleaned == "this is a null transcription"
    assert result.stt_backend == "null"
    assert result.to_dict()["verbatim"] == result.verbatim


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        transcribe_bytes(b"\x00\x00", Settings(stt_backend="nope"))
    with pytest.raises(ValueError):
        transcribe_bytes(b"\x00\x00", Settings(stt_backend="null", llm_backend="nope"))


def test_http_service_roundtrip():
    import io

    import numpy as np
    import soundfile as sf

    app = create_app(Settings(stt_backend="null", llm_backend="null"))
    client = TestClient(app)
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    r = client.post("/transcribe", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["verbatim"] and body["cleaned"]
    assert client.get("/health").json()["status"] == "ok"


def test_null_llm_echoes():
    assert NullLLM().complete("sys", "user text") == "user text"
    assert NullSTT().transcribe(b"").text == "this is a null transcription"


def test_cleanup_falls_back_when_llm_backend_raises():
    class ExplodingLLM:
        name = "exploding"

        def complete(self, system: str, user: str) -> str:
            raise ConnectionError("ollama down")

    cleaner = IntentCleaner(ExplodingLLM())
    result = cleaner.clean("turn on the lights")
    assert result.cleaned == "turn on the lights"
    assert result.notes == "cleanup backend error"


def test_cleanup_falls_back_on_non_object_json():
    cleaner = IntentCleaner(ScriptedLLM('["a", "list"]'))
    result = cleaner.clean("hello world")
    assert result.cleaned == "hello world"
    assert "parse failed" in result.notes


def test_transcribe_file_accepts_bytes_and_path(tmp_path):
    import numpy as np
    import soundfile as sf

    from oddbeaker_stt.core import transcribe_file

    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    settings = Settings(stt_backend="null", llm_backend="null")
    by_path = transcribe_file(str(path), settings)
    by_bytes = transcribe_file(path.read_bytes(), settings)
    assert by_path.verbatim == by_bytes.verbatim == "this is a null transcription"
