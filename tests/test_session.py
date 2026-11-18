import numpy as np
import pytest

from oddbeaker_stt.config import Settings
from oddbeaker_stt.session import Session
from oddbeaker_stt.stt import NullSTT


def _tone_chunk(seconds=0.1, rate=16000, amp=0.3):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amp * 32767).astype(np.int16).tobytes()


def _silence_chunk(seconds=0.1, rate=16000):
    return (np.zeros(int(rate * seconds), dtype=np.int16)).tobytes()


def test_turn_end_detection_and_finish():
    settings = Settings(stt_backend="null", llm_backend="null", turn_silence_seconds=0.3)
    session = Session(settings, stt=NullSTT("set a timer for five, no wait, ten minutes"))
    assert not session.turn_ended()
    session.feed(_tone_chunk())          # speech
    assert session.has_speech()
    assert not session.turn_ended()
    session.feed(_silence_chunk(0.2))    # not enough silence yet
    assert not session.turn_ended()
    session.feed(_silence_chunk(0.2))    # crosses the 0.3s window
    assert session.turn_ended()
    result = session.finish()
    assert result is not None
    assert result.verbatim == "set a timer for five, no wait, ten minutes"
    assert result.cleaned == result.verbatim  # NullLLM echoes
    # buffers reset after finish
    assert session.buffered_seconds() == 0.0
    assert not session.turn_ended()


def test_finish_without_speech_returns_none():
    session = Session(Settings(stt_backend="null", llm_backend="null"))
    session.feed(_silence_chunk(0.5))
    assert session.finish() is None


def test_silence_only_never_triggers_turn_end():
    session = Session(Settings(stt_backend="null", llm_backend="null"))
    for _ in range(20):
        session.feed(_silence_chunk(0.1))
    assert not session.turn_ended()
