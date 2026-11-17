"""Tests for the pluggable turn-end detection layer."""

import numpy as np
import pytest

from oddbeaker_stt.config import Settings
from oddbeaker_stt.session import Session
from oddbeaker_stt.stt import NullSTT
from oddbeaker_stt.turn import SilenceWindowDetector, chunk_rms


def _tone_chunk(seconds=0.1, rate=16000, amp=0.3):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * amp * 32767).astype(np.int16).tobytes()


def _silence_chunk(seconds=0.1, rate=16000):
    return np.zeros(int(rate * seconds), dtype=np.int16).tobytes()


class AlwaysEndedDetector:
    """Minimal custom detector proving the protocol is enough to plug in."""

    name = "always-ended"

    def __init__(self):
        self.fed = 0

    def feed(self, pcm16_chunk: bytes, sample_rate: int) -> None:
        self.fed += 1

    def has_speech(self) -> bool:
        return True

    def turn_ended(self) -> bool:
        return self.fed >= 2

    def reset(self) -> None:
        self.fed = 0


def test_chunk_rms_basics():
    assert chunk_rms(b"") == 0.0
    assert chunk_rms(_silence_chunk()) == 0.0
    assert chunk_rms(_tone_chunk()) > 0.1


def test_silence_window_detector_state_and_reset():
    det = SilenceWindowDetector(silence_seconds=0.3, rms_threshold=0.01)
    assert not det.has_speech() and not det.turn_ended()
    det.feed(_tone_chunk(), 16000)
    assert det.has_speech() and not det.turn_ended()
    det.feed(_silence_chunk(0.4), 16000)
    assert det.turn_ended()
    det.reset()
    assert not det.has_speech() and not det.turn_ended()


def test_session_accepts_custom_detector():
    det = AlwaysEndedDetector()
    session = Session(Settings(stt_backend="null", llm_backend="null"),
                      stt=NullSTT("hello"), detector=det)
    session.feed(_tone_chunk())
    assert not session.turn_ended()
    session.feed(_tone_chunk())
    assert session.turn_ended()
    result = session.finish()
    assert result is not None and result.verbatim == "hello"
    assert det.fed == 0  # reset() cleared detector state with the buffers


def test_default_detector_comes_from_settings():
    settings = Settings(stt_backend="null", llm_backend="null",
                        turn_silence_seconds=0.05, silence_rms=0.01)
    session = Session(settings, stt=NullSTT())
    session.feed(_tone_chunk())
    session.feed(_silence_chunk(0.1))
    assert session.turn_ended()


def test_detector_protocol_satisfied_by_default_impl():
    # SilenceWindowDetector must satisfy the TurnDetector protocol shape.
    det = SilenceWindowDetector()
    for attr in ("feed", "has_speech", "turn_ended", "reset"):
        assert callable(getattr(det, attr))