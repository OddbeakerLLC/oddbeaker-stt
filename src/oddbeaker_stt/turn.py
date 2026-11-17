"""Turn-end detection.

v1 ships a silence-window detector only (locked decision). The
:class:`TurnDetector` protocol keeps the backend pluggable so a real VAD
(silero / webrtcvad) can slot in later without changing the Session or HTTP
API.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TurnDetector(Protocol):
    """Decides when the speaker has finished a turn.

    Implementations receive every audio chunk in order and must be
    stateful per session. ``feed`` returns nothing; callers poll
    ``turn_ended()`` / ``has_speech()``.
    """

    def feed(self, pcm16_chunk: bytes, sample_rate: int) -> None:
        """Consume one chunk of PCM16 mono audio."""
        ...

    def has_speech(self) -> bool:
        """True once speech (not just noise) has been seen."""
        ...

    def turn_ended(self) -> bool:
        """True once the turn is considered complete."""
        ...

    def reset(self) -> None:
        """Clear all state for reuse."""
        ...


def chunk_rms(pcm16_chunk: bytes) -> float:
    """RMS (0..1) of a PCM16 mono chunk; 0.0 for empty input."""
    if not pcm16_chunk:
        return 0.0
    x = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


class SilenceWindowDetector:
    """Energy-based turn-end: a silence window after speech ends the turn.

    This is the v1 default. It is deliberately dumb about content — it only
    measures level — so it works with any mic and no model download. Swap in
    a VAD-backed detector for noisy environments later.
    """

    name = "silence-window"

    def __init__(self, silence_seconds: float = 1.2, rms_threshold: float = 0.01):
        self.silence_seconds = silence_seconds
        self.rms_threshold = rms_threshold
        self._silence_s = 0.0
        self._speech_seen = False

    def feed(self, pcm16_chunk: bytes, sample_rate: int) -> None:
        rms = chunk_rms(pcm16_chunk)
        if rms >= self.rms_threshold:
            self._speech_seen = True
            self._silence_s = 0.0
        else:
            self._silence_s += len(pcm16_chunk) / (2 * sample_rate)

    def has_speech(self) -> bool:
        return self._speech_seen

    def turn_ended(self) -> bool:
        return self._speech_seen and self._silence_s >= self.silence_seconds

    def reset(self) -> None:
        self._silence_s = 0.0
        self._speech_seen = False