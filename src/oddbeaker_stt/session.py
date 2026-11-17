"""Streaming session with turn-end detection.

Feed audio chunks as they arrive; the session detects end-of-turn by silence
(energy-based) and emits a final IntentResult for the completed utterance.
Corrections may span the whole utterance, so cleanup always runs on the full
turn transcript — never on partials.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .config import Settings
from .core import IntentResult
from .llm import load_llm
from .stt import STTBackend, load_stt


class Session:
    """One utterance at a time: feed() audio, poll turn_ended(), finish().

    Usage (push-to-talk or VAD-driven apps):
        session = Session()
        for chunk in mic_stream():
            session.feed(chunk)
            if session.turn_ended():
                result = session.finish()   # IntentResult (verbatim + cleaned)
                break
    """

    def __init__(self, settings: Settings | None = None, *,
                 stt: STTBackend | None = None,
                 sample_rate: int = 16000):
        self.settings = settings or Settings()
        self.sample_rate = sample_rate
        self._stt = stt or load_stt(self.settings)
        self._buffer = bytearray()
        self._silence_s = 0.0
        self._speech_seen = False
        self._started_at: Optional[float] = None

    def feed(self, pcm16_chunk: bytes) -> None:
        """Feed one chunk of PCM16 mono audio at self.sample_rate."""
        if self._started_at is None:
            self._started_at = time.time()
        self._buffer.extend(pcm16_chunk)
        rms = self._chunk_rms(pcm16_chunk)
        if rms >= self.settings.silence_rms:
            self._speech_seen = True
            self._silence_s = 0.0
        else:
            self._silence_s += len(pcm16_chunk) / (2 * self.sample_rate)

    def _chunk_rms(self, chunk: bytes) -> float:
        if not chunk:
            return 0.0
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

    def turn_ended(self) -> bool:
        """True once speech was seen and the trailing silence window elapsed."""
        return self._speech_seen and self._silence_s >= self.settings.turn_silence_seconds

    def has_speech(self) -> bool:
        return self._speech_seen

    def buffered_seconds(self) -> float:
        return len(self._buffer) / (2 * self.sample_rate)

    def finish(self, cleaner_factory: Callable | None = None) -> Optional[IntentResult]:
        """Transcribe the buffered turn and run the intent cleanup.

        Returns None if no speech was captured. Buffers are reset afterwards.
        """
        if not self._buffer or not self._speech_seen:
            self.reset()
            return None
        pcm16 = bytes(self._buffer)
        self.reset()
        from .cleanup import IntentCleaner

        cleaner = (cleaner_factory or (lambda: IntentCleaner(load_llm(self.settings))))()
        transcript = self._stt.transcribe(pcm16, self.sample_rate)
        cleanup = cleaner.clean(transcript.text)
        return IntentResult.from_parts(transcript, cleanup, self._stt.name, cleaner.llm.name)

    def reset(self) -> None:
        self._buffer = bytearray()
        self._silence_s = 0.0
        self._speech_seen = False
        self._started_at = None