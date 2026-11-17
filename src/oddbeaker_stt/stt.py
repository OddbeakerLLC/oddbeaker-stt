"""Pluggable STT backends.

A backend turns raw PCM16 mono 16 kHz audio bytes into a verbatim transcript.
Heavy dependencies (faster-whisper) are imported lazily so the package stays
lightweight for embedders that bring their own backend.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str | None = None
    duration_s: float | None = None


class STTBackend(Protocol):
    name: str

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> Transcript:
        """Transcribe PCM16 mono audio bytes to verbatim text."""
        ...


class NullSTT:
    """Dev/test backend: returns a fixed transcript without touching audio."""

    name = "null"

    def __init__(self, text: str = "this is a null transcription"):
        self.text = text

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> Transcript:
        return Transcript(text=self.text)


class FasterWhisperSTT:
    """Verbatim STT via faster-whisper (CTranslate2 Whisper)."""

    name = "faster-whisper"

    def __init__(self, model: str = "base", device: str = "cpu",
                 compute_type: str = "int8", language: str | None = None):
        from faster_whisper import WhisperModel  # lazy heavy import

        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> Transcript:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio, language=self._language, beam_size=1, vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text=text, language=info.language,
                          duration_s=getattr(info, "duration", None))


def load_stt(settings) -> STTBackend:
    """Build the configured STT backend."""
    kind = settings.stt_backend
    if kind == "null":
        return NullSTT()
    if kind in ("faster-whisper", "whisper"):
        return FasterWhisperSTT(
            model=settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            language=settings.stt_language,
        )
    raise ValueError(f"unknown stt_backend: {kind!r}")


def decode_audio_file(data: bytes, target_rate: int = 16000) -> bytes:
    """Decode any soundfile-supported audio file to PCM16 mono @ target_rate."""
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # downmix to mono
    if rate != target_rate:
        # simple linear resample; good enough for speech prototypes
        n_out = int(round(len(audio) * target_rate / rate))
        if n_out > 0:
            idx = np.linspace(0, len(audio) - 1, n_out)
            audio = np.interp(idx, np.arange(len(audio)), audio)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
