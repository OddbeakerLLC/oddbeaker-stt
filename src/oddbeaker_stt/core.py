"""Core library API: one-shot transcription + cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cleanup import CleanupResult, Correction, IntentCleaner
from .config import Settings
from .llm import load_llm
from .stt import Transcript, decode_audio_file, load_stt

__all__ = ["IntentResult", "transcribe_file", "transcribe_bytes"]


@dataclass
class IntentResult:
    """Both outputs are always present: verbatim + cleaned intent."""

    verbatim: str
    cleaned: str
    corrections: list[Correction] = field(default_factory=list)
    notes: str = ""
    language: Optional[str] = None
    duration_s: Optional[float] = None
    stt_backend: str = ""
    llm_backend: str = ""

    @classmethod
    def from_parts(cls, transcript: Transcript, cleanup: CleanupResult,
                   stt_name: str, llm_name: str) -> "IntentResult":
        return cls(
            verbatim=transcript.text,
            cleaned=cleanup.cleaned,
            corrections=cleanup.corrections,
            notes=cleanup.notes,
            language=transcript.language,
            duration_s=transcript.duration_s,
            stt_backend=stt_name,
            llm_backend=llm_name,
        )

    def to_dict(self) -> dict:
        return {
            "verbatim": self.verbatim,
            "cleaned": self.cleaned,
            "corrections": [{"said": c.said, "meant": c.meant} for c in self.corrections],
            "notes": self.notes,
            "language": self.language,
            "duration_s": self.duration_s,
            "stt_backend": self.stt_backend,
            "llm_backend": self.llm_backend,
        }


def transcribe_bytes(pcm16: bytes, settings: Settings | None = None,
                     *, sample_rate: int = 16000) -> IntentResult:
    """PCM16 mono audio bytes -> verbatim + intent-cleaned text."""
    settings = settings or Settings()
    stt = load_stt(settings)
    cleaner = IntentCleaner(load_llm(settings))
    transcript = stt.transcribe(pcm16, sample_rate)
    cleanup = cleaner.clean(transcript.text)
    return IntentResult.from_parts(transcript, cleanup, stt.name, cleaner.llm.name)


def transcribe_file(path_or_bytes, settings: Settings | None = None) -> IntentResult:
    """Audio file (path or raw bytes of any soundfile-supported format)
    -> verbatim + intent-cleaned text."""
    settings = settings or Settings()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            data = f.read()
    pcm16 = decode_audio_file(data)
    return transcribe_bytes(pcm16, settings)