"""Intent-aware speech-to-text.

Pipeline: capture -> verbatim STT -> LLM cleanup pass -> clean statement of
user intent. Both the verbatim transcript and the cleaned intent text are
always exposed. The FastAPI HTTP service (oddbeaker_stt.service) is the
canonical integration surface; this library import path is secondary.
"""

from .config import Settings
from .core import IntentResult, Correction, transcribe_file, transcribe_bytes
from .session import Session
from .turn import SilenceWindowDetector, TurnDetector

__version__ = "0.2.0"

__all__ = [
    "Settings",
    "IntentResult",
    "Correction",
    "transcribe_file",
    "transcribe_bytes",
    "Session",
    "SilenceWindowDetector",
    "TurnDetector",
    "__version__",
]