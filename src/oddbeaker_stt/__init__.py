"""Intent-aware speech-to-text.

Pipeline: capture -> verbatim STT -> LLM cleanup pass -> clean statement of
user intent. Both the verbatim transcript and the cleaned intent text are
always exposed.
"""

from .config import Settings
from .core import IntentResult, Correction, transcribe_file, transcribe_bytes
from .session import Session

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "IntentResult",
    "Correction",
    "transcribe_file",
    "transcribe_bytes",
    "Session",
    "__version__",
]