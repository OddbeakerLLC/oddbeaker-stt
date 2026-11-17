"""Intent cleanup: turn a verbatim transcript into what the speaker meant.

The cleanup pass runs on the FULL transcript, because corrections may span the
whole utterance ("set a timer for five— no, wait, ten minutes"). The LLM must
not invent content that is not in the transcript; it only resolves
self-corrections, false starts, mid-sentence direction changes and filler.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMBackend

SYSTEM_PROMPT = """\
You clean up raw speech-to-text transcripts. The transcript is verbatim: it \
contains false starts, self-corrections ("oops, I mean", "no wait"), \
mid-sentence direction changes, repetitions and filler words.

Your job: output what the speaker MEANT, as one clean statement or request.

Rules:
- Resolve self-corrections: the final stated version wins. Discard abandoned \
wording, but keep any content from before the correction that the speaker did \
not retract.
- Remove filler ("um", "uh", "like", "you know") and stutters.
- Preserve the speaker's meaning, names, numbers and units exactly. Never \
invent facts, never answer the request, never add suggestions.
- If the transcript is already clean, return it essentially unchanged.
- If the transcript is unintelligible, set "cleaned" to "" and say so in \
"notes".

Reply with ONLY a JSON object:
{"cleaned": "<clean statement of intent>", "corrections": [{"said": "...", \
"meant": "..."}], "notes": "<optional, empty if none>"}\
"""


@dataclass
class Correction:
    said: str
    meant: str


@dataclass
class CleanupResult:
    cleaned: str
    corrections: list[Correction] = field(default_factory=list)
    notes: str = ""
    raw_reply: str = ""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates fences/prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


class IntentCleaner:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def clean(self, verbatim: str) -> CleanupResult:
        if not verbatim.strip():
            return CleanupResult(cleaned="")
        try:
            reply = self.llm.complete(SYSTEM_PROMPT, verbatim)
        except Exception:  # noqa: BLE001 - backend/network errors must not fail the pipeline
            return CleanupResult(cleaned=verbatim, notes="cleanup backend error",
                                 raw_reply="")
        try:
            data = _extract_json(reply)
            if not isinstance(data, dict):
                raise ValueError("cleanup reply is not a JSON object")
            cleaned = str(data.get("cleaned", "")).strip()
            corrections = [
                Correction(said=str(c.get("said", "")), meant=str(c.get("meant", "")))
                for c in data.get("corrections", []) if isinstance(c, dict)
            ]
            notes = str(data.get("notes", "") or "")
            return CleanupResult(cleaned=cleaned, corrections=corrections,
                                 notes=notes, raw_reply=reply)
        except (ValueError, KeyError, TypeError):
            # Never fail the pipeline because the LLM misformatted: fall back
            # to the verbatim transcript so both outputs remain usable.
            return CleanupResult(cleaned=verbatim, notes="cleanup parse failed",
                                 raw_reply=reply)