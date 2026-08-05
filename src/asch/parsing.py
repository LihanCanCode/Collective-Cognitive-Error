"""Parse model output into the variables the analysis needs.

Parsing failures are recorded as ``None``, never guessed. A silently mis-parsed answer would
show up as a spurious conformity or non-conformity event, so the parse rate is itself a reported
diagnostic -- if it drops, the prompt format is failing and the cell needs a rerun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

ANSWER_RE = re.compile(r"\bAnswer\s*[:\-]\s*\(?([ABC])\b", re.IGNORECASE)
FALLBACK_RE = re.compile(r"^\s*\(?([ABC])[\).\s]", re.MULTILINE)
CONFIDENCE_RE = re.compile(r"\bConfidence\s*[:\-]\s*(\d{1,3})", re.IGNORECASE)
ERRORS_FOUND_RE = re.compile(r"\bErrors found\s*[:\-]\s*(\d{1,3})", re.IGNORECASE)


class Stance(str, Enum):
    """Classification of the naive agent's answer relative to truth and the majority.

    Mirrors Shehata & Li's ADOPTED/REJECTED/IGNORED/UNKNOWN scheme so results are directly
    comparable with theirs.
    """

    ADOPTED = "adopted"      # took the majority's wrong answer -- conformity
    REJECTED = "rejected"    # held the correct answer under pressure -- independence
    IGNORED = "ignored"      # third answer: neither truth nor majority
    UNKNOWN = "unknown"      # unparseable


@dataclass
class ParsedAnswer:
    answer: str | None
    confidence: int | None
    raw: str

    @property
    def parsed(self) -> bool:
        return self.answer is not None


def parse_answer(text: str) -> ParsedAnswer:
    """Extract the answer letter and self-reported confidence.

    Tries the instructed ``Answer: X`` format first, then falls back to a bare leading letter,
    which covers models that drop the label but still answer cleanly. Anything else is a
    genuine parse failure and is left as None.
    """
    match = ANSWER_RE.search(text) or FALLBACK_RE.search(text)
    answer = match.group(1).upper() if match else None

    conf_match = CONFIDENCE_RE.search(text)
    confidence = None
    if conf_match:
        value = int(conf_match.group(1))
        confidence = value if 0 <= value <= 100 else None

    return ParsedAnswer(answer=answer, confidence=confidence, raw=text)


def classify_stance(answer: str | None, correct: str, majority: str | None) -> Stance:
    """Label the naive agent's answer.

    ``majority`` is None in the control condition, where there is nothing to conform to -- a
    wrong answer there is baseline error, not conformity, which is exactly why every run carries
    its own n=0 cell.
    """
    if answer is None:
        return Stance.UNKNOWN
    if answer == correct:
        return Stance.REJECTED
    if majority is not None and answer == majority:
        return Stance.ADOPTED
    return Stance.IGNORED


def confederate_complied(text: str, assigned: str) -> bool:
    """Did the confederate actually assert the answer it was scripted to assert?

    Models sometimes break character and answer honestly. Those trials never applied the intended
    social pressure, so keeping them would dilute the effect toward zero. They are discarded and
    the discard rate is reported.
    """
    parsed = parse_answer(text)
    return parsed.answer == assigned


def parse_error_count(text: str) -> int | None:
    """Study 3: number of errors the reviewer claims to have found."""
    match = ERRORS_FOUND_RE.search(text)
    return int(match.group(1)) if match else None


def effort_proxy(text: str) -> int:
    """Whitespace token count of the response, used as the effort/vigilance proxy.

    Deliberately tokenizer-independent so the measure is comparable across model families.
    """
    return len(text.split())
