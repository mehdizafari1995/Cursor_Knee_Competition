"""Conservative language / script cues for radiology-report audits.

This is a heuristic inventory, not a language-id system. It never claims a
document's language with certainty; it records which scripts and function-word
cues were observed.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

FUNCTION_WORD_CUES: dict[str, tuple[str, ...]] = {
    "en": ("the", "and", "with", "without", "there", "is", "of"),
    "es": ("los", "las", "del", "una", "con", "sin", "hallazgos"),
    "fr": ("les", "des", "une", "avec", "sans", "dans"),
    "de": ("und", "der", "die", "das", "ohne", "mit"),
    "it": ("della", "delle", "una", "con", "senza"),
    "pt": ("uma", "com", "sem", "dos", "das"),
    "nl": ("een", "van", "niet", "met", "zonder"),
}

SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("latin", 0x0000, 0x024F),
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("arabic", 0x0600, 0x06FF),
    ("hebrew", 0x0590, 0x05FF),
    ("cjk", 0x4E00, 0x9FFF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("thai", 0x0E00, 0x0E7F),
)


def scripts_in_text(text: str) -> list[str]:
    found: set[str] = set()
    for char in text:
        code = ord(char)
        if char.isspace() or char.isdigit() or char in ".,;:!?-_/()[]{}\"'":
            continue
        for name, start, end in SCRIPT_RANGES:
            if start <= code <= end:
                found.add(name)
                break
        else:
            if code > 127:
                found.add("other_non_ascii")
    return sorted(found)


def function_word_cues(text: str) -> dict[str, int]:
    words = [match.group(0).lower() for match in _WORD.finditer(text)]
    counts: dict[str, int] = {}
    for lang, cues in FUNCTION_WORD_CUES.items():
        n = sum(1 for word in words if word in cues)
        if n:
            counts[lang] = n
    return counts


def summarise_reports(reports: list[str]) -> dict:
    """Aggregate heuristic cues. Empty / missing reports are counted separately."""
    n = len(reports)
    empty = 0
    non_ascii = 0
    script_counter: Counter[str] = Counter()
    cue_counter: Counter[str] = Counter()
    for raw in reports:
        text = "" if raw is None or (isinstance(raw, float)) else str(raw)
        if not text.strip():
            empty += 1
            continue
        if any(ord(ch) > 127 for ch in text):
            non_ascii += 1
        for script in scripts_in_text(text):
            script_counter[script] += 1
        for lang, count in function_word_cues(text).items():
            cue_counter[lang] += count
    return {
        "n_reports": n,
        "n_empty": empty,
        "n_non_empty": n - empty,
        "n_with_non_ascii": non_ascii,
        "scripts_observed": dict(script_counter),
        "function_word_cues": dict(cue_counter),
        "note": (
            "Language cues are heuristic (Unicode script + function words). "
            "They are not a verified language identification of each report."
        ),
    }
