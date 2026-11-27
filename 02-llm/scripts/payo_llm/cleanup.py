"""
Fixed, pre-registered cleanup of every reply shown to raters (analysis plan v1.8).

The 4B models ignore length and opener instructions (pilot prompts v2-v4), so the reply text is
normalized in code, the same way for every source (Gemma, SEA-LION and the community reply), so that the
cleanup itself cannot reveal the source:

1. Remove emoji.
2. Remove leading interjections followed by punctuation ("Uy,", "Naku!", "Okay, so", ...), then capitalize.
3. If the reply was cut off by max_new_tokens, drop the unfinished last sentence.
4. If it is longer than MAX_WORDS, keep whole sentences up to MAX_WORDS. If the first sentence alone is
   longer, cut at MAX_WORDS words.

The raw model output is never changed: it stays in the run JSONL files. Only rater sheets use this text.
"""

import re
from dataclasses import dataclass

MAX_WORDS = 110
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
# Interjections that open a reply: removed only when followed by punctuation (or "so"), so a content word
# like "Grabe yung pinagdadaanan mo" is kept.
INTERJECTION = re.compile(
    r"^\s*(?:(?:okay|ok)[,]?\s+so[,]?\s+|"
    r"(?:uy|huy|naku|hala|ay|hay|haay|hays|oh|wow|okay|ok|sige|grabe|bes|beh|sis)\s*[,!.…]+\s*)",
    re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?…]+[\"”’)\]]*(?=\s|$)")


@dataclass(frozen=True)
class Cleaned:
    """A cleaned reply and what was done to it."""

    text: str
    words: int
    ops: tuple[str, ...]


def _sentence_ends(text: str) -> list[int]:
    """Character offsets just after each sentence end."""
    return [m.end() for m in SENTENCE_END.finditer(text)]


def clean_reply(text: str, truncated: bool = False, max_words: int = MAX_WORDS) -> Cleaned:
    """
    Apply the fixed cleanup rules to one reply.

    Args:
        text: The reply as generated (or the community reply after rubric rule 5).
        truncated: True when generation hit max_new_tokens.
        max_words: Word cap.

    Returns:
        The cleaned text, its word count and the operations applied
        (emoji, opener, unfinished, trimmed, hard_cut).
    """
    ops = []
    out = text.strip()
    stripped = EMOJI.sub("", out)
    if stripped != out:
        ops.append("emoji")
        out = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    opener = False
    while (match := INTERJECTION.match(out)) and match.end() < len(out):
        out, opener = out[match.end():], True
    if opener:
        ops.append("opener")
        out = out[:1].upper() + out[1:]
    if truncated and (ends := _sentence_ends(out)) and ends[-1] < len(out.rstrip()):
        out, _ = out[:ends[-1]], ops.append("unfinished")
    if len(out.split()) > max_words:
        fitting = [end for end in _sentence_ends(out) if len(out[:end].split()) <= max_words]
        if fitting:
            out, _ = out[:fitting[-1]], ops.append("trimmed")
        else:
            out, _ = " ".join(out.split()[:max_words]), ops.append("hard_cut")
    out = out.strip()
    return Cleaned(out, len(out.split()), tuple(ops))
