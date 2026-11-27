"""
Fixed, pre-registered cleanup of every reply shown to raters (analysis plan v1.8; rules 2-5, 7 and 9 added in v1.9).

The 4B models ignore length and opener instructions (pilot prompts v2-v4), so the reply text is
normalized in code, the same way for every source (Gemma, SEA-LION and the community reply), so that the
cleanup itself cannot reveal the source:

1. Remove emoji.
2. Remove markdown emphasis markers ("*big*", "**big**", "__big__" -> "big"); single underscores are kept,
   so usernames like u/some_name_here are not touched.
3. Replace a spaced en/em dash (" – ", " — ") with ", " and an unspaced one ("2–3") with "-".
4. Straighten curly quotes (‘ ’ -> ', “ ” -> ").
5. Collapse repeated spaces, drop spaces at line ends, and keep at most one blank line between paragraphs.
6. Remove leading interjections followed by punctuation ("Uy,", "Naku!", "Okay, so", ...), then capitalize.
7. Remove a sentence-initial "Honestly," and capitalize what follows.
8. If the reply was cut off by max_new_tokens, drop the unfinished last sentence.
9. Drop closing sentences that are only a stock sign-off ("Good luck!", "Good luck with your shopping!",
   "Kaya mo 'yan!", "Laban lang!"), as long as other text remains.
10. If it is longer than MAX_WORDS, keep whole sentences up to MAX_WORDS. If the first sentence alone is
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
# Markdown emphasis: **x**, *x* or __x__ around non-space text on one line
MARKDOWN = re.compile(r"(\*\*|\*|__)(?=\S)([^*\n]+?)(?<=\S)\1")
SPACED_DASH = re.compile(r"\s+[\u2013\u2014]\s+")
DASH = re.compile("[\u2013\u2014]")
QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
HONESTLY = re.compile(r"(?:^|(?<=[.!?…] ))honestly,\s+(\w)", re.IGNORECASE | re.MULTILINE)
# A whole closing sentence that is only a stock sign-off, with an optional short tail without a comma
# ("with your shopping") and an optional address ("OP", "po"); anything longer is content and is kept.
SIGNOFF = re.compile(
    r"(?:good ?luck|best of luck|kaya mo (?:yan|'yan|ito|'to|to)|kayang-kaya mo (?:yan|'yan)|laban lang|laban|"
    r"you got this|you can do this|fighting|take care|ingat)(?: [^.!?…,\n]{1,40})?(?:,? (?:op|po|bes|sis|beh))?"
    r"[.!?…]*", re.IGNORECASE)
SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")


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
        (emoji, markdown, dash, quotes, spacing, opener, honestly, unfinished, signoff, trimmed, hard_cut).
    """
    ops = []
    out = text.strip()
    stripped = EMOJI.sub("", out)
    if stripped != out:
        ops.append("emoji")
        out = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    if (plain := MARKDOWN.sub(r"\2", out)) != out:
        out, _ = plain, ops.append("markdown")
    if DASH.search(out):
        out = DASH.sub("-", SPACED_DASH.sub(", ", out))
        ops.append("dash")
    if (straight := out.translate(QUOTES)) != out:
        out, _ = straight, ops.append("quotes")
    spaced = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", out)))
    if spaced != out:
        out, _ = spaced, ops.append("spacing")
    opener = False
    while (match := INTERJECTION.match(out)) and match.end() < len(out):
        out, opener = out[match.end():], True
    if opener:
        ops.append("opener")
        out = out[:1].upper() + out[1:]
    if (frank := HONESTLY.sub(lambda m: m.group(1).upper(), out)) != out:
        out, _ = frank, ops.append("honestly")
    if truncated and (ends := _sentence_ends(out)) and ends[-1] < len(out.rstrip()):
        out, _ = out[:ends[-1]], ops.append("unfinished")
    signoff = False
    while len(ends := _sentence_ends(out.rstrip())) >= 2 and SIGNOFF.fullmatch(out[ends[-2]:].strip()):
        out, signoff = out[:ends[-2]], True
    if signoff:
        ops.append("signoff")
    if len(out.split()) > max_words:
        fitting = [end for end in _sentence_ends(out) if len(out[:end].split()) <= max_words]
        if fitting:
            out, _ = out[:fitting[-1]], ops.append("trimmed")
        else:
            out, _ = " ".join(out.split()[:max_words]), ops.append("hard_cut")
    out = out.strip()
    return Cleaned(out, len(out.split()), tuple(ops))
