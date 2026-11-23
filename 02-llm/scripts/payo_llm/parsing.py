"""Parse Task A (emotion recognition) outputs, following the prompt's parsing rule."""

import json
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Recognition:
    """
    A parsed Task A output.

    `valid` refers to the emotion label only, because that is what RQ1 scores: an invalid emotion
    counts as incorrect. Concept and confidence problems are listed in `invalid_reason` and set
    their own field to None, without making the emotion invalid. `concept_alias` is True when the
    concept came from the schema's fixed alias map rather than an exact concept id.
    """

    emotion: str | None
    cultural_concept: str | None
    confidence: int | None
    valid: bool
    invalid_reason: str | None
    concept_alias: bool = False


def first_json_object(text: str) -> dict | None:
    """
    Return the first JSON object in a model output, ignoring any text or code fences around it.

    Args:
        text: Raw model output.

    Returns:
        The decoded object, or None if no `{...}` decodes to a JSON object.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _label(value: object, allowed: frozenset[str]) -> str | None:
    """Lowercase and trim a label, and return it only if it is an exact allowed id."""
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label if label in allowed else None


def _confidence(value: object) -> int | None:
    """Accept an integer 0-100 (also as a whole-number float or numeric string)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if isinstance(value, int) and 0 <= value <= 100:
        return value
    return None


def parse_recognition(text: str, emotion_ids: frozenset[str], concept_ids: frozenset[str],
                      concept_aliases: Mapping[str, str] | None = None) -> Recognition:
    """
    Parse a Task A output. Labels must match a schema id exactly after lowercasing and trimming;
    nothing is mapped to a nearest label. The one exception is `concept_aliases`, a fixed map
    (from the schema) of emotion ids that name exactly one concept, applied to the concept field only.

    Args:
        text: Raw model output.
        emotion_ids: Allowed emotion ids.
        concept_ids: Allowed cultural-concept ids (includes "none").
        concept_aliases: Exact-id map for the concept field, e.g. {"pakikisama_strain": "pakikisama"}.

    Returns:
        The parsed result with validity flags.
    """
    obj = first_json_object(text)
    if obj is None:
        return Recognition(None, None, None, valid=False, invalid_reason="no_json")

    problems: list[str] = []
    emotion = _label(obj.get("emotion"), emotion_ids)
    if emotion is None:
        problems.append("missing_emotion" if "emotion" not in obj else "unknown_emotion")
    concept = _label(obj.get("cultural_concept"), concept_ids)
    alias = _label(obj.get("cultural_concept"), frozenset(concept_aliases or ()))
    if concept is None and alias is not None:
        concept = concept_aliases[alias]
    if concept is None:
        problems.append("missing_concept" if "cultural_concept" not in obj else "unknown_concept")
    confidence = _confidence(obj.get("confidence"))
    if confidence is None:
        problems.append("missing_confidence" if "confidence" not in obj else "bad_confidence")

    return Recognition(
        emotion=emotion,
        cultural_concept=concept,
        confidence=confidence,
        valid=emotion is not None,
        invalid_reason=";".join(problems) or None,
        concept_alias=alias is not None and concept == concept_aliases[alias],
    )
