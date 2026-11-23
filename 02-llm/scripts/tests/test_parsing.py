"""Task A parsing: exact schema ids only, never coerced to a nearest label."""

import pytest

from payo_llm.parsing import parse_recognition

EMOTIONS = frozenset({"hiya", "tampo", "worry", "none_practical"})
CONCEPTS = frozenset({"utang_na_loob", "hiya", "none"})


def parse(text):
    return parse_recognition(text, EMOTIONS, CONCEPTS)


def test_clean_json():
    result = parse('{"emotion": "tampo", "cultural_concept": "none", "confidence": 80}')
    assert (result.emotion, result.cultural_concept, result.confidence, result.valid) == ("tampo", "none", 80, True)
    assert result.invalid_reason is None


def test_json_inside_prose_and_code_fence():
    result = parse('Sure!\n```json\n{"emotion": "Hiya ", "cultural_concept": "HIYA", "confidence": 55}\n```')
    assert result.emotion == "hiya" and result.cultural_concept == "hiya" and result.valid


def test_unknown_emotion_is_invalid_not_coerced():
    result = parse('{"emotion": "shame", "cultural_concept": "none", "confidence": 90}')
    assert result.emotion is None and not result.valid
    assert result.invalid_reason == "unknown_emotion"


def test_no_json():
    result = parse("The poster feels hiya.")
    assert not result.valid and result.invalid_reason == "no_json"


def test_bad_concept_and_confidence_keep_emotion_valid():
    result = parse('{"emotion": "worry", "cultural_concept": "bayanihan", "confidence": 150}')
    assert result.valid and result.emotion == "worry"
    assert result.cultural_concept is None and result.confidence is None
    assert result.invalid_reason == "unknown_concept;bad_confidence"


def test_missing_fields():
    result = parse('{"emotion": "worry"}')
    assert result.invalid_reason == "missing_concept;missing_confidence"


@pytest.mark.parametrize("raw, expected", [("70", 70), (70.0, 70), (True, None), (70.5, None), (-1, None)])
def test_confidence_forms(raw, expected):
    import json

    text = json.dumps({"emotion": "worry", "cultural_concept": "none", "confidence": raw})
    assert parse(text).confidence == expected


ALIASES = {"pakikisama_strain": "pakikisama", "utang_na_loob_guilt": "utang_na_loob"}


def test_concept_alias_maps_emotion_id_in_concept_field_only():
    concepts = CONCEPTS | {"pakikisama"}
    result = parse_recognition('{"emotion": "hiya", "cultural_concept": " Pakikisama_Strain", "confidence": 90}',
                               EMOTIONS, concepts, ALIASES)
    assert result.cultural_concept == "pakikisama" and result.concept_alias and result.invalid_reason is None
    exact = parse_recognition('{"emotion": "hiya", "cultural_concept": "hiya", "confidence": 90}',
                              EMOTIONS, concepts, ALIASES)
    assert exact.cultural_concept == "hiya" and not exact.concept_alias
    emotion_field = parse_recognition('{"emotion": "pakikisama", "cultural_concept": "none", "confidence": 90}',
                                      EMOTIONS, concepts, ALIASES)
    assert not emotion_field.valid  # aliases never touch the emotion field


def test_no_alias_map_keeps_strict_rule():
    result = parse('{"emotion": "hiya", "cultural_concept": "pakikisama_strain", "confidence": 90}')
    assert result.cultural_concept is None and result.invalid_reason == "unknown_concept"
    assert not result.concept_alias


def test_skips_a_broken_brace_before_the_real_object():
    result = parse('{oops} then {"emotion": "hiya", "cultural_concept": "none", "confidence": 10}')
    assert result.valid and result.emotion == "hiya"
