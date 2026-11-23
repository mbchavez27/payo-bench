"""Sheets round-trip and validation; annotation collection, adjudication, merges and the κ gate."""

import pandas as pd
import pytest

from payo_llm.schema import load_schema
from payo_eval.annotation import adjudicate, collect, disagreements, finalize, resolve_merges
from payo_eval.instruments import annotation_columns
from payo_eval.sheets import Column, read_sheet, validate, write_sheet

SCHEMA = load_schema()
FIELDS = ["exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept", "expression", "domain",
          "need_type", "notes"]


def test_write_read_roundtrip_and_no_overwrite(tmp_path):
    frame = pd.DataFrame({"post_id": ["a", "b"], "post": ["long text " * 20, "short"]})
    path = write_sheet(tmp_path / "x.xlsx", frame, [Column("score", int_range=(1, 5)), Column("label", ("x", "y"))],
                       ["post"], prefill=pd.DataFrame({"label": ["x", ""]}))
    back = read_sheet(path)
    assert back["post_id"].tolist() == ["a", "b"] and back["label"].tolist() == ["x", ""]
    with pytest.raises(SystemExit):
        write_sheet(path, frame, [])


def test_validate_choices_ranges_multi_and_skip():
    columns = [Column("flag", ("yes", "no")), Column("score", int_range=(1, 5), skip_if={"flag": "yes"}),
               Column("tags", ("none", "a", "b"), multi=True), Column("best", ("A", "B"))]
    frame = pd.DataFrame([
        {"id": "1", "flag": "no", "score": "6", "tags": "a;c", "best": "b"},
        {"id": "2", "flag": "yes", "score": "", "tags": "none", "best": "A"},
        {"id": "3", "flag": "no", "score": "", "tags": "a", "best": "A"},
    ])
    problems = validate(frame, columns, "id", "rater")
    assert any("id 1, score" in p for p in problems)
    assert any("id 1, tags" in p and "'c'" in p for p in problems)
    assert not any("id 1, best" in p for p in problems)   # case-insensitive
    assert not any("id 2" in p for p in problems)         # skipped when excluded
    assert any("id 3, score: empty" in p for p in problems)


def sheet(rows):
    return pd.DataFrame(rows, columns=["post_id", *FIELDS]).fillna("")


def label(pid, emotion, concept="none", expression="explicit", exclude="no", reason=""):
    return [pid, exclude, reason, emotion, concept, expression, "family", "both", ""]


def test_collect_disagreements_and_adjudicate():
    a = sheet([label("p1", "hiya"), label("p2", "worry"), label("p3", "", exclude="yes", reason="bot_or_spam")])
    b = sheet([label("p1", "tampo"), label("p2", "worry")])
    raw = collect({"ann_a": a, "ann_b": b}, SCHEMA)
    table = disagreements(raw)
    assert table["post_id"].tolist() == ["p1"]
    assert table.loc[0, "agreed_primary_emotion"] == "" and table.loc[0, "agreed_domain"] == "family"
    decisions = pd.DataFrame([{"post_id": "p1", **dict(zip(FIELDS[:-1], label("p1", "tampo")[1:-1], strict=True))}])
    labels = adjudicate(raw, decisions).set_index("post_id")
    assert labels.loc["p1", "primary_emotion"] == "tampo" and labels.loc["p1", "label_source"] == "adjudicated"
    assert labels.loc["p2", "label_source"] == "agreed" and labels.loc["p3", "label_source"] == "single"
    with pytest.raises(SystemExit):
        adjudicate(raw, None)  # p1 still needs a decision


def test_collect_rejects_bad_values():
    bad = sheet([label("p1", "shame")])
    with pytest.raises(SystemExit, match="shame"):
        collect({"ann_a": bad}, SCHEMA)


def test_resolve_merges_threshold_chain_and_missing_target():
    labels = ["a", "b", "c", "d"]
    mapping = resolve_merges({"a": 5, "b": 8, "c": 40}, {"a": "b", "b": "c"}, 15, labels)
    assert mapping == {"a": "c", "b": "c"}
    assert resolve_merges({"a": 20, "b": 30}, {"a": "b"}, 15, labels) == {}
    with pytest.raises(SystemExit, match="no merge target"):
        resolve_merges({"a": 3}, {}, 15, labels)


def test_finalize_types_from_schema_and_gate():
    rows_a, rows_b = [], []
    for i in range(40):
        emotion = "hiya" if i < 20 else "worry"
        rows_a.append(label(f"p{i}", emotion))
        rows_b.append(label(f"p{i}", "worry" if i < 20 else "hiya"))  # complete disagreement
    raw = collect({"ann_a": sheet(rows_a), "ann_b": sheet(rows_b)}, SCHEMA)
    decisions = pd.DataFrame([{"post_id": f"p{i}", **dict(zip(FIELDS[:-1], label(f"p{i}", "hiya" if i < 20 else "worry")[1:-1],
                                                             strict=True))} for i in range(40)])
    result = finalize(raw, decisions, SCHEMA, {}, 15, 0.40)
    types = result.labels.set_index("post_id")["emotion_type"]
    assert types["p0"] == "culture_bound" and types["p30"] == "shared"
    assert any(f.startswith("hiya") for f in result.gate_failures)
