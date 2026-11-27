"""Pilot report: health summaries and flags on synthetic records, without any model."""

import json
import math

import pandas as pd
import pytest

from payo_llm.pilot import (advice_health, annotation_summary, completeness, flags, load_outputs,
                            recognition_health, render_table, style_flags, style_health, timing,
                            wilson_ci)
from payo_llm.schema import Schema

SCHEMA = Schema({
    "version": "test",
    "reply_language": "Taglish",
    "emotions": [
        {"id": "hiya", "type": "culture_bound", "definition": "d"},
        {"id": "worry", "type": "shared", "definition": "d"},
        {"id": "none_practical", "type": "none", "definition": "d"},
    ],
    "cultural_concepts": [{"id": "hiya", "definition": "d"}, {"id": "none", "definition": "d"}],
})
SHEET_COLUMNS = ["post_id", "annotator", "exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept",
                 "expression", "domain", "need_type", "notes"]


def rec(post_id, model="gemma", emotion="worry", valid=True, confidence=70, error=None, **extra):
    return {"post_id": post_id, "model": model, "task": "recognition", "condition": "none", "seed": 42,
            "emotion": emotion if valid else None, "cultural_concept": "none", "confidence": confidence,
            "valid": valid, "invalid_reason": None if valid else "unknown_emotion", "error": error,
            "latency_s": 2.0, "n_prompt_tokens": 1000, "truncated": False, **extra}


def adv(post_id, words, model="gemma", condition="P0", truncated=False):
    task = "advice-p0" if condition == "P0" else "advice-oracle"
    return {"post_id": post_id, "model": model, "task": task, "condition": condition, "seed": 42,
            "reply_words": words, "truncated": truncated, "error": None, "latency_s": 10.0,
            "n_prompt_tokens": 1200}


def sheet(rows):
    frame = pd.DataFrame(rows, columns=SHEET_COLUMNS)
    return frame.fillna("")


def test_load_outputs_keeps_last_record_per_key(tmp_path):
    folder = tmp_path / "emotion-recognition"
    folder.mkdir()
    lines = [rec("a", error="timeout"), rec("a", emotion="hiya"), rec("b")]
    (folder / "predictions_gemma.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    (tmp_path / "variance").mkdir()
    (tmp_path / "variance" / "x.jsonl").write_text(json.dumps(rec("z")) + "\n")
    outputs = load_outputs(tmp_path)
    assert sorted(outputs["post_id"]) == ["a", "b"]
    assert outputs.set_index("post_id").loc["a", "emotion"] == "hiya"


def test_load_outputs_empty_folder(tmp_path):
    assert load_outputs(tmp_path).empty


def test_completeness_counts_missing_errors_and_not_run():
    outputs = pd.DataFrame([rec("a"), rec("b", error="prompt_too_long")])
    table = completeness(outputs, {"recognition": ["a", "b", "c"], "advice-p0": ["a"]}, models=("gemma",))
    row = table[table["task"] == "recognition"].iloc[0]
    assert (row.ok, row.errors, row.missing) == (1, 1, 2)
    assert "prompt_too_long: 1" in row.error_types
    assert table[table["task"] == "advice-p0"].iloc[0].error_types == "not run"


def test_recognition_health_and_collapse_flags():
    outputs = pd.DataFrame([rec(str(i), emotion="worry") for i in range(8)]
                           + [rec("8", valid=False), rec("9", valid=False)]
                           + [rec(str(i), model="sealion", emotion="hiya" if i % 2 else "worry", confidence=i * 10)
                              for i in range(10)])
    summary, labels, same = recognition_health(outputs)
    gemma = summary.set_index("model").loc["gemma"]
    assert gemma.valid_rate == pytest.approx(0.8)
    assert gemma.top_label == "worry" and gemma.top_share == 1.0
    assert "(invalid)" in set(labels["emotion"])
    assert same == pytest.approx(4 / 8)  # posts 0-7 valid for both; sealion says worry on even ids
    warnings = flags(pd.DataFrame(), summary, pd.DataFrame(), pd.DataFrame(),
                     annotation_summary(sheet([]), SCHEMA, n_silver=100), n_silver=100)
    text = " ".join(warnings)
    assert "gemma: 20% of Task A outputs are invalid" in text
    assert "possible label collapse" in text and "sealion: 'worry'" not in text
    assert "every confidence is 70" in text


def test_advice_health_lengths_and_flags():
    outputs = pd.DataFrame([adv("a", 150), adv("b", 200, truncated=True), adv("a", 170, condition="P-Oracle"),
                            adv("b", 230, condition="P-Oracle")])
    community = pd.Series({"a": 40, "b": 60, "zzz": 999})
    summary, diff = advice_health(outputs, community)
    p0 = summary.set_index("source").loc["gemma P0"]
    assert (p0["median"], p0["over_cap"], p0["truncated"]) == (175, 1.0, 0.5)
    assert summary.set_index("source").loc["community (same posts)", "max"] == 60
    assert diff.iloc[0].median_diff == 25
    text = " ".join(flags(pd.DataFrame(), pd.DataFrame(), summary, pd.DataFrame(),
                          annotation_summary(sheet([]), SCHEMA, n_silver=100), n_silver=100))
    assert "gemma P0: 50% of replies hit max_new_tokens" in text
    assert "gemma P0: 100% of replies exceed 110 words" in text
    assert "community" not in text


def test_timing_context_flag():
    times = timing(pd.DataFrame([rec("a", n_prompt_tokens=7900)]), num_ctx=8192)
    assert times.iloc[0].ctx_share > 0.9
    text = " ".join(flags(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), times,
                          annotation_summary(sheet([]), SCHEMA, n_silver=100), n_silver=100))
    assert "context window" in text


def test_annotation_retention_projection_and_schema_types():
    rows = [["p1", "A", "", "", "hiya", "hiya", "implicit", "family", "emotional", ""],
            ["p2", "A", "", "", "worry", "none", "explicit", "money", "practical", ""],
            ["p3", "A", "", "", "worry", "none", "explicit", "money", "both", ""],
            ["p4", "A", "yes", "minor", "", "", "", "", "", ""],
            ["p5", "A", "", "", "", "", "", "", "", ""]]  # blank: not labeled yet
    summary = annotation_summary(sheet(rows), SCHEMA, n_silver=40, n_target=300, min_label=15)
    assert (summary.n_labeled, summary.n_excluded) == (4, 1)
    assert summary.retention == pytest.approx(0.75)
    assert summary.posts_needed == 400
    table = summary.emotions.set_index("emotion")
    assert table.loc["hiya", "type"] == "culture_bound"
    assert table.loc["worry", "pool_estimate"] == pytest.approx(2 / 3 * 40 * 0.75)  # 20
    assert table.loc["hiya", "below_min"] and not table.loc["worry", "below_min"]
    assert table.loc["none_practical", "n"] == 0
    assert summary.concept_present == pytest.approx(1 / 3)
    text = " ".join(flags(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary, n_silver=40))
    assert "need 400 silver posts but only 40 exist" in text
    assert "merge candidates, analysis plan §2): hiya." in text
    assert "Not seen in the pilot labels" in text and "none_practical" in text


def test_annotation_unknown_labels_and_disagreements():
    rows = [["p1", "A", "", "", "hiya", "none", "implicit", "family", "emotional", ""],
            ["p1", "B", "", "", "shame", "none", "implicit", "family", "emotional", ""],
            ["p2", "A", "", "", "worry", "none", "explicit", "money", "practical", ""],
            ["p2", "B", "", "", "worry", "none", "explicit", "money", "practical", ""]]
    summary = annotation_summary(sheet(rows), SCHEMA, n_silver=1000)
    assert summary.unknown_labels == ["shame"]
    agreement = summary.agreement.set_index("field")
    assert agreement.loc["primary_emotion", "agree_share"] == 0.5
    assert agreement.loc["domain", "agree_share"] == 1.0
    row = summary.disagreements.iloc[0]
    assert (row.post_id, row.field, row["values"]) == ("p1", "primary_emotion", "A: hiya; B: shame")


def test_blank_sheet_is_reported_cleanly():
    summary = annotation_summary(sheet([["p1", "", "", "", "", "", "", "", "", ""]]), SCHEMA, n_silver=100)
    assert summary.n_labeled == 0 and summary.posts_needed is None and math.isnan(summary.retention)
    assert flags(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary, n_silver=100) == []


def test_wilson_and_render_table():
    low, high = wilson_ci(27, 30)
    assert 0.73 < low < 0.75 and 0.96 < high < 0.98
    assert render_table(pd.DataFrame()) == "_none_"
    table = render_table(pd.DataFrame({"a": [0.123, math.nan], "b": ["x|y", "z"]}))
    assert "| 0.12 | x\\|y |" in table and "|  | z |" in table


def test_advice_health_cap_and_grid_rows():
    grid = [{**adv("a", w), "task": "advice-oracle-grid", "condition": f"P-Oracle:e{i}|c"}
            for i, w in enumerate([80, 100, 120, 30])]
    outputs = pd.DataFrame([adv("a", 110), adv("b", 111), *grid])
    summary, diff = advice_health(outputs, pd.Series({"a": 40, "b": 60}))
    rows = summary.set_index("source")
    assert rows.loc["gemma P0", "over_cap"] == 0.5  # 110 is at the cap, 111 is over
    g = rows.loc["gemma P-Oracle grid"]
    assert (g["n"], g["over_cap"], g["under_half_target"]) == (4, 0.25, 0.25)
    assert diff.empty  # grid replies never enter the paired P-Oracle - P0 gap
    text = " ".join(flags(pd.DataFrame(), pd.DataFrame(), summary, pd.DataFrame(),
                          annotation_summary(sheet([]), SCHEMA, n_silver=100), n_silver=100))
    assert "gemma P-Oracle grid: 25% of replies exceed 110 words" in text


def test_style_health_flags_opener_emoji_and_persona():
    texts = ["Uy, naku, gets ko 😊", "Uy naku! As a girl, I get it.", "Uy, naku. We're here for you.",
             "Mahirap talaga yan."]
    outputs = pd.DataFrame([{**adv(str(i), 5), "raw_output": t} for i, t in enumerate(texts)])
    community = pd.Series({"0": "To each their own", "1": "Pass ako dyan", "2": "ok", "3": "k"})
    style = style_health(outputs, community).set_index("source")
    p0 = style.loc["gemma P0"]
    assert (p0["top_opener"], p0["top_opener_share"]) == ("uy naku", 0.75)
    assert p0["emoji_share"] == 0.25 and p0["persona_n"] == 2
    assert (p0["top_first_word"], p0["top_first_word_share"]) == ("uy", 0.75)
    assert style.loc["community (same posts)", "persona_n"] == 0
    text = " ".join(style_flags(style.reset_index()))
    assert "75% of replies open with 'uy naku'" in text
    assert "2 persona claims" in text and "emoji" in text


def test_style_flags_first_word_when_openers_vary():
    texts = ["Uy naku a", "Uy gets b", "Uy grabe c", "Hala d"]
    outputs = pd.DataFrame([{**adv(str(i), 5), "raw_output": t} for i, t in enumerate(texts)])
    text = " ".join(style_flags(style_health(outputs, pd.Series(dtype=str))))
    assert "75% of replies start with 'uy'" in text


def test_style_health_persona_variants_and_taglish_share():
    outputs = pd.DataFrame([{**adv("a", 5), "raw_output": "From a female perspective, ok."},
                            {**adv("b", 5), "raw_output": "Mahirap talaga yan pero kaya mo."}])
    style = style_health(outputs, pd.Series(dtype=str)).set_index("source").loc["gemma P0"]
    assert style["persona_n"] == 1
    assert style["tl_word_share"] == pytest.approx(4 / 11)  # talaga, yan, pero, mo out of 11 tokens


def test_style_health_flags_markdown_and_dashes_and_reports_diba():
    texts = ["Big *year* \u2013 di ba?", "Sobrang *hirap* \u2014 diba", "Kaya mo yan \u2013 promise", "Ok lang."]
    outputs = pd.DataFrame([{**adv(str(i), 5), "raw_output": t} for i, t in enumerate(texts)])
    community = pd.Series({str(i): "Pass ako dyan - seryoso" for i in range(4)})
    style = style_health(outputs, community).set_index("source")
    p0 = style.loc["gemma P0"]
    assert (p0["markdown_share"], p0["dash_share"], p0["diba_share"]) == (0.5, 0.75, 0.5)
    assert style.loc["community (same posts)", "dash_share"] == 0.0  # a plain hyphen is not a dash
    text = " ".join(style_flags(style.reset_index()))
    assert "50% of replies have markdown emphasis" in text and "75% of replies have an en/em dash" in text
    assert "di ba" not in text  # reported, never flagged


def test_style_health_flags_spacing_quotes_honestly_and_signoff():
    texts = ["Honestly, ok lang.  Good luck!", "Kaya mo \u2018yan.  Kaya mo 'yan!", "Ayos. Honestly, ayos.", "Ok."]
    outputs = pd.DataFrame([{**adv(str(i), 5), "raw_output": t} for i, t in enumerate(texts)])
    style = style_health(outputs, pd.Series({str(i): "ok lang" for i in range(4)})).set_index("source")
    p0 = style.loc["gemma P0"]
    assert (p0["double_space_share"], p0["curly_quote_share"]) == (0.5, 0.25)
    assert (p0["honestly_share"], p0["signoff_share"]) == (0.5, 0.5)
    text = " ".join(style_flags(style.reset_index()))
    assert "a double space" in text and "a stock sign-off" in text and "'Honestly,'" in text
    assert "curly quotes" in text


def test_persona_claims_with_curly_apostrophes_are_counted():
    outputs = pd.DataFrame([{**adv("a", 5), "raw_output": "From a girl’s perspective, ok. I’ve been there."}])
    assert style_health(outputs, pd.Series(dtype=str)).set_index("source").loc["gemma P0", "persona_n"] == 2
