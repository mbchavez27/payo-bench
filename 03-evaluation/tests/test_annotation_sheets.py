"""Rebuilding the annotation sheets keeps the sample, the posts and their order, and uses the current schema."""

import os

import numpy as np
import pytest
from openpyxl import load_workbook

from payo_llm.schema import load_schema
from payo_eval import steps
from payo_eval.paths import get_paths, load_config
from payo_eval.sheets import read_sheet
from payo_eval.synthetic import SEED, add_keyword_cues, fake_silver, fill_sheet, synthetic_config, truth_table


@pytest.fixture
def sampled(tmp_path):
    before = dict(os.environ)
    env = {"PAYO_GOLD_DIR": tmp_path / "gold", "PAYO_WORK_DIR": tmp_path / "work",
           "PAYO_SILVER_PATH": tmp_path / "silver.parquet", "PAYO_RESULTS_DIR": tmp_path / "results",
           "PAYO_FIGURES_DIR": tmp_path / "figures", "PAYO_EVAL_CONFIG": tmp_path / "config.toml",
           "PAYO_PILOT_DIR": tmp_path / "pilot", "PAYO_PILOT_POSTS": tmp_path / "pilot" / "pilot_posts.csv"}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        rng = np.random.default_rng(SEED)
        silver = fake_silver(700, rng)
        synthetic_config(env["PAYO_EVAL_CONFIG"])
        cfg = load_config()
        silver = add_keyword_cues(silver, truth_table(silver["post_id"].tolist(), rng), cfg["prescreen"]["keywords"], rng)
        silver.to_parquet(env["PAYO_SILVER_PATH"], index=False)
        (tmp_path / "pilot").mkdir()
        silver[["post_id"]].iloc[:20].to_csv(env["PAYO_PILOT_POSTS"], index=False)
        steps.sample()
        yield get_paths(), cfg["roster"]["annotators"]
    finally:
        os.environ.clear()
        os.environ.update(before)


def _emotion_dropdown(path):
    lists = load_workbook(path)["lists"]
    values = {cell.value for row in lists.iter_rows() for cell in row if cell.value}
    return values


def test_rebuild_keeps_posts_and_order_and_uses_schema(sampled):
    paths, people = sampled
    before = {p: read_sheet(paths.sheet("annotation", p))["post_id"].tolist() for p in people}
    sample_bytes, assignment_bytes = paths.sample.read_bytes(), paths.assignment.read_bytes()

    steps.annotation_sheets(force=True)

    for person in people:
        assert read_sheet(paths.sheet("annotation", person))["post_id"].tolist() == before[person]
    assert paths.sample.read_bytes() == sample_bytes and paths.assignment.read_bytes() == assignment_bytes
    values = _emotion_dropdown(paths.sheet("annotation", people[0]))
    assert load_schema().emotion_ids <= values
    assert "konsensya" in values and not {"utang_na_loob_guilt", "pakikisama_strain"} & values


def test_rebuild_refuses_when_a_sheet_has_labels(sampled):
    paths, people = sampled
    fill_sheet(paths.sheet("annotation", people[0]), lambda row: {"primary_emotion": "worry"})
    with pytest.raises(SystemExit, match=people[0]):
        steps.annotation_sheets(force=True)


def test_sheets_open_on_a_guide_that_covers_every_allowed_value(sampled):
    paths, people = sampled
    book = load_workbook(paths.sheet("annotation", people[0]))
    assert book.sheetnames[0] == "how_to_annotate" and book.active.title == "how_to_annotate"
    guide = book["how_to_annotate"]
    rows = {row[0]: row for row in guide.iter_rows(values_only=True) if row and row[0]}
    schema = load_schema()
    for key in ("emotions", "cultural_concepts", "exclusion_reasons", "expression", "domain", "need_type"):
        for item in schema.raw[key]:
            assert item["id"] in rows, item["id"]
    for emotion in schema.emotion_ids:  # every emotion has its codebook cue
        assert rows[emotion][3], emotion
    assert read_sheet(paths.sheet("annotation", people[0]))["post_id"].tolist()  # labels tab still read back


def test_worked_examples_use_only_allowed_values_and_follow_the_skip_rule():
    from payo_eval.instruments import WORKED_EXAMPLES, annotation_columns

    columns = {c.name: c for c in annotation_columns(load_schema())}
    for _, labels, why in WORKED_EXAMPLES:
        assert why
        for field, value in labels.items():
            assert field in columns
            if value and columns[field].choices:
                assert value in columns[field].choices, (field, value)
        if labels["exclude_flag"] == "yes":
            assert set(labels) == {"exclude_flag", "exclude_reason"}
        else:
            assert {"primary_emotion", "cultural_concept", "expression", "domain", "need_type"} <= set(labels)
