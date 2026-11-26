"""Prompt rendering against the real prompt files and label schema."""

import difflib

import pytest

from payo_llm import ROOT
from payo_llm.data import GoldLabel, Post
from payo_llm.prompts import format_post, load_prompt, render
from payo_llm.schema import load_schema
from run import PROMPT_FILES, build_jobs, grid_condition, parse_args

PROMPT_DIR = ROOT / "02-llm" / "prompts"
POST = Post("abc123", "Paano ko sasabihin sa magulang ko?", "Ako ang breadwinner pero pagod na ako.", None)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.mark.parametrize("task", list(PROMPT_FILES))
def test_every_prompt_renders_without_leftover_placeholders(task, schema):
    template = load_prompt(PROMPT_DIR / PROMPT_FILES[task])
    gold = {POST.post_id: GoldLabel("konsensya", "family_obligation")}
    jobs, skipped = build_jobs(task, [POST], schema, template, gold)
    assert skipped == 0
    assert "{{" not in jobs[0].prompt
    assert POST.context in jobs[0].prompt


def test_oracle_is_p0_plus_only_the_background_block(schema):
    gold = {POST.post_id: GoldLabel("hiya", "none")}
    p0 = build_jobs("advice-p0", [POST], schema, load_prompt(PROMPT_DIR / PROMPT_FILES["advice-p0"]), gold)[0][0]
    oracle = build_jobs("advice-oracle", [POST], schema,
                        load_prompt(PROMPT_DIR / PROMPT_FILES["advice-oracle"]), gold)[0][0]
    changes = [line for line in difflib.ndiff(p0.prompt.splitlines(), oracle.prompt.splitlines())
               if line[:1] in "+-"]
    assert all(line.startswith("+") for line in changes), "Oracle must only add lines"
    added = "\n".join(changes)
    assert "Background from a trained reader" in added
    assert schema.emotion("hiya")["definition"] in added


def test_oracle_skips_posts_without_gold(schema):
    template = load_prompt(PROMPT_DIR / PROMPT_FILES["advice-oracle"])
    jobs, skipped = build_jobs("advice-oracle", [POST], schema, template, gold={})
    assert jobs == [] and skipped == 1


def test_recognition_prompt_lists_every_schema_id(schema):
    template = load_prompt(PROMPT_DIR / PROMPT_FILES["recognition"])
    prompt = build_jobs("recognition", [POST], schema, template, None)[0][0].prompt
    for label in schema.emotion_ids | schema.concept_ids:
        assert f"- {label}:" in prompt


def test_render_rejects_missing_values():
    template = load_prompt(PROMPT_DIR / PROMPT_FILES["advice-p0"])
    with pytest.raises(KeyError):
        render(template, {"post": "x"})


def test_format_post_omits_empty_previous_attempts():
    assert "Previous Attempts" not in format_post("goal", "ctx", None)
    assert "Previous Attempts" not in format_post("goal", "ctx", "  ")
    assert format_post("goal", "ctx", "tried").endswith("Previous Attempts: tried")


def test_oracle_grid_covers_every_label_pair_and_matches_the_oracle_prompt(schema):
    template = load_prompt(PROMPT_DIR / PROMPT_FILES["advice-oracle-grid"])
    jobs, skipped = build_jobs("advice-oracle-grid", [POST], schema, template, None)
    assert skipped == 0 and len(jobs) == len(schema.emotion_ids) * len(schema.concept_ids)
    assert len({j.condition for j in jobs}) == len(jobs)
    assert {j.oracle_label for j in jobs} == {(e, c) for e in schema.emotion_ids for c in schema.concept_ids}
    # The grid cell for a label pair is the exact prompt advice-oracle renders for that gold label
    cell = next(j for j in jobs if j.oracle_label == ("tampo", "hiya"))
    oracle = build_jobs("advice-oracle", [POST], schema, template, {POST.post_id: GoldLabel("tampo", "hiya")})[0][0]
    assert cell.prompt == oracle.prompt and cell.condition == grid_condition("tampo", "hiya")


def test_grid_task_refuses_variance(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py", "--profile", "full", "--task", "advice-oracle-grid", "--variance"])
    with pytest.raises(SystemExit):
        parse_args()
