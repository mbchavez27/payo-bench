"""Prompt rendering against the real prompt files and label schema."""

import difflib

import pytest

from payo_llm import ROOT
from payo_llm.data import GoldLabel, Post
from payo_llm.prompts import format_post, load_prompt, render
from payo_llm.schema import load_schema
from run import PROMPT_FILES, build_jobs

PROMPT_DIR = ROOT / "02-llm" / "prompts"
POST = Post("abc123", "Paano ko sasabihin sa magulang ko?", "Ako ang breadwinner pero pagod na ako.", None)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.mark.parametrize("task", list(PROMPT_FILES))
def test_every_prompt_renders_without_leftover_placeholders(task, schema):
    template = load_prompt(PROMPT_DIR / PROMPT_FILES[task])
    gold = {POST.post_id: GoldLabel("utang_na_loob_guilt", "family_obligation")}
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
