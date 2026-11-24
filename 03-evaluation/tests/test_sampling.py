"""Seeded draws: allocation, stratified sample, assignment with overlap and avoid rules, rated set."""

import pandas as pd
import pytest

from payo_eval.paths import load_config
from payo_eval.sampling import (allocate, assign, draw_prescreen_pool, keyword_prescreen, n_to_draw, rated_set,
                                stratified_sample)


def test_allocate_floor_proportional_and_caps():
    take = allocate({"a": 100, "b": 50, "c": 3}, 60, floor=10)
    assert sum(take.values()) == 60
    assert take["c"] == 3                     # capped at what it has
    assert take["a"] > take["b"] >= 10        # proportional above the floor


def test_allocate_never_exceeds_available():
    assert allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}


def test_n_to_draw():
    assert n_to_draw(300, 0.9) == 334


def test_prescreen_pool_excludes_pilot_and_is_deterministic():
    ids = [f"p{i}" for i in range(100)]
    first = draw_prescreen_pool(ids, {"p1", "p2"}, 50, seed=42)
    assert len(first) == 50 and not {"p1", "p2"} & set(first)
    assert first == draw_prescreen_pool(ids, {"p1", "p2"}, 50, seed=42)


def prescreen(n=400):
    rows = []
    for i in range(n):
        rows.append({"post_id": f"p{i:03d}", "exclude_flag": "yes" if i % 20 == 0 else "no",
                     "concept_likely": "yes" if i % 3 else "no",
                     "domain_guess": ["family", "money", "romantic", "other"][i % 4]})
    return pd.DataFrame(rows)


def test_stratified_sample_share_floor_and_exclusions():
    sample = stratified_sample(prescreen(), 120, concept_share=0.67, domain_floor=5, seed=1)
    assert len(sample) == 120 and sample["post_id"].is_unique
    excluded = set(prescreen().query("exclude_flag == 'yes'")["post_id"])
    assert not excluded & set(sample["post_id"])
    assert (sample["arm"] == "culture").sum() == round(120 * 0.67)
    assert sample.groupby(["arm", "domain_guess"]).size().min() >= 5
    assert sample.equals(stratified_sample(prescreen(), 120, 0.67, 5, seed=1))


def test_assign_overlap_balance_and_distinct_people():
    posts = [f"p{i}" for i in range(100)]
    plan = assign(posts, ["a", "b", "c"], 0.25, seed=3)
    assert (plan["slot"] == 2).sum() == 25
    doubled = plan.groupby("post_id")["person"].agg(list)
    assert all(len(set(p)) == len(p) for p in doubled)
    loads = plan["person"].value_counts()
    assert loads.max() - loads.min() <= 1


def test_assign_respects_avoid_when_possible():
    posts = [f"p{i}" for i in range(30)]
    avoid = {p: {"a"} for p in posts}
    plan = assign(posts, ["a", "b", "c"], 0.0, seed=0, avoid=avoid)
    assert "a" not in set(plan["person"]) and not plan["conflict"].any()
    forced = assign(posts, ["a"], 0.0, seed=0, avoid=avoid)
    assert forced["conflict"].all()


def test_fixed_adjudicator_never_on_double_posts():
    posts = [f"p{i}" for i in range(100)]
    plan = assign(posts, ["a", "b", "c", "adj"], 0.25, seed=4, not_on_doubles={"adj"})
    doubles = set(plan.loc[plan["slot"] == 2, "post_id"])
    assert not plan[plan["post_id"].isin(doubles) & (plan["person"] == "adj")].shape[0]
    assert (plan["person"] == "adj").sum() > 0  # still annotates single-coded posts
    assert (plan["slot"] == 2).sum() == 25


def test_assign_needs_two_people_for_overlap():
    with pytest.raises(SystemExit):
        assign(["p1"], ["a"], 0.5, seed=0)


def test_rated_set_size_and_strata():
    labels = pd.DataFrame({"post_id": [f"p{i}" for i in range(200)],
                           "emotion_type": ["culture_bound", "shared", "none", "shared"] * 50,
                           "domain": ["family", "money"] * 100})
    rated = rated_set(labels, 80, seed=2)
    assert len(rated) == 80 and rated["post_id"].is_unique
    share = rated["emotion_type"].value_counts(normalize=True)
    assert share["shared"] == pytest.approx(0.5, abs=0.02)


KEYWORDS = {"concept": ["hiya", "nahihiya", "utang na loob"], "exclude": ["suicidal", "self-harm"],
            "domain": {"family": ["mama", "kapatid"], "romantic": ["jowa"], "friendship": ["barkada"],
                       "work_school": ["boss"], "money": ["utang", "pera"]}}


def silver_posts(texts):
    return pd.DataFrame({"post_id": [f"p{i}" for i in range(len(texts))], "problem_goal": texts,
                         "context": [""] * len(texts), "previous_attempts": [None] * len(texts)})


def test_keyword_prescreen_whole_words_and_phrases():
    out = keyword_prescreen(silver_posts(["Nahihiya ako sa mama ko", "hiyang sa gamot", "malaki ang Utang\nna loob ko",
                                          "may utang siya"]), set(), KEYWORDS).set_index("post_id")
    assert out.loc["p0", "concept_likely"] == "yes" and out.loc["p0", "notes"] == "nahihiya"
    assert out.loc["p1", "concept_likely"] == "no"                  # hiyang is a different word
    assert out.loc["p2", "concept_likely"] == "yes"                 # phrase across a line break
    assert out.loc["p3", "concept_likely"] == "no"                  # bare utang is money, not a concept


def test_keyword_prescreen_domain_rule_follows_codebook():
    out = keyword_prescreen(silver_posts(["utang ng kapatid ko, pera", "jowa jowa at boss", "jowa at boss",
                                          "pera lang", "wala"]), set(), KEYWORDS).set_index("post_id")
    assert out["domain_guess"].tolist() == ["family", "romantic", "romantic", "money", "other"]


def test_keyword_prescreen_crisis_exclusion_and_pilot_ids():
    out = keyword_prescreen(silver_posts(["I feel suicidal", "self-harm noon", "okay lang"]), {"p2"}, KEYWORDS)
    assert out["post_id"].tolist() == ["p0", "p1"]
    assert (out["exclude_flag"] == "yes").all() and (out["exclude_reason"] == "self_harm_crisis").all()
    assert out.equals(keyword_prescreen(silver_posts(["I feel suicidal", "self-harm noon", "okay lang"]), {"p2"}, KEYWORDS))


def test_config_keyword_lists_are_valid():
    keywords = load_config()["prescreen"]["keywords"]
    assert keywords["concept"] and keywords["exclude"]
    assert set(keywords["domain"]) == {"family", "romantic", "friendship", "work_school", "money"}
