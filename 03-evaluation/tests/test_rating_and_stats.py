"""Blinded items, rating collection, and every statistic against hand-computed values."""

import math

import numpy as np
import pandas as pd
import pytest

from payo_llm.schema import load_schema
from payo_eval.agreement import cohen_kappa, one_vs_rest_kappa, ordinal_alpha, pairs
from payo_eval.analyses import interpret_oracle
from payo_eval.metrics import accuracy, bootstrap_ci, ece, macro_f1
from payo_eval.models import contrast, holm, mcnemar, tost_paired, wilcoxon
from payo_eval.rating import bestfit_key, build_items, clean_community, collect_ratings

SCHEMA = load_schema()


def test_clean_community_drops_quotes_and_edit_trailer():
    text = "> quoting OP\nGood advice here.\n\nEdit: thanks for the gold!\nmore"
    assert clean_community(text) == "Good advice here."
    assert clean_community("> only a quote") == "> only a quote"  # never empty


def items():
    posts = pd.DataFrame({"post_id": ["p1", "p2"]})
    community = pd.DataFrame({"post_id": ["p1", "p2"], "reply_body": ["hi there", "yo"], "reply_score": [5, 9]})
    replies = {(m, c): pd.DataFrame({"post_id": ["p1", "p2"], "raw_output": [f"{m} {c} one", f"{m} {c} two"],
                                     "reply_words": [3, 3]})
               for m in ("gemma", "sealion") for c in ("P0", "P-Oracle")}
    return build_items(posts, community, replies, seed=1)


def test_build_items_blinds_and_shuffles():
    key, texts = items()
    assert len(key) == 10 and key["item_id"].is_unique
    assert set(key.groupby("post_id").size()) == {5}
    assert "reply" not in key.columns and set(texts.columns) == {"item_id", "reply"}
    assert not texts["reply"].str.contains("item_id").any()
    letters = bestfit_key(key, seed=1)
    p0 = key[key["condition"].isin(["none", "P0"])]
    assert set(letters["item_id"]) == set(p0["item_id"]) and len(letters) == 6


def test_build_items_requires_every_llm_reply():
    posts = pd.DataFrame({"post_id": ["p1"]})
    community = pd.DataFrame({"post_id": ["p1"], "reply_body": ["x"], "reply_score": [1]})
    with pytest.raises(SystemExit, match="missing"):
        build_items(posts, community, {}, seed=1)


def test_collect_ratings_validates_and_joins():
    key, _ = items()
    letters = bestfit_key(key, seed=1)
    plan = pd.DataFrame({"post_id": ["p1", "p2"], "person": ["r1", "r1"], "slot": [1, 1]})
    good = {"empathy": "4", "cultural_fit": "3", "helpfulness": "5", "register_tone": "4",
            "safe_escalation": "not_needed", "harm_flags": "none", "strategy": "autonomy", "notes": ""}
    sheet = pd.DataFrame([{"item_id": i, **good} for i in key["item_id"]])
    best = pd.DataFrame({"post_ref": sorted(key["post_ref"].unique()), "best": ["a", "B"], "notes": ["", ""]})
    ratings, choices = collect_ratings({"r1": sheet}, {"r1": best}, SCHEMA, plan, key, letters)
    assert len(ratings) == 10 and ratings["empathy"].dtype.kind == "i" and set(ratings["slot"]) == {1}
    assert set(choices["best_item_id"]) <= set(letters["item_id"])
    bad = sheet.copy()
    bad.loc[0, "harm_flags"] = "none;illegal"
    bad.loc[1, "empathy"] = "7"
    with pytest.raises(SystemExit) as error:
        collect_ratings({"r1": bad}, {"r1": best}, SCHEMA, plan, key, letters)
    assert "none' combined" in str(error.value) and "empathy" in str(error.value)


def test_contrast_matches_hand_computation():
    params = pd.Series({"a": 1.0, "b": 2.0})
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "b"], columns=["a", "b"])
    c = contrast(params, cov, {"a": 1, "b": -1}, "a-b")
    assert c.estimate == pytest.approx(-1.0)
    assert c.se == pytest.approx(math.sqrt(0.04 + 0.09 - 2 * 0.01))
    assert c.p_less < 0.05 < c.p_greater


def test_mcnemar_counts_and_or():
    result = mcnemar([1, 1, 0, 0, 1], [1, 0, 1, 1, 0])
    assert (result["a_only"], result["b_only"]) == (2, 2) and result["odds_ratio"] == 1.0


def test_wilcoxon_rank_biserial():
    out = wilcoxon([1, 2, 3, 0, -1])
    assert out["n_nonzero"] == 4
    # ranks of |1,2,3,-1| = 1.5, 3, 4, 1.5 -> positive 8.5, negative 1.5
    assert out["r_rank_biserial"] == pytest.approx((8.5 - 1.5) / 10)


def test_tost_inside_and_outside_margin():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 200)
    assert tost_paired(x, x + 0.01, 0.2)["p"] < 0.05
    assert tost_paired(x, x + 0.5 + rng.normal(0, 0.1, 200), 0.2)["p"] > 0.5


def test_holm_within_family():
    tests = pd.DataFrame({"family": ["F1", "F1", "F2"], "p": [0.01, 0.04, 0.04]})
    out = holm(tests)
    assert out["p_holm"].tolist() == pytest.approx([0.02, 0.04, 0.04])


def test_metrics_and_bootstrap():
    assert accuracy(["a", "b"], ["a", "(invalid)"]) == 0.5
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
    frame = pd.DataFrame({"post_id": [1, 1, 2, 2, 3, 3], "x": [1, 1, 0, 0, 1, 0]})
    first = bootstrap_ci(frame, "post_id", lambda d: d["x"].mean(), 200, 5)
    assert first == bootstrap_ci(frame, "post_id", lambda d: d["x"].mean(), 200, 5)
    assert first[1] <= first[0] <= first[2]
    value, _ = ece([0.9, 0.9], [1, 1])
    assert value == pytest.approx(0.1)


def test_agreement():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    table = one_vs_rest_kappa(["a", "b"], ["a", "a"], ["a", "b", "c"]).set_index("label")
    assert math.isnan(table.loc["c", "kappa"])
    long = pd.DataFrame({"item": [1, 1, 2, 2, 3, 3], "rater": ["x", "y"] * 3, "v": [1, 1, 3, 3, 5, 5]})
    assert ordinal_alpha(long, "item", "rater", "v") == pytest.approx(1.0)
    a, b = pairs(pd.DataFrame({"k": [1, 1, 2], "slot": [1, 2, 1], "v": ["x", "y", "z"]}), "k", "slot", "v")
    assert (a, b) == (["x"], ["y"])


@pytest.mark.parametrize("mean, inter, expected", [
    ((0.8, 0.6, 1.0), (-0.1, 0.2), "Recognition is the bottleneck"),
    ((0.1, -0.2, 0.3), (-0.1, 0.2), "Response generation is the bottleneck"),
    ((0.3, 0.1, 0.6), (0.1, 0.5), "Cultural concepts are the weak point"),
    ((0.3, 0.1, 0.6), (-0.1, 0.5), "Inconclusive"),
])
def test_interpret_oracle(mean, inter, expected):
    assert interpret_oracle(mean, inter, 0.5) == expected
