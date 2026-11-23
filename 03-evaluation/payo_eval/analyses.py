"""
Every analysis in the plan, one function per section. Notebooks call these and save the tables.

Each test that enters a Holm family is returned as a row with `family`, `test`, `estimate`, `lower`,
`upper` and `p`, so `07-multiple-comparisons` can correct them together (§10).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from payo_eval.agreement import cohen_kappa, ordinal_alpha, pairs, percent_agreement
from payo_eval.metrics import (accuracy, bootstrap_ci, confusion, ece, macro_f1, majority_baseline, per_class,
                               wilson_ci)
from payo_eval.models import contrast, dummies, gee_logit, holm, mcnemar, mixed, tost_paired, wilcoxon

DIMENSIONS = ("empathy", "cultural_fit", "helpfulness")
TYPES = ("culture_bound", "shared")


def _test(family: str, name: str, hypothesis: str, estimate: float, lower: float, upper: float, p: float,
          scale: str, note: str = "") -> dict:
    return {"family": family, "test": name, "hypothesis": hypothesis, "estimate": estimate, "lower": lower,
            "upper": upper, "p": p, "scale": scale, "note": note}


# ---------------------------------------------------------------- RQ1 (§3)

def recognition_metrics(pred: pd.DataFrame, b: int, seed: int) -> pd.DataFrame:
    """Accuracy and macro-F1 with post-bootstrap CIs, overall and by emotion_type and expression, per model."""
    rows = []
    groups = [("all", "all", pred)]
    groups += [("emotion_type", t, pred[pred["emotion_type"] == t]) for t in sorted(pred["emotion_type"].unique())]
    groups += [("expression", e, pred[pred["expression"] == e]) for e in sorted(pred["expression"].unique())]
    for by, level, frame in groups:
        for model, rows_m in frame.groupby("model"):
            acc = bootstrap_ci(rows_m, "post_id", lambda d: accuracy(d["emotion"], d["pred"]), b, seed)
            f1 = bootstrap_ci(rows_m, "post_id", lambda d: macro_f1(d["emotion"], d["pred"]), b, seed)
            rows.append({"by": by, "level": level, "model": model, "n": len(rows_m),
                         "accuracy": acc[0], "acc_lower": acc[1], "acc_upper": acc[2],
                         "macro_f1": f1[0], "f1_lower": f1[1], "f1_upper": f1[2],
                         "invalid_rate": float((~rows_m["valid"]).mean())})
    return pd.DataFrame(rows)


def recognition_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-class metrics, confusion matrices, majority baseline and concept metrics, per model."""
    out: dict[str, pd.DataFrame] = {}
    first = pred.drop_duplicates("post_id")
    base = majority_baseline(first["emotion"].tolist())
    out["baseline"] = pd.DataFrame([base])
    concept_rows = []
    for model, rows in pred.groupby("model"):
        out[f"per_class_{model}"] = per_class(rows["emotion"].tolist(), rows["pred"].tolist())
        out[f"confusion_{model}"] = confusion(rows["emotion"].tolist(), rows["pred"].tolist())
        for rule, col in (("alias", "concept_pred"), ("strict", "concept_pred_strict")):
            concept_rows.append({"model": model, "rule": rule, "n": len(rows),
                                 "accuracy": accuracy(rows["concept"], rows[col]),
                                 "macro_f1": macro_f1(rows["concept"].tolist(), rows[col].tolist()),
                                 "invalid_rate": float((rows[col] == "(invalid)").mean()),
                                 "alias_share": float(rows["concept_alias"].mean())})
    out["concept"] = pd.DataFrame(concept_rows)
    return out


def recognition_tests(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    GEE (H1a, H1b, H3a) and McNemar between models (overall and culture-bound).

    Returns:
        (Holm-family test rows, McNemar table).
    """
    data = pred[pred["emotion_type"].isin(TYPES)].copy()
    x = pd.DataFrame({"cb": (data["emotion_type"] == "culture_bound").astype(float),
                      "sealion": (data["model"] == "sealion").astype(float),
                      "implicit": (data["expression"] == "implicit").astype(float)}, index=data.index)
    x["cb_x_sealion"] = x["cb"] * x["sealion"]
    tests = []
    try:
        fit = gee_logit(data["correct"], x, data["post_id"])
        params, cov = fit.params, fit.cov_params()
        specs = [("H1a", "culture-bound vs shared (avg over models)", {"cb": 1, "cb_x_sealion": 0.5}, "p_less"),
                 ("H1b", "implicit vs explicit", {"implicit": 1}, "p_less"),
                 ("H3a", "SEA-LION vs Gemma within culture-bound", {"sealion": 1, "cb_x_sealion": 1}, "p_greater")]
        for name, hyp, weights, side in specs:
            c = contrast(params, cov, weights, name)
            tests.append(_test("F1", name, hyp, np.exp(c.estimate), np.exp(c.lower), np.exp(c.upper),
                               getattr(c, side), "odds ratio"))
    except Exception as error:  # noqa: BLE001 - report the failure instead of hiding the whole table
        tests.append(_test("F1", "GEE", "model did not fit", np.nan, np.nan, np.nan, np.nan, "", str(error)))

    wide = pred.pivot_table(index=["post_id", "emotion_type"], columns="model", values="correct").dropna().reset_index()
    rows = []
    for subset, frame in (("all", wide), ("culture_bound", wide[wide["emotion_type"] == "culture_bound"])):
        if len(frame) and {"gemma", "sealion"} <= set(frame.columns):
            m = mcnemar(frame["gemma"], frame["sealion"])
            rows.append({"subset": subset, **m})
            tests.append(_test("F1", f"McNemar ({subset})", "SEA-LION vs Gemma correctness, same posts",
                               m["odds_ratio"], np.nan, np.nan, m["p"], "discordant OR (sealion/gemma)"))
    return pd.DataFrame(tests), pd.DataFrame(rows)


def calibration(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ECE per model and the reliability-diagram bins (descriptive only, v1.1)."""
    summary, bins = [], []
    for model, rows in pred.groupby("model"):
        rows = rows[rows["confidence"].notna()]
        value, table = ece(rows["confidence"] / 100, rows["correct"])
        summary.append({"model": model, "n": len(rows), "ece": value, "n_unique_confidence": rows["confidence"].nunique()})
        bins.append(table.assign(model=model))
    return pd.DataFrame(summary), pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()


# ---------------------------------------------------------------- ratings (§4.1, §5)

def _rater_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    raters = sorted(frame["rater"].unique())
    return dummies(frame, "rater", raters[0], "rater", raters)


def source_model(frame: pd.DataFrame, dim: str, log_length: bool = False):
    """
    `rating ~ source + rater + (1 | post)` with community as the reference source (§4.1).

    Args:
        frame: P0-level rating rows (community, gemma P0, sealion P0).
        dim: Rating column.
        log_length: Add centered log(reply_words) (the v1.1 length sensitivity check).

    Returns:
        (fixed-effect params, covariance, fit).
    """
    x = pd.concat([dummies(frame, "source", "community", "source", ["gemma", "sealion"]), _rater_dummies(frame)], axis=1)
    if log_length:
        words = np.log(frame["reply_words"].clip(lower=1).astype(float))
        x["log_words"] = words - words.mean()
    return mixed(frame[dim], x, frame["post_id"])


def p0_level(ratings: pd.DataFrame) -> pd.DataFrame:
    """Rating rows for the three P0-level sources."""
    return ratings[ratings["condition"].isin(["none", "P0"])].copy()


def advice_vs_community(ratings: pd.DataFrame, margin: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    H2a contrasts per dimension, the non-inferiority check for helpfulness, and the length sensitivity refit.

    Returns:
        (Holm-family test rows, full contrast table incl. sensitivity).
    """
    frame = p0_level(ratings)
    tests, table = [], []
    for dim in DIMENSIONS:
        for sensitivity in (False, True):
            params, cov, fit = source_model(frame, dim, log_length=sensitivity)
            for model in ("gemma", "sealion"):
                c = contrast(params, cov, {f"source_{model}": -1}, f"community - {model}")
                row = {"dimension": dim, "contrast": c.name, "length_adjusted": sensitivity, **{
                    k: v for k, v in c.as_dict().items() if k != "name"}, "warnings": "; ".join(fit.payo_warnings)}
                if dim == "helpfulness":
                    llm_minus = -c.estimate
                    z = (llm_minus + margin) / c.se if c.se > 0 else np.nan
                    row |= {"llm_minus_community": llm_minus, "ni_lower": -c.upper,
                            "non_inferior": -c.upper > -margin, "p_noninferiority": float(stats.norm.sf(z))}
                table.append(row)
                if sensitivity:
                    continue
                if dim == "helpfulness":
                    tests.append(_test("F2", f"H2a helpfulness NI ({model})", f"{model} not worse than community by {margin}",
                                       -c.estimate, -c.upper, -c.lower, row["p_noninferiority"], "points (LLM - community)"))
                else:
                    tests.append(_test("F2", f"H2a {dim} (community - {model})", "community higher",
                                       c.estimate, c.lower, c.upper, c.p_greater, "points"))
    contrasts = pd.DataFrame(table)
    flags = []
    for (dim, name), rows in contrasts.groupby(["dimension", "contrast"]):
        base, adj = rows[~rows["length_adjusted"]].iloc[0], rows[rows["length_adjusted"]].iloc[0]
        changed = np.sign(base.estimate) != np.sign(adj.estimate) or (adj.lower <= 0 <= adj.upper and not
                                                                     base.lower <= 0 <= base.upper)
        flags.append({"dimension": dim, "contrast": name, "length_dependent": bool(changed)})
    return pd.DataFrame(tests), contrasts.merge(pd.DataFrame(flags), on=["dimension", "contrast"])


def upvote_confound(ratings: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between the community reply's upvotes and its ratings (primary ratings)."""
    rows = ratings[(ratings["source"] == "community") & (ratings["slot"] == 1)]
    out = []
    for dim in DIMENSIONS:
        rho, p = stats.spearmanr(rows["reply_score"], rows[dim]) if len(rows) > 2 else (np.nan, np.nan)
        out.append({"dimension": dim, "n": len(rows), "spearman_rho": rho, "p": p})
    return pd.DataFrame(out)


def length_by_source(ratings: pd.DataFrame) -> pd.DataFrame:
    """Reply length per source and condition (one row per item)."""
    items = ratings.drop_duplicates("item_id")
    return items.groupby("source_condition")["reply_words"].describe().reset_index()


def best_fit_rates(best: pd.DataFrame) -> pd.DataFrame:
    """Win rate per source among the three P0-level replies (primary choices), with Wilson CIs."""
    primary = best[best["slot"] == 1]
    rows = []
    for source in ("community", "gemma", "sealion"):
        k, n = int((primary["best_source"] == source).sum()), len(primary)
        low, high = wilson_ci(k, n)
        rows.append({"source": source, "wins": k, "n_posts": n, "rate": k / n if n else np.nan, "lower": low, "upper": high})
    return pd.DataFrame(rows)


def model_comparison(ratings: pd.DataFrame, margin_esc: float, min_posts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    H3b: SEA-LION − Gemma on cultural fit and register (same §4.1 models), and the escalation TOST.

    Returns:
        (Holm-family test rows, escalation table).
    """
    frame = p0_level(ratings)
    tests = []
    for dim in ("cultural_fit", "register_tone"):
        params, cov, _ = source_model(frame, dim)
        c = contrast(params, cov, {"source_sealion": 1, "source_gemma": -1}, "sealion - gemma")
        tests.append(_test("F3", f"H3b {dim} (sealion - gemma)", "SEA-LION higher", c.estimate, c.lower, c.upper,
                           c.p_greater, "points"))
    primary = frame[frame["slot"] == 1]
    need = primary.groupby("post_id")["safe_escalation"].agg(lambda s: bool(s.isin(["appropriate", "missing"]).any()))
    needed = need[need].index
    llm = primary[primary["post_id"].isin(needed) & primary["source"].isin(["gemma", "sealion"])]
    wide = llm.assign(missing=(llm["safe_escalation"] == "missing").astype(float)).pivot_table(
        index="post_id", columns="source", values="missing").dropna()
    n = len(wide)
    if n and {"gemma", "sealion"} <= set(wide.columns):
        result = tost_paired(wide["sealion"], wide["gemma"], margin_esc)
        note = "" if n >= min_posts else f"fewer than {min_posts} posts need escalation: descriptive only"
        tests.append(_test("F3", "H3b escalation TOST (sealion - gemma)", f"missed-escalation gap within ±{margin_esc}",
                           result["diff"], result["ci90_low"], result["ci90_high"],
                           result["p"] if n >= min_posts else np.nan, "rate difference (90% CI)", note))
        table = pd.DataFrame([{"posts_needing_escalation": n, "gemma_missing": wide["gemma"].mean(),
                               "sealion_missing": wide["sealion"].mean(), **result, "note": note}])
    else:
        table = pd.DataFrame([{"posts_needing_escalation": n, "note": "no posts need escalation"}])
    return pd.DataFrame(tests), table


def harm_rates(ratings: pd.DataFrame) -> pd.DataFrame:
    """Share of primary ratings with any harm flag, per source and condition, with Wilson CIs."""
    primary = ratings[ratings["slot"] == 1]
    rows = []
    for name, group in primary.groupby("source_condition"):
        k = int((group["harm_flags"].fillna("none") != "none").sum())
        low, high = wilson_ci(k, len(group))
        rows.append({"source": name, "flagged": k, "n": len(group), "rate": k / len(group), "lower": low, "upper": high})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- drift (§4.2)

def drift(ratings: pd.DataFrame, b: int, seed: int, min_cell: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    H2b: P(autonomy | model) − P(autonomy | community), paired over posts, with McNemar.

    Returns:
        (Holm-family test rows, drift table incl. the neutral-excluded sensitivity, subgroup table).
    """
    primary = p0_level(ratings)
    primary = primary[primary["slot"] == 1]
    codes = primary.pivot_table(index=["post_id", "domain", "concept"], columns="source", values="strategy",
                                aggfunc="first").dropna().reset_index()
    tests, table, groups = [], [], []
    for model in ("gemma", "sealion"):
        if model not in codes:
            continue
        frame = codes.assign(m=(codes[model] == "autonomy").astype(int), c=(codes["community"] == "autonomy").astype(int))
        est = bootstrap_ci(frame, "post_id", lambda d: d["m"].mean() - d["c"].mean(), b, seed)
        test = mcnemar(frame["c"], frame["m"])
        table.append({"model": model, "rule": "all codes", "n_posts": len(frame), "drift": est[0],
                      "lower": est[1], "upper": est[2], "mcnemar_p": test["p"], "odds_ratio": test["odds_ratio"]})
        tests.append(_test("F2", f"H2b drift ({model})", "more autonomy than community", est[0], est[1], est[2],
                           test["p"], "share difference", "McNemar p (two-sided)"))

        def no_neutral(d: pd.DataFrame, model: str = model) -> float:
            m = d.loc[d[model] != "neutral_practical", model]
            c = d.loc[d["community"] != "neutral_practical", "community"]
            return (m == "autonomy").mean() - (c == "autonomy").mean() if len(m) and len(c) else np.nan

        sens = bootstrap_ci(codes, "post_id", no_neutral, b, seed)
        table.append({"model": model, "rule": "neutral excluded", "n_posts": len(codes), "drift": sens[0],
                      "lower": sens[1], "upper": sens[2], "mcnemar_p": np.nan, "odds_ratio": np.nan})
        for by in ("domain", "concept"):
            for level, cell in frame.groupby(by):
                ci = bootstrap_ci(cell, "post_id", lambda d: d["m"].mean() - d["c"].mean(), b, seed)
                groups.append({"model": model, "by": by, "level": level, "n_posts": len(cell), "drift": ci[0],
                               "lower": ci[1], "upper": ci[2], "small_cell": len(cell) < min_cell})
    return pd.DataFrame(tests), pd.DataFrame(table), pd.DataFrame(groups)


# ---------------------------------------------------------------- linking (§6)

def _post_mean(ratings: pd.DataFrame, dim: str) -> pd.DataFrame:
    """Mean rating per post x source x condition over raters."""
    return ratings.groupby(["post_id", "source", "condition", "emotion_type"], as_index=False)[dim].mean()


def recognition_to_advice(ratings: pd.DataFrame, min_posts: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    H4a: the mixed model with correctness, and the within-post contrast (one model right, one wrong).

    Returns:
        (Holm-family test rows, model table, within-post table).
    """
    frame = ratings[(ratings["condition"] == "P0") & ratings["source"].isin(["gemma", "sealion"])].copy()
    frame["correct"] = frame["correct"].astype(float)
    x = pd.concat([
        frame[["correct"]],
        dummies(frame, "emotion_type", "shared", "type", ["culture_bound", "none"]),
        dummies(frame, "source", "gemma", "model", ["sealion"]),
        dummies(frame, "domain", sorted(frame["domain"].unique())[0], "domain"),
        dummies(frame, "expression", "explicit", "expr", ["implicit"]),
        _rater_dummies(frame)], axis=1)
    x["correct_x_cb"] = x["correct"] * x["type_culture_bound"]
    x["correct_x_none"] = x["correct"] * x["type_none"]
    length = frame["post_length_words"].astype(float)
    x["post_length_100w"] = (length - length.mean()) / 100
    tests, table, within = [], [], []
    for dim in DIMENSIONS:
        params, cov, fit = mixed(frame[dim], x, frame["post_id"])
        weights = {"correct": 1.0}
        if "correct_x_cb" in params:
            weights["correct_x_cb"] = 0.5
        c = contrast(params, cov, weights, "correct vs incorrect (avg culture-bound, shared)")
        table.append({"dimension": dim, **c.as_dict(), "warnings": "; ".join(fit.payo_warnings)})
        tests.append(_test("F4", f"H4a {dim}", "higher when recognized", c.estimate, c.lower, c.upper, c.p_greater,
                           "points"))

        means = _post_mean(frame.assign(correct=frame["correct"]), dim)
        correct = frame.drop_duplicates(["post_id", "source"])[["post_id", "source", "correct"]]
        wide = means.merge(correct, on=["post_id", "source"])
        one = wide.groupby("post_id").filter(lambda g: len(g) == 2 and g["correct"].sum() == 1)
        diff = one.groupby("post_id").apply(
            lambda g: g.loc[g["correct"] == 1, dim].iloc[0] - g.loc[g["correct"] == 0, dim].iloc[0])
        w = wilcoxon(diff.to_numpy()) if len(diff) else wilcoxon([])
        within.append({"dimension": dim, **w, "descriptive_only": len(diff) < min_posts})
    return pd.DataFrame(tests), pd.DataFrame(table), pd.DataFrame(within)


def oracle_effect(ratings: pd.DataFrame, b: int, seed: int, margin: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    H4b: P-Oracle − P0 per post, the paired Wilcoxon tests, the three-way model and the interpretation rule.

    Returns:
        (Holm-family test rows, paired table, interpretation table).
    """
    llm = ratings[ratings["source"].isin(["gemma", "sealion"])].copy()
    tests, paired, reading = [], [], []
    for dim in DIMENSIONS:
        means = _post_mean(llm, dim)
        wide = means.pivot_table(index=["post_id", "source", "emotion_type"], columns="condition", values=dim).dropna()
        wide = wide.reset_index().assign(delta=lambda d: d["P-Oracle"] - d["P0"])
        for (model, etype), cell in wide.groupby(["source", "emotion_type"]):
            paired.append({"dimension": dim, "model": model, "emotion_type": etype, **wilcoxon(cell["delta"])})
        for model, cell in wide.groupby("source"):
            paired.append({"dimension": dim, "model": model, "emotion_type": "all", **wilcoxon(cell["delta"])})

        data = llm[llm["emotion_type"].isin(TYPES)]
        x = pd.DataFrame({"oracle": (data["condition"] == "P-Oracle").astype(float),
                          "cb": (data["emotion_type"] == "culture_bound").astype(float),
                          "sealion": (data["source"] == "sealion").astype(float)}, index=data.index)
        x["oracle_x_cb"] = x["oracle"] * x["cb"]
        x["oracle_x_sealion"] = x["oracle"] * x["sealion"]
        x["cb_x_sealion"] = x["cb"] * x["sealion"]
        x["oracle_x_cb_x_sealion"] = x["oracle"] * x["cb"] * x["sealion"]
        x = pd.concat([x, _rater_dummies(data)], axis=1)
        params, cov, _ = mixed(data[dim], x, data["post_id"])
        inter = contrast(params, cov, {"oracle_x_cb": 1, "oracle_x_cb_x_sealion": 0.5}, "oracle x culture-bound")
        tests.append(_test("F4", f"H4b {dim}", "oracle gain larger for culture-bound", inter.estimate, inter.lower,
                           inter.upper, inter.p_greater, "points (interaction)"))
        mean = bootstrap_ci(wide, "post_id", lambda d: d["delta"].mean(), b, seed)
        reading.append({"dimension": dim, "mean_delta": mean[0], "lower": mean[1], "upper": mean[2],
                        "interaction": inter.estimate, "inter_lower": inter.lower, "inter_upper": inter.upper,
                        "reading": interpret_oracle(mean, (inter.lower, inter.upper), margin)})
    return pd.DataFrame(tests), pd.DataFrame(paired), pd.DataFrame(reading)


def interpret_oracle(mean: Sequence[float], interaction_ci: Sequence[float], margin: float) -> str:
    """The pre-specified §6.2 reading of the oracle results."""
    est, low, high = mean
    readings = []
    if est >= margin and low > 0:
        readings.append("Recognition is the bottleneck")
    elif -margin < low and high < margin:
        readings.append("Response generation is the bottleneck")
    if interaction_ci[0] > 0:
        readings.append("Cultural concepts are the weak point")
    return "; ".join(readings) or "Inconclusive"


def prompt_tokens(replies: pd.DataFrame) -> pd.DataFrame:
    """Mean prompt tokens per model and condition (prompt-length confound, §6.2)."""
    return replies.groupby(["model", "condition"])["n_prompt_tokens"].agg(["count", "mean", "median"]).reset_index()


# ---------------------------------------------------------------- reliability (§9)

def rating_reliability(ratings: pd.DataFrame, best: pd.DataFrame, tentative_below: float) -> pd.DataFrame:
    """R2 (strategy κ) and R3 (α per dimension, % agreement on escalation and best fit)."""
    rows = []
    for dim in (*DIMENSIONS, "register_tone"):
        alpha = ordinal_alpha(ratings, "item_id", "rater", dim)
        rows.append({"id": "R3", "measure": f"{dim} α (ordinal)", "value": alpha,
                     "n_items": int((ratings.groupby("item_id").size() > 1).sum()),
                     "tentative": bool(alpha < tentative_below) if alpha == alpha else None})
    a, b = pairs(ratings, "item_id", "slot", "safe_escalation")
    rows.append({"id": "R3", "measure": "safe_escalation % agreement", "value": percent_agreement(a, b), "n_items": len(a)})
    a, b = pairs(best, "post_id", "slot", "best_source")
    rows.append({"id": "R3", "measure": "best fit % agreement", "value": percent_agreement(a, b), "n_items": len(a)})
    a, b = pairs(ratings, "item_id", "slot", "strategy")
    rows.append({"id": "R2", "measure": "strategy Cohen κ", "value": cohen_kappa(a, b), "n_items": len(a)})
    return pd.DataFrame(rows)


def generation_variance(variance: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """
    R4: Task A label stability across sampled seeds (and vs. the greedy run), and advice length spread.

    The advice strategy-change part was dropped in v1.2 (no extra coding), so advice is length only.
    """
    if variance.empty:
        return pd.DataFrame([{"measure": "variance runs", "value": np.nan, "note": "no variance outputs yet"}])
    ok = variance[variance["error"].isna() | (variance["error"] == "")]
    rows = []
    rec = ok[ok["task"] == "recognition"]
    for model, group in rec.groupby("model"):
        labels = group.pivot_table(index="post_id", columns="seed", values="emotion", aggfunc="first")
        same = labels.nunique(axis=1, dropna=False) == 1
        rows.append({"measure": f"{model}: Task A same label across seeds", "value": float(same.mean()), "n": len(labels)})
        greedy = main[main["model"] == model].set_index("post_id")["pred_original"]
        agree = labels.apply(lambda col: col == greedy.reindex(labels.index)).mean(axis=1)
        rows.append({"measure": f"{model}: Task A agreement with greedy (mean over seeds)", "value": float(agree.mean()),
                     "n": len(labels)})
    adv = ok[ok["task"] != "recognition"]
    for (model, condition), group in adv.groupby(["model", "condition"]):
        sd = group.groupby("post_id")["reply_words"].std()
        rows.append({"measure": f"{model} {condition}: median SD of reply words across seeds",
                     "value": float(sd.median()), "n": len(sd)})
    return pd.DataFrame(rows)


def apply_holm(tests: pd.DataFrame) -> pd.DataFrame:
    """Holm within each family (§10)."""
    return holm(tests)


# ---------------------------------------------------------------- pre-screen selection bias (§2 step 3)

def prescreen_bias(prescreen: pd.DataFrame, gold: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    How well the quick pre-screen predicted the gold labels, on the sampled posts that have gold labels.

    Args:
        prescreen: post_id, concept_likely, domain_guess (the screener's sheet, saved to gold).
        gold: Kept gold posts (post_id, concept, domain).

    Returns:
        concept_share (gold concept present by concept_likely), accuracy (sensitivity, specificity,
        domain agreement and κ).
    """
    df = gold.merge(prescreen[["post_id", "concept_likely", "domain_guess"]], on="post_id")
    df["concept_present"] = (df["concept"] != "none") & (df["concept"] != "")
    share = df.groupby("concept_likely")["concept_present"].agg(["size", "mean"]).reset_index()
    share = share.rename(columns={"size": "n_posts", "mean": "share_concept_present"})
    yes = df["concept_likely"] == "yes"
    tp, fn = int((yes & df["concept_present"]).sum()), int((~yes & df["concept_present"]).sum())
    tn, fp = int((~yes & ~df["concept_present"]).sum()), int((yes & ~df["concept_present"]).sum())
    rows = [{"measure": "sensitivity (concept_likely=yes | concept present)", "value": tp / (tp + fn) if tp + fn else np.nan,
             "n": tp + fn},
            {"measure": "specificity (concept_likely=no | no concept)", "value": tn / (tn + fp) if tn + fp else np.nan,
             "n": tn + fp},
            {"measure": "domain_guess = gold domain", "value": percent_agreement(df["domain_guess"], df["domain"]),
             "n": len(df)},
            {"measure": "domain Cohen κ", "value": cohen_kappa(df["domain_guess"], df["domain"]), "n": len(df)}]
    return {"concept_share": share, "accuracy": pd.DataFrame(rows)}
