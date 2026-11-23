"""
Analysis-ready tables from the frozen gold layer. Notebooks only read through these functions.

Merges (analysis plan §2) are applied here, to gold labels and model predictions alike.
"""

import json
from pathlib import Path

import pandas as pd

from payo_llm.pilot import load_outputs
from payo_llm.schema import Schema, load_schema
from payo_eval.paths import Paths, get_paths

MODELS = ("gemma", "sealion")


def merges(paths: Paths | None = None) -> dict[str, str]:
    """The merge map written by `workflow.py finalize-labels` (label -> final label)."""
    path = (paths or get_paths()).er / "merges.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `workflow.py finalize-labels` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def labels(paths: Paths | None = None) -> pd.DataFrame:
    """
    Kept gold posts with everything the models condition on.

    Returns:
        post_id, emotion (final), emotion_original, emotion_type, concept, expression, domain, need_type,
        arm, post_length_words.
    """
    paths = paths or get_paths()
    gold = pd.read_csv(paths.adjudicated, dtype=str).fillna("")
    gold = gold[gold["exclude_flag"] != "yes"]
    sample = pd.read_csv(paths.sample, dtype={"post_id": str})
    out = gold.merge(sample[["post_id", "arm", "post_length_words"]], on="post_id", how="left")
    out = out.rename(columns={"emotion_final": "emotion", "primary_emotion": "emotion_original",
                              "cultural_concept": "concept"})
    cols = ["post_id", "emotion", "emotion_original", "emotion_type", "concept", "expression", "domain",
            "need_type", "arm", "post_length_words"]
    return out[cols].reset_index(drop=True)


def predictions(paths: Paths | None = None, schema: Schema | None = None) -> pd.DataFrame:
    """
    Task A outputs joined with gold, with merges applied to predictions as to gold.

    Returns:
        One row per post x model: gold columns plus pred (merged), pred_original, valid, correct,
        concept_pred (alias rule), concept_pred_strict, concept_correct(_strict), confidence.
    """
    paths = paths or get_paths()
    schema = schema or load_schema()
    mapping = merges(paths)
    outputs = load_outputs(paths.gold)
    rec = outputs[(outputs["task"] == "recognition") & (outputs["error"].isna() | (outputs["error"] == ""))]
    rec = rec[["post_id", "model", "emotion", "cultural_concept", "confidence", "valid",
               *[c for c in ("concept_alias",) if c in rec]]].copy()
    if "concept_alias" not in rec:
        rec["concept_alias"] = False
    rec["concept_alias"] = rec["concept_alias"].fillna(False).astype(bool)
    rec = rec.rename(columns={"emotion": "pred_original"})
    rec["pred_original"] = rec["pred_original"].fillna("")
    gold = labels(paths)
    df = gold.merge(rec, on="post_id", how="inner")
    df["valid"] = df["valid"].astype(bool)
    df["pred"] = df["pred_original"].map(lambda e: mapping.get(e, e) if e else "(invalid)")
    df.loc[~df["valid"], "pred"] = "(invalid)"
    df["correct"] = (df["pred"] == df["emotion"]).astype(int)
    df["concept_pred"] = df["cultural_concept"].fillna("(invalid)")
    df["concept_pred_strict"] = df["concept_pred"].where(~df["concept_alias"], "(invalid)")
    df["concept_correct"] = (df["concept_pred"] == df["concept"]).astype(int)
    df["concept_correct_strict"] = (df["concept_pred_strict"] == df["concept"]).astype(int)
    missing = set(gold["post_id"]) - set(df["post_id"])
    if missing:
        print(f"Note: {len(missing)} gold posts have no Task A output yet.")
    return df.drop(columns=["cultural_concept"]).reset_index(drop=True)


def replies(paths: Paths | None = None) -> pd.DataFrame:
    """Task B records (main run): post_id, model, condition, reply_words, n_prompt_tokens, truncated."""
    outputs = load_outputs((paths or get_paths()).gold)
    adv = outputs[outputs["task"].isin(["advice-p0", "advice-oracle"])]
    adv = adv[adv["error"].isna() | (adv["error"] == "")]
    return adv[["post_id", "model", "condition", "reply_words", "n_prompt_tokens", "truncated"]].reset_index(drop=True)


def ratings(paths: Paths | None = None) -> pd.DataFrame:
    """
    Every author rating joined with the blinding key, gold labels and Task A correctness.

    Returns:
        One row per item x rater: item_id, rater, slot, rating columns, post_id, source, condition,
        reply_words, reply_score, gold columns, correct (LLM items; NaN for community).
    """
    paths = paths or get_paths()
    rated = pd.read_csv(paths.ratings, dtype={"item_id": str, "rater": str, "harm_flags": str, "strategy": str,
                                              "safe_escalation": str})
    key = pd.read_csv(paths.items_key, dtype={"item_id": str, "post_id": str})
    df = rated.merge(key, on="item_id").merge(labels(paths), on="post_id")
    correct = predictions(paths)[["post_id", "model", "correct"]].rename(columns={"model": "source"})
    df = df.merge(correct, on=["post_id", "source"], how="left")
    df["source_condition"] = df["source"].where(df["condition"] == "none", df["source"] + "_" + df["condition"])
    return df


def bestfit(paths: Paths | None = None) -> pd.DataFrame:
    """Best-fit choices with the chosen item's source: post_id, rater, slot, best_item_id, best_source."""
    paths = paths or get_paths()
    best = pd.read_csv(paths.bestfit, dtype={"post_id": str, "rater": str, "best_item_id": str})
    key = pd.read_csv(paths.items_key, dtype={"item_id": str})
    return best.merge(key[["item_id", "source"]].rename(columns={"item_id": "best_item_id", "source": "best_source"}),
                      on="best_item_id")


def variance(paths: Paths | None = None) -> pd.DataFrame:
    """Sampled-decoding runs (R4): every record under gold/variance/."""
    folder: Path = (paths or get_paths()).gold / "variance"
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return pd.DataFrame(rows)
