"""
Blinded rating items for the rated set, the rater sheets, and collecting the ratings back.

The key (item_id -> post, source, condition) is written to gold and never to a rater's sheet. Sheets
show only a post reference, the post, and reply text, shuffled within each post (rating rubric rules 1-6).
"""

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from payo_eval.instruments import BEST_FIT, bestfit_columns, rating_columns
from payo_eval.sheets import normalize, validate
from payo_llm.cleanup import clean_reply
from payo_llm.schema import Schema

LLM_ITEMS = (("gemma", "P0"), ("sealion", "P0"), ("gemma", "P-Oracle"), ("sealion", "P-Oracle"))
P0_LEVEL = (("community", "none"), ("gemma", "P0"), ("sealion", "P0"))
TRAILER = re.compile(r"^\s*\**\s*(edit|update|eta)\s*\d*\s*\**\s*[:\-–]", re.IGNORECASE)


def clean_community(text: str) -> str:
    """
    Rubric rule 5: drop Reddit quote lines (`>`) and everything from an `Edit:` / `Update:` trailer on.

    Args:
        text: The top community reply (already PII-masked in silver).

    Returns:
        The cleaned reply, or the original text if cleaning would leave nothing.
    """
    kept = []
    for line in text.splitlines():
        if TRAILER.match(line):
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned or text.strip()


def build_items(posts: pd.DataFrame, community: pd.DataFrame, replies: Mapping[tuple[str, str], pd.DataFrame],
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Five blinded items per rated post, in a random order within the post.

    Args:
        posts: Rated posts (post_id).
        community: post_id, reply_body, reply_score from silver.
        replies: (model, condition) -> post_id, raw_output, truncated from the full run.
        seed: Random seed.

    Returns:
        (key: item_id, post_id, post_ref, source, condition, position, reply_words, reply_words_raw,
         clean_ops, reply_score; texts: item_id, reply) — texts go only to the rater sheets. Every reply,
         community included, gets the same fixed cleanup (payo_llm.cleanup, v1.8); reply_words is what raters see.
    """
    rng = np.random.default_rng(seed)
    community = community.set_index("post_id")
    tables = {k: v.set_index("post_id") for k, v in replies.items()}
    missing = [(pid, m, c) for pid in posts["post_id"] for (m, c), t in tables.items() if pid not in t.index]
    missing += [(pid, m, c) for pid in posts["post_id"] for (m, c) in LLM_ITEMS if (m, c) not in tables]
    if missing:
        raise SystemExit(f"{len(missing)} LLM replies missing for rated posts, e.g. {missing[:3]}. "
                         "Finish the full run (P0 and P-Oracle) first.")
    order = [posts["post_id"].iloc[i] for i in rng.permutation(len(posts))]
    refs = {pid: f"P{n:03d}" for n, pid in enumerate(order, start=1)}
    used: set[str] = set()
    key_rows, text_rows = [], []
    for pid in order:
        reply = clean_community(str(community.loc[pid, "reply_body"]))
        items = [("community", "none", reply, False, community.loc[pid, "reply_score"])]
        for model, condition in LLM_ITEMS:
            row = tables[(model, condition)].loc[pid]
            items.append((model, condition, str(row["raw_output"]), bool(row.get("truncated", False)), np.nan))
        for position, index in enumerate(rng.permutation(len(items)), start=1):
            source, condition, raw, truncated, score = items[index]
            cleaned = clean_reply(raw, truncated=truncated)
            item_id = _new_id(rng, used)
            key_rows.append({"item_id": item_id, "post_id": pid, "post_ref": refs[pid], "source": source,
                             "condition": condition, "position": position, "reply_words": cleaned.words,
                             "reply_words_raw": len(raw.split()), "clean_ops": ";".join(cleaned.ops),
                             "reply_score": score})
            text_rows.append({"item_id": item_id, "reply": cleaned.text})
    return pd.DataFrame(key_rows), pd.DataFrame(text_rows)


def _new_id(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        item_id = "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), size=6))
        if item_id not in used:
            used.add(item_id)
            return item_id


def bestfit_key(key: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Letters A-C for the three P0-level replies of each post, shuffled per post.

    Returns:
        post_id, post_ref, letter, item_id.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    p0 = key[[tuple(x) in P0_LEVEL for x in key[["source", "condition"]].itertuples(index=False)]]
    for (pid, ref), group in p0.sort_values("post_ref").groupby(["post_id", "post_ref"]):
        shuffled = group.iloc[rng.permutation(len(group))]
        for letter, item_id in zip(BEST_FIT, shuffled["item_id"], strict=True):
            rows.append({"post_id": pid, "post_ref": ref, "letter": letter, "item_id": item_id})
    return pd.DataFrame(rows)


def rater_frames(rater: str, assignment: pd.DataFrame, key: pd.DataFrame, texts: pd.DataFrame,
                 post_texts: pd.Series, bf_key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The read-only part of one rater's two sheets.

    Returns:
        (rating rows: post_ref, item_id, post, reply — one row per item, posts in post_ref order;
         best-fit rows: post_ref, post, reply_A, reply_B, reply_C).
    """
    mine = assignment.loc[assignment["person"] == rater, "post_id"]
    items = key[key["post_id"].isin(mine)].merge(texts, on="item_id").sort_values(["post_ref", "position"])
    items["post"] = items["post_id"].map(post_texts)
    rating = items[["post_ref", "item_id", "post", "reply"]].reset_index(drop=True)
    letters = bf_key[bf_key["post_id"].isin(mine)].merge(texts, on="item_id")
    wide = letters.pivot_table(index=["post_id", "post_ref"], columns="letter", values="reply", aggfunc="first")
    wide = wide.reset_index().sort_values("post_ref")
    wide["post"] = wide["post_id"].map(post_texts)
    best = wide[["post_ref", "post", *BEST_FIT]].rename(columns={k: f"reply_{k}" for k in BEST_FIT})
    return rating, best.reset_index(drop=True)


def collect_ratings(sheets: Mapping[str, pd.DataFrame], bestfits: Mapping[str, pd.DataFrame], schema: Schema,
                    assignment: pd.DataFrame, key: pd.DataFrame, bf_key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate every rater's sheets against the rubric and their assignment, and return long tables.

    Returns:
        (ratings: item_id, rater, slot, rating columns; bestfit: post_id, rater, slot, best_item_id).
    """
    columns, bf_columns = rating_columns(schema), bestfit_columns()
    problems, ratings, choices = [], [], []
    slots = assignment.rename(columns={"person": "rater"})[["post_id", "rater", "slot"]]
    for rater, sheet in sheets.items():
        problems += validate(sheet, columns, "item_id", rater)
        expected = set(key.loc[key["post_id"].isin(slots.loc[slots["rater"] == rater, "post_id"]), "item_id"])
        got = set(sheet.get("item_id", pd.Series(dtype=str)))
        if expected != got:
            problems.append(f"{rater}: sheet items don't match the assignment "
                            f"({len(expected - got)} missing, {len(got - expected)} unexpected)")
        problems += _harm_none_alone(sheet, rater)
        ratings.append(normalize(sheet, columns).assign(rater=rater))
    for rater, sheet in bestfits.items():
        problems += validate(sheet, bf_columns, "post_ref", f"{rater} best fit")
        choices.append(sheet.assign(rater=rater, best=sheet["best"].str.upper()))
    if problems:
        raise SystemExit("Rating sheets have problems:\n  " + "\n  ".join(problems))

    names = [c.name for c in columns]
    long = pd.concat(ratings, ignore_index=True)[["item_id", "rater", *names]]
    long = long.merge(key[["item_id", "post_id"]], on="item_id").merge(slots, on=["post_id", "rater"])
    for dim in ("empathy", "cultural_fit", "helpfulness", "register_tone"):
        long[dim] = long[dim].astype(int)
    best = pd.concat(choices, ignore_index=True)[["post_ref", "rater", "best", "notes"]]
    best = best.merge(bf_key, left_on=["post_ref", "best"], right_on=["post_ref", "letter"])
    best = best.merge(slots, on=["post_id", "rater"])
    return (long.drop(columns="post_id"),
            best[["post_id", "rater", "slot", "item_id", "notes"]].rename(columns={"item_id": "best_item_id"}))


def _harm_none_alone(sheet: pd.DataFrame, rater: str) -> list[str]:
    """`none` can't be combined with a harm flag."""
    out = []
    for _, row in sheet.iterrows():
        parts = [p.strip().lower() for p in str(row.get("harm_flags", "")).split(";") if p.strip()]
        if "none" in parts and len(parts) > 1:
            out.append(f"{rater}, item_id {row['item_id']}, harm_flags: 'none' combined with other flags")
    return out
