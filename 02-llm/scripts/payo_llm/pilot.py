"""
Pilot health report: are the prompts, parsing, lengths, timing and label balance fit for the full run?

This module never compares model output with gold labels. The pilot is not a result, and
model-vs-gold numbers come only from the full run (03-evaluation/analysis-plan.md). Model output is
summarized for health only (validity, label collapse, length, timing); label balance comes from the
annotation sheet alone.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from payo_llm.cleanup import DASH, EMOJI, HONESTLY, MARKDOWN, QUOTES, SIGNOFF
from payo_llm.records import KEY_FIELDS
from payo_llm.schema import Schema

# Warning thresholds for the go/no-go discussion. These are pilot heuristics, not statistical tests.
MAX_INVALID_RATE = 0.10        # Task A outputs that are not a schema emotion
MAX_TOP_LABEL_SHARE = 0.50     # one predicted emotion dominating = possible label collapse
MAX_WORDS = 110                # word cap (advice-*_v5.md; payo_llm/cleanup.py enforces it on rater-facing text)
TARGET_WORDS = 90              # their target length ("about 90 words")
MAX_OVER_CAP_SHARE = 0.10
MAX_CTX_SHARE = 0.90           # longest prompt vs the context window
MAX_OPENER_SHARE = 0.30        # one opening phrase across replies = a style tell for raters (v1.8)
MAX_EMOJI_EXCESS = 0.20        # emoji (and, since v1.9, markdown or dash) share above the community's
DOUBLE_SPACE = re.compile(r"\S {2,}\S")
CURLY = re.compile("[\u2018\u2019\u201c\u201d]")
DIBA = re.compile(r"\bdi\s?ba\b", re.IGNORECASE)  # a tag question the models overuse; content, so reported only
# Invented persona or presence claims (advice-*_v3.md forbids them); fixed list, whole-phrase, case-insensitive
PERSONA = re.compile(r"\b(as an? (girl|guy|woman|man|mom|mother|dad|father|student|female|male)|(female|male|woman\'s|man\'s|girl\'s|guy\'s) perspective|we're here|we are here|"
                     r"i'm here for you|i am here for you|ako rin|i've been there|i have been there|i also)\b",
                     re.IGNORECASE)
# Common Filipino function words: their share of tokens tracks how much of a reply is Filipino (Taglish)
TL_WORDS = frozenset("ang ng mga sa na ko mo ka ako ikaw siya kayo tayo natin namin nila niya yung iyong ito yan "
                     "lang naman pa po rin din ba kasi pero talaga hindi wag huwag may wala kung kapag pag para "
                     "dahil sobra ganun ganyan diba yun nga eh daw raw ay at".split())
ANNOTATION_FIELDS = ("exclude_flag", "primary_emotion", "cultural_concept", "expression", "domain", "need_type")
SHEET_CATEGORIES = ("expression", "domain", "need_type")


def load_outputs(out_dir: Path) -> pd.DataFrame:
    """
    Read every main-run JSONL file of a run folder (variance runs are skipped).

    Reruns append a retry after a failed generation, so only the last record per key is kept.

    Args:
        out_dir: The run's output folder, e.g. 02-llm/pilot/outputs.

    Returns:
        One row per generation; empty (with the key columns) when nothing has run yet.
    """
    paths = sorted((out_dir / "emotion-recognition").glob("predictions_*.jsonl"))
    paths += sorted((out_dir / "advice-generation").glob("replies_*.jsonl"))
    paths += sorted((out_dir / "advice-generation" / "oracle_grid").glob("replies_*.jsonl"))
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        return pd.DataFrame(columns=[*KEY_FIELDS, "error"])
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=list(KEY_FIELDS), keep="last").reset_index(drop=True)


def _ok(outputs: pd.DataFrame, task: str) -> pd.DataFrame:
    """Records of one task that finished without an error."""
    rows = outputs[outputs["task"] == task]
    return rows[rows["error"].isna() | (rows["error"] == "")]


def completeness(outputs: pd.DataFrame, expected: dict[str, list[str]],
                 models: tuple[str, ...] = ("gemma", "sealion")) -> pd.DataFrame:
    """
    Count finished, failed and missing generations per model and task.

    Args:
        outputs: From load_outputs.
        expected: Post ids each task should cover (P-Oracle: only the labeled posts).
        models: Models of the run.

    Returns:
        One row per model x task that has any record; tasks with none are "not run".
    """
    rows = []
    for task, ids in expected.items():
        for model in models:
            records = outputs[(outputs["task"] == task) & (outputs["model"] == model)]
            if records.empty:
                rows.append({"model": model, "task": task, "expected": len(ids), "ok": 0, "errors": 0,
                             "missing": len(ids), "error_types": "not run"})
                continue
            failed = records[records["error"].notna() & (records["error"] != "")]
            done = (set(records["post_id"]) - set(failed["post_id"])) & set(ids)
            rows.append({
                "model": model, "task": task, "expected": len(ids), "ok": len(done), "errors": len(failed),
                "missing": len(set(ids) - done),
                "error_types": ", ".join(f"{k}: {v}" for k, v in failed["error"].value_counts().items()),
            })
    return pd.DataFrame(rows)


def recognition_health(outputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    """
    Summarize Task A outputs without any gold labels.

    Args:
        outputs: From load_outputs.

    Returns:
        (per-model summary, predicted-emotion counts by model, same-label rate between the two models
        on posts where both outputs are valid, or None when it cannot be computed).
    """
    rec = _ok(outputs, "recognition")
    if rec.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    summary = []
    for model, rows in rec.groupby("model"):
        valid = rows[rows["valid"].astype(bool)]
        top = valid["emotion"].value_counts()
        concepts = valid["cultural_concept"].dropna()
        confidence = rows["confidence"].dropna()
        reasons = rows["invalid_reason"].dropna().value_counts()
        summary.append({
            "model": model, "n": len(rows), "valid_rate": len(valid) / len(rows),
            "invalid_reasons": ", ".join(f"{k}: {v}" for k, v in reasons.items()),
            "top_label": top.index[0] if len(top) else "",
            "top_share": top.iloc[0] / len(valid) if len(top) else math.nan,
            "concept_none_rate": (concepts == "none").mean() if len(concepts) else math.nan,
            "concept_alias_n": int(rows["concept_alias"].fillna(False).astype(bool).sum())
            if "concept_alias" in rows else 0,
            "conf_min": confidence.min() if len(confidence) else math.nan,
            "conf_median": confidence.median() if len(confidence) else math.nan,
            "conf_max": confidence.max() if len(confidence) else math.nan,
            "conf_unique": confidence.nunique(),
        })
    labels = rec.assign(emotion=rec["emotion"].fillna("(invalid)"))
    distribution = labels.pivot_table(index="emotion", columns="model", values="post_id", aggfunc="count",
                                      fill_value=0)
    distribution.columns.name = None
    distribution = distribution.reset_index()

    both = rec[rec["valid"].astype(bool)].pivot_table(index="post_id", columns="model", values="emotion",
                                                      aggfunc="first").dropna()
    same = (both.iloc[:, 0] == both.iloc[:, 1]).mean() if both.shape[1] == 2 and len(both) else None
    return pd.DataFrame(summary), distribution, same


def advice_health(outputs: pd.DataFrame, community_words: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize Task B reply lengths, with the community reply on the same posts for comparison.

    Args:
        outputs: From load_outputs.
        community_words: Word count of the top community reply, indexed by post_id.

    Returns:
        (length summary per source and condition, Oracle minus P0 length per model).
    """
    grid = _ok(outputs, "advice-oracle-grid").assign(condition="P-Oracle grid")  # one row for all label pairs
    adv = pd.concat([_ok(outputs, "advice-p0"), _ok(outputs, "advice-oracle"), grid])
    if adv.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = [(f"{m} {c}", rows["reply_words"], rows["truncated"].astype(bool))
              for (m, c), rows in adv.groupby(["model", "condition"])]
    posts = adv["post_id"].unique()
    community = community_words.reindex(posts).dropna()
    groups.append(("community (same posts)", community, pd.Series(False, index=community.index)))

    summary = [{
        "source": name, "n": len(words), "median": words.median(), "q1": words.quantile(0.25),
        "q3": words.quantile(0.75), "max": words.max(), "over_cap": (words > MAX_WORDS).mean(),
        "under_half_target": (words < TARGET_WORDS / 2).mean(), "truncated": truncated.mean(), "empty": (words == 0).sum(),
    } for name, words, truncated in groups]

    wide = adv[adv["condition"] != "P-Oracle grid"].pivot_table(index=["model", "post_id"], columns="condition", values="reply_words", aggfunc="first")
    diff = []
    if {"P0", "P-Oracle"} <= set(wide.columns):
        paired = (wide["P-Oracle"] - wide["P0"]).dropna()
        for model, values in paired.groupby(level="model"):
            diff.append({"model": model, "n_pairs": len(values), "median_diff": values.median(),
                         "mean_diff": values.mean()})
    return pd.DataFrame(summary), pd.DataFrame(diff)


def _opener(text: str) -> str:
    """First two words, lowercased and without punctuation."""
    words = re.findall(r"[\w'’-]+", text.lower())
    return " ".join(words[:2])


def _tl_share(texts: pd.Series) -> float:
    """Share of all tokens that are common Filipino function words (TL_WORDS)."""
    tokens = [w for t in texts for w in re.findall(r"[\w'’-]+", t.lower())]
    return sum(w in TL_WORDS for w in tokens) / len(tokens) if tokens else math.nan


def style_health(outputs: pd.DataFrame, community_text: pd.Series) -> pd.DataFrame:
    """
    Surface style tells that could reveal the source to raters, in code (no model, no gold).

    Args:
        outputs: From load_outputs.
        community_text: Top community reply text, indexed by post_id.

    Returns:
        One row per source (model x condition, grid as one row, community on the same posts): the most common
        two-word opener and first word with their shares, the share of replies with an emoji, markdown
        emphasis, an en/em dash, "di ba", a double space, curly quotes, a sentence-initial "Honestly," or a closing
        stock sign-off, persona-claim matches, and the Filipino function-word share of tokens.
    """
    grid = _ok(outputs, "advice-oracle-grid").assign(condition="P-Oracle grid")
    adv = pd.concat([_ok(outputs, "advice-p0"), _ok(outputs, "advice-oracle"), grid])
    if adv.empty:
        return pd.DataFrame()
    groups = [(f"{m} {c}", rows["raw_output"].fillna("")) for (m, c), rows in adv.groupby(["model", "condition"])]
    groups.append(("community (same posts)", community_text.reindex(adv["post_id"].unique()).dropna()))
    rows = []
    for name, texts in groups:
        openers = texts.map(_opener).value_counts()
        firsts = texts.map(lambda t: _opener(t).split(" ")[0]).value_counts()
        rows.append({
            "source": name, "n": len(texts),
            "top_opener": openers.index[0] if len(openers) else "",
            "top_opener_share": openers.iloc[0] / len(texts) if len(openers) else math.nan,
            "top_first_word": firsts.index[0] if len(firsts) else "",
            "top_first_word_share": firsts.iloc[0] / len(texts) if len(firsts) else math.nan,
            "emoji_share": _share(texts, EMOJI),
            "markdown_share": _share(texts, MARKDOWN),
            "dash_share": _share(texts, DASH),
            "diba_share": _share(texts, DIBA),
            "double_space_share": _share(texts, DOUBLE_SPACE),
            "curly_quote_share": _share(texts, CURLY),
            "honestly_share": _share(texts, HONESTLY),
            "signoff_share": texts.map(_ends_with_signoff).mean() if len(texts) else math.nan,
            "persona_n": sum(len(PERSONA.findall(t.translate(QUOTES))) for t in texts),  # curly apostrophes too
            "tl_word_share": _tl_share(texts),
        })
    return pd.DataFrame(rows)


def _ends_with_signoff(text: str) -> bool:
    """True when the last sentence is only a stock sign-off (the closing rule in payo_llm/cleanup.py)."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return len(parts) > 1 and bool(SIGNOFF.fullmatch(parts[-1]))


def _share(texts: pd.Series, pattern: re.Pattern) -> float:
    """Share of texts in which the pattern occurs."""
    return texts.map(lambda t: bool(pattern.search(t))).mean() if len(texts) else math.nan


def style_flags(style: pd.DataFrame) -> list[str]:
    """Warnings from style_health: a dominant opener, emoji/markdown/dashes well above community, any persona claim."""
    if style.empty:
        return []
    community = style[style["source"].str.startswith("community")]
    bases = {col: community[col].iloc[0] if len(community) else 0.0
             for col in ("emoji_share", "markdown_share", "dash_share", "double_space_share", "curly_quote_share",
                         "honestly_share", "signoff_share")}
    base = bases["emoji_share"]
    out = []
    for row in style[~style["source"].str.startswith("community")].itertuples():
        if row.top_opener_share > MAX_OPENER_SHARE:
            out.append(f"{row.source}: {row.top_opener_share:.0%} of replies open with '{row.top_opener}'.")
        elif row.top_first_word_share > MAX_OPENER_SHARE:
            out.append(f"{row.source}: {row.top_first_word_share:.0%} of replies start with '{row.top_first_word}'.")
        if row.emoji_share > base + MAX_EMOJI_EXCESS:
            out.append(f"{row.source}: {row.emoji_share:.0%} of replies have an emoji (community {base:.0%}).")
        for col, what in (("markdown_share", "markdown emphasis"), ("dash_share", "an en/em dash"),
                          ("double_space_share", "a double space"), ("curly_quote_share", "curly quotes"),
                          ("honestly_share", "a sentence-initial 'Honestly,'"),
                          ("signoff_share", "a stock sign-off as the last sentence")):
            if getattr(row, col) > bases[col] + MAX_EMOJI_EXCESS:
                out.append(f"{row.source}: {getattr(row, col):.0%} of replies have {what} (community {bases[col]:.0%}).")
        if row.persona_n:
            out.append(f"{row.source}: {row.persona_n} persona claims (e.g. 'as a girl', 'we're here').")
    return out


def timing(outputs: pd.DataFrame, num_ctx: int) -> pd.DataFrame:
    """
    Latency and prompt length per model and task.

    Args:
        outputs: From load_outputs.
        num_ctx: The backend's context window in tokens.

    Returns:
        One row per model x task.
    """
    done = outputs[outputs["error"].isna() | (outputs["error"] == "")]
    if done.empty:
        return pd.DataFrame()
    rows = []
    for (model, task), group in done.groupby(["model", "task"]):
        tokens = group["n_prompt_tokens"].dropna()
        max_tokens = tokens.max() if len(tokens) else math.nan
        rows.append({
            "model": model, "task": task, "n": len(group), "median_s": group["latency_s"].median(),
            "p95_s": group["latency_s"].quantile(0.95), "total_min": group["latency_s"].sum() / 60,
            "max_prompt_tokens": max_tokens, "ctx_share": max_tokens / num_ctx,
        })
    return pd.DataFrame(rows)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (stable at small n)."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


@dataclass(frozen=True)
class AnnotationSummary:
    """What the pilot labels say about retention and label balance."""

    n_rows: int
    n_labeled: int
    n_excluded: int
    retention: float
    retention_ci: tuple[float, float]
    posts_needed: int | None
    exclude_reasons: pd.DataFrame
    emotions: pd.DataFrame
    emotion_types: pd.DataFrame
    categories: dict[str, pd.DataFrame]
    concept_present: float
    unknown_labels: list[str]
    agreement: pd.DataFrame
    disagreements: pd.DataFrame


def _share_table(values: pd.Series, name: str) -> pd.DataFrame:
    counts = values.value_counts()
    return pd.DataFrame({name: counts.index, "n": counts.values, "share": counts.values / max(len(values), 1)})


def annotation_summary(sheet: pd.DataFrame, schema: Schema, n_silver: int, n_target: int = 300,
                       min_label: int = 15) -> AnnotationSummary:
    """
    Retention and label balance from the pilot annotation sheet.

    Rows are pooled across annotators. `emotion_type` always comes from the schema, never the sheet.

    Args:
        sheet: The annotation sheet (annotation-template columns), read as strings.
        schema: The label schema.
        n_silver: Posts in the silver pool, for the per-label pool estimate.
        n_target: Posts the main sample needs (≈300).
        min_label: The merge threshold (analysis plan §2).

    Returns:
        The summary; on a blank sheet every count is zero.
    """
    sheet = sheet.fillna("").astype(str).apply(lambda col: col.str.strip())
    excluded = sheet["exclude_flag"].str.lower() == "yes"
    labeled = excluded | (sheet["primary_emotion"] != "")
    kept = sheet[labeled & ~excluded]
    n_labeled, n_excluded = int(labeled.sum()), int(excluded.sum())
    retention = (n_labeled - n_excluded) / n_labeled if n_labeled else math.nan
    posts_needed = math.ceil(n_target / retention) if n_labeled and retention > 0 else None

    emotion = kept["primary_emotion"].str.lower()
    counts = emotion.value_counts()
    order = [item["id"] for item in schema.raw["emotions"]]
    table = pd.DataFrame({"emotion": order})
    table["type"] = [schema.emotion(e)["type"] for e in order]
    table["n"] = [int(counts.get(e, 0)) for e in order]
    table["share"] = table["n"] / max(len(kept), 1)
    table["expected_in_300_random"] = table["share"] * n_target
    table["pool_estimate"] = table["share"] * n_silver * (0 if math.isnan(retention) else retention)
    table["below_min"] = table["pool_estimate"] < min_label
    types = table.groupby("type", as_index=False)["n"].sum()
    types["share"] = types["n"] / max(len(kept), 1)

    concept = kept["cultural_concept"].str.lower()
    unknown = sorted(set(emotion) - schema.emotion_ids) + sorted(set(concept) - schema.concept_ids - {""})
    categories = {field: _share_table(kept[field].str.lower(), field) for field in SHEET_CATEGORIES}
    reasons = _share_table(sheet.loc[excluded, "exclude_reason"].replace("", "(none given)"), "exclude_reason")

    agreement, disagreements = _disagreements(sheet[labeled])
    return AnnotationSummary(
        n_rows=len(sheet), n_labeled=n_labeled, n_excluded=n_excluded, retention=retention,
        retention_ci=wilson_ci(n_labeled - n_excluded, n_labeled), posts_needed=posts_needed,
        exclude_reasons=reasons, emotions=table, emotion_types=types, categories=categories,
        concept_present=((concept != "") & (concept != "none")).mean() if len(kept) else math.nan,
        unknown_labels=unknown, agreement=agreement, disagreements=disagreements,
    )


def _disagreements(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-field agreement and the disagreeing values, on posts labeled by more than one annotator."""
    multi = labeled[labeled.groupby("post_id")["post_id"].transform("size") > 1]
    if multi.empty:
        return pd.DataFrame(), pd.DataFrame()
    agreement, rows = [], []
    posts = multi.groupby("post_id")
    for field in ANNOTATION_FIELDS:
        agree = 0
        for post_id, group in posts:
            values = group[field].str.lower()
            if values.nunique() == 1:
                agree += 1
            else:
                rows.append({"post_id": post_id, "field": field, "values": "; ".join(
                    f"{a or '?'}: {v or '-'}" for a, v in zip(group["annotator"], values, strict=True))})
        agreement.append({"field": field, "posts": posts.ngroups, "agree_share": agree / posts.ngroups})
    return pd.DataFrame(agreement), pd.DataFrame(rows)


def flags(complete: pd.DataFrame, recognition: pd.DataFrame, advice: pd.DataFrame, times: pd.DataFrame,
          annotations: AnnotationSummary, n_silver: int, min_label: int = 15) -> list[str]:
    """
    Plain-language warnings for the go/no-go discussion.

    Args:
        complete, recognition, advice, times: Tables from the functions above.
        annotations: From annotation_summary.
        n_silver: Posts in the silver pool.
        min_label: The merge threshold.

    Returns:
        Warnings; empty when nothing tripped a threshold.
    """
    out = []
    for row in complete.itertuples():
        if row.error_types == "not run":
            continue
        if row.errors:
            out.append(f"{row.model} {row.task}: {row.errors} failed generations ({row.error_types}).")
        if row.missing:
            out.append(f"{row.model} {row.task}: {row.missing} of {row.expected} posts have no output.")
    for row in recognition.itertuples():
        if 1 - row.valid_rate > MAX_INVALID_RATE:
            out.append(f"{row.model}: {1 - row.valid_rate:.0%} of Task A outputs are invalid "
                       f"(> {MAX_INVALID_RATE:.0%}); check the prompt or parsing.")
        if row.top_share > MAX_TOP_LABEL_SHARE:
            out.append(f"{row.model}: '{row.top_label}' is {row.top_share:.0%} of valid predictions "
                       f"(> {MAX_TOP_LABEL_SHARE:.0%}); possible label collapse.")
        if row.conf_unique == 1:
            out.append(f"{row.model}: every confidence is {row.conf_min:g}; confidence is not informative.")
    for row in advice.itertuples():
        if row.source.startswith("community"):
            continue
        if row.truncated > 0:
            out.append(f"{row.source}: {row.truncated:.0%} of replies hit max_new_tokens.")
        if row.over_cap > MAX_OVER_CAP_SHARE:
            out.append(f"{row.source}: {row.over_cap:.0%} of replies exceed {MAX_WORDS} words.")
        if row.empty:
            out.append(f"{row.source}: {row.empty} empty replies.")
    for row in times.itertuples():
        if row.ctx_share > MAX_CTX_SHARE:
            out.append(f"{row.model} {row.task}: longest prompt uses {row.ctx_share:.0%} of the context window.")
    a = annotations
    if a.unknown_labels:
        out.append(f"Annotation sheet has labels that are not schema ids: {', '.join(a.unknown_labels)}.")
    if a.posts_needed is not None and a.posts_needed > n_silver:
        out.append(f"At {a.retention:.0%} retention, 300 posts need {a.posts_needed} silver posts "
                   f"but only {n_silver} exist.")
    if a.n_labeled:
        below = a.emotions[a.emotions["below_min"]]
        rare = below.loc[below["n"] > 0, "emotion"].tolist()
        unseen = below.loc[below["n"] == 0, "emotion"].tolist()
        if rare:
            out.append(f"Projected below {min_label} posts in the whole retained pool (merge candidates, "
                       f"analysis plan §2): {', '.join(rare)}.")
        if unseen:
            out.append(f"Not seen in the pilot labels (too few posts to rule them out; watch in the main pass): "
                       f"{', '.join(unseen)}.")
    if not a.disagreements.empty:
        out.append(f"{len(a.disagreements)} annotator disagreements to discuss (codebook §8).")
    return out


def render_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (floats to 2 decimals)."""
    if frame.empty:
        return "_none_"

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.2f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])
