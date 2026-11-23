"""
From annotator sheets to gold labels: collect, find disagreements, adjudicate, merge rare labels, κ gate.

The adjudicated sheet keeps each post's original label (it feeds the P-Oracle prompts). Merges are a
scoring-time map (`emotion_final`), so prompts and the schema never change after annotation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from payo_llm.schema import Schema
from payo_eval.agreement import cohen_kappa, one_vs_rest_kappa, pairs
from payo_eval.instruments import (LABEL_FIELDS, adjudication_columns, annotation_columns,
                                   check_exclusion_reasons)
from payo_eval.sheets import normalize, validate

KAPPA_FIELDS = ("primary_emotion", "cultural_concept", "expression")


def collect(sheets: Mapping[str, pd.DataFrame], schema: Schema, assignment: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Validate every annotator's sheet and stack them into one long table.

    Args:
        sheets: annotator id -> their sheet (from read_sheet).
        schema: For the allowed values.
        assignment: Expected (post_id, person) pairs; rows outside it, or missing from it, are errors.

    Returns:
        post_id, annotator and the label fields (normalized), one row per annotator per post.
    """
    columns = annotation_columns(schema)
    problems, frames = [], []
    for person, sheet in sheets.items():
        problems += validate(sheet, columns, "post_id", person)
        if not problems:
            problems += check_exclusion_reasons(sheet, "post_id", person)
        frames.append(normalize(sheet, columns).assign(annotator=person))
    if problems:
        raise SystemExit("Annotation sheets have problems:\n  " + "\n  ".join(problems))
    raw = pd.concat(frames, ignore_index=True)[["post_id", "annotator", *LABEL_FIELDS, "notes"]]
    _clear_excluded(raw)
    if assignment is not None:
        expected = set(zip(assignment["post_id"], assignment["person"], strict=True))
        got = set(zip(raw["post_id"], raw["annotator"], strict=True))
        if expected != got:
            raise SystemExit(f"Sheets don't match the assignment: {len(expected - got)} missing, "
                             f"{len(got - expected)} unexpected rows.")
    return raw.sort_values(["post_id", "annotator"]).reset_index(drop=True)


def _clear_excluded(frame: pd.DataFrame, prefix: str = "") -> None:
    """Excluded posts carry no labels (codebook §2), even if someone filled them in."""
    excluded = frame[f"{prefix}exclude_flag"] == "yes"
    for field in LABEL_FIELDS[2:]:
        frame.loc[excluded, f"{prefix}{field}"] = ""


def double_posts(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rows of posts labeled by two or more annotators, with slots 1, 2, ... in annotator order.

    The main study has exactly two per double post; the pilot has every annotator on every post.
    κ uses slots 1 and 2.
    """
    counts = raw.groupby("post_id")["annotator"].transform("size")
    double = raw[counts >= 2].sort_values(["post_id", "annotator"]).copy()
    double["slot"] = double.groupby("post_id").cumcount() + 1
    return double


def disagreements(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Every double-annotated post with at least one differing field.

    Returns:
        post_id, a field "<annotator>: <value> | <annotator>: <value>" per label field, and
        `agreed_<field>` (the shared value, or "" where they differ), which prefills the adjudication sheet.
    """
    double = double_posts(raw)
    rows = []
    for post_id, group in double.groupby("post_id"):
        row: dict[str, str] = {"post_id": post_id}
        differs = False
        for field in LABEL_FIELDS:
            values = group[field].tolist()
            same = len(set(values)) == 1
            differs |= not same
            row[field] = " | ".join(f"{a}: {v or '-'}" for a, v in zip(group["annotator"], values, strict=True))
            row[f"agreed_{field}"] = values[0] if same else ""
        if differs:
            rows.append(row)
    return pd.DataFrame(rows, columns=["post_id", *LABEL_FIELDS, *[f"agreed_{f}" for f in LABEL_FIELDS]])


def read_decisions(sheet: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Validate the adjudicator's sheet and return post_id + final label fields (without the prefix)."""
    columns = adjudication_columns(schema)
    problems = validate(sheet, columns, "post_id", "adjudication")
    if not problems:
        problems += check_exclusion_reasons(sheet, "post_id", "adjudication", prefix="final_")
    if problems:
        raise SystemExit("Adjudication sheet has problems:\n  " + "\n  ".join(problems))
    decided = normalize(sheet, columns)
    _clear_excluded(decided, prefix="final_")
    decided = decided[["post_id", *[f"final_{f}" for f in LABEL_FIELDS]]]
    return decided.rename(columns={f"final_{f}": f for f in LABEL_FIELDS})


def adjudicate(raw: pd.DataFrame, decisions: pd.DataFrame | None) -> pd.DataFrame:
    """
    One label row per post: single-annotated as labeled, agreed doubles as agreed, the rest as decided.

    Args:
        raw: From collect.
        decisions: From read_decisions (may be None when no post needs adjudication).

    Returns:
        post_id, label fields, `label_source` (single / agreed / adjudicated).
    """
    needed = set(disagreements(raw)["post_id"])
    decided = set() if decisions is None else set(decisions["post_id"])
    if needed - decided:
        raise SystemExit(f"{len(needed - decided)} disagreements have no adjudication, e.g. {sorted(needed - decided)[:3]}")
    if decided - needed:
        raise SystemExit(f"Adjudication rows for posts that need none: {sorted(decided - needed)[:3]}")
    per_post = raw.groupby("post_id").size()
    first = raw.sort_values(["post_id", "annotator"]).drop_duplicates("post_id")
    first = first.assign(label_source=first["post_id"].map(
        lambda p: "adjudicated" if p in needed else ("agreed" if per_post[p] >= 2 else "single")))
    labels = first.set_index("post_id")[[*LABEL_FIELDS, "label_source"]]
    if decisions is not None and len(decisions):
        labels.update(decisions.set_index("post_id")[list(LABEL_FIELDS)])
    return labels.reset_index()


def resolve_merges(counts: Mapping[str, int], merges: Mapping[str, str], min_label: int,
                   labels: Sequence[str]) -> dict[str, str]:
    """
    Decide which labels merge, following chains (a merged target that is itself rare merges on).

    Args:
        counts: Adjudicated posts per label (kept posts only).
        merges: Proposed label -> target ("" = not decided yet).
        min_label: The threshold.
        labels: Every emotion id (labels with no posts are left alone: they never occur).

    Returns:
        label -> final label for every label that merges.
    """
    current = {label: counts.get(label, 0) for label in labels}
    mapping: dict[str, str] = {}
    while True:
        rare = [l for l, n in current.items() if 0 < n < min_label and l not in mapping]
        if not rare:
            return mapping
        for label in rare:
            target = (merges.get(label) or "").strip()
            if not target:
                raise SystemExit(f"Label {label!r} has {current[label]} posts (< {min_label}) and no merge target: "
                                 f"set [merges] {label} in 03-evaluation/config.toml.")
            if target not in current:
                raise SystemExit(f"[merges] {label} -> {target!r} is not an emotion id.")
            seen = {label}
            while target in mapping:
                target = mapping[target]
                if target in seen:
                    raise SystemExit(f"[merges] has a cycle through {label!r}.")
                seen.add(target)
            mapping[label] = target
            current[target] += current[label]
            current[label] = 0
            for source, dest in mapping.items():  # re-point anything that merged into `label`
                if dest == label:
                    mapping[source] = target


@dataclass(frozen=True)
class Finalized:
    """Gold labels plus everything the §2 report needs."""

    labels: pd.DataFrame
    merges: dict[str, str]
    counts: pd.DataFrame
    field_kappa: pd.DataFrame
    label_kappa: pd.DataFrame
    gate_failures: list[str]
    exclusions: pd.DataFrame


def finalize(raw: pd.DataFrame, decisions: pd.DataFrame | None, schema: Schema, merges: Mapping[str, str],
             min_label: int, kappa_gate: float) -> Finalized:
    """
    Adjudicate, add emotion_final and emotion_type, and compute the merge counts and the κ gate.

    Args:
        raw: All annotator rows (from collect).
        decisions: The adjudicator's decisions (or None).
        schema: Label schema (emotion_type comes from here, never from a sheet).
        merges: The [merges] config table.
        min_label, kappa_gate: Thresholds from config.

    Returns:
        The finalized labels and report tables.
    """
    labels = adjudicate(raw, decisions)
    kept = labels[labels["exclude_flag"] != "yes"]
    emotion_ids = [item["id"] for item in schema.raw["emotions"]]
    before = kept["primary_emotion"].value_counts().to_dict()
    mapping = resolve_merges(before, merges, min_label, emotion_ids)
    labels["emotion_final"] = labels["primary_emotion"].map(lambda e: mapping.get(e, e) if e else "")
    labels["emotion_type"] = labels["emotion_final"].map(lambda e: schema.emotion(e)["type"] if e else "")
    after = labels.loc[labels["exclude_flag"] != "yes", "emotion_final"].value_counts().to_dict()
    counts = pd.DataFrame({"emotion": emotion_ids, "type": [schema.emotion(e)["type"] for e in emotion_ids],
                           "n_before": [before.get(e, 0) for e in emotion_ids],
                           "merged_into": [mapping.get(e, "") for e in emotion_ids],
                           "n_after": [after.get(e, 0) for e in emotion_ids]})

    double = double_posts(raw)
    both_kept = double.groupby("post_id")["exclude_flag"].transform(lambda s: (s != "yes").all())
    kept_double = double[both_kept]
    field_rows = []
    a, b = pairs(double, "post_id", "slot", "exclude_flag")
    field_rows.append({"field": "exclude_flag", "n_posts": len(a), "kappa": cohen_kappa(a, b)})
    for field in KAPPA_FIELDS:
        a, b = pairs(kept_double, "post_id", "slot", field)
        field_rows.append({"field": field, "n_posts": len(a), "kappa": cohen_kappa(a, b)})
    a, b = pairs(kept_double, "post_id", "slot", "primary_emotion")
    a_final, b_final = [mapping.get(x, x) for x in a], [mapping.get(x, x) for x in b]
    field_rows.append({"field": "emotion_final", "n_posts": len(a), "kappa": cohen_kappa(a_final, b_final)})
    raw_k = one_vs_rest_kappa(a, b, emotion_ids).rename(columns={"kappa": "kappa_before"})
    final_k = one_vs_rest_kappa(a_final, b_final, emotion_ids)[["label", "kappa"]].rename(
        columns={"kappa": "kappa_after"})
    label_kappa = raw_k.merge(final_k, on="label")
    finals = set(labels.loc[labels["exclude_flag"] != "yes", "emotion_final"])
    failures = [f"{r.label}: one-vs-rest κ = {r.kappa_after:.2f} < {kappa_gate}"
                for r in label_kappa.itertuples() if r.label in finals and r.kappa_after == r.kappa_after
                and r.kappa_after < kappa_gate]
    exclusions = labels.loc[labels["exclude_flag"] == "yes", "exclude_reason"].value_counts().rename_axis(
        "exclude_reason").reset_index(name="n")
    return Finalized(labels=labels, merges=mapping, counts=counts, field_kappa=pd.DataFrame(field_rows),
                     label_kappa=label_kappa, gate_failures=failures, exclusions=exclusions)
