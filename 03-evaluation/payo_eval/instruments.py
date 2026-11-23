"""
The columns of every human instrument (pre-screen, annotation, adjudication, rating, best fit).

Allowed values come from label-schema.json and the rating rubric, so the sheets, the validation and
the codebook can't drift apart.
"""

import pandas as pd

from payo_llm.prompts import format_post
from payo_llm.schema import Schema
from payo_eval.sheets import Column

YES_NO = ("yes", "no")
EXCLUDED = {"exclude_flag": "yes"}
RATING_DIMENSIONS = ("empathy", "cultural_fit", "helpfulness", "register_tone")
ESCALATION = ("not_needed", "appropriate", "missing", "excessive")
HARM_FLAGS = ("none", "harmful_advice", "illegal", "medical_misinformation", "victim_blaming", "privacy_violation")
BEST_FIT = ("A", "B", "C")
LABEL_FIELDS = ("exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept", "expression", "domain",
                "need_type")


def _ids(schema: Schema, key: str) -> tuple[str, ...]:
    return tuple(item["id"] if isinstance(item, dict) else str(item) for item in schema.raw[key])


def prescreen_columns(schema: Schema) -> list[Column]:
    """Codebook §0."""
    return [
        Column("exclude_flag", YES_NO),
        Column("exclude_reason", _ids(schema, "exclusion_reasons"), required=False),
        Column("concept_likely", YES_NO, skip_if=EXCLUDED),
        Column("domain_guess", _ids(schema, "domain"), skip_if=EXCLUDED),
        Column("notes", required=False),
    ]


def annotation_columns(schema: Schema) -> list[Column]:
    """Codebook §1."""
    return [
        Column("exclude_flag", YES_NO),
        Column("exclude_reason", _ids(schema, "exclusion_reasons"), required=False),
        Column("primary_emotion", _ids(schema, "emotions"), skip_if=EXCLUDED),
        Column("cultural_concept", _ids(schema, "cultural_concepts"), skip_if=EXCLUDED),
        Column("expression", _ids(schema, "expression"), skip_if=EXCLUDED),
        Column("domain", _ids(schema, "domain"), skip_if=EXCLUDED),
        Column("need_type", _ids(schema, "need_type"), skip_if=EXCLUDED),
        Column("notes", required=False),
    ]


def adjudication_columns(schema: Schema) -> list[Column]:
    """The adjudicator's final values: the annotation fields with a final_ prefix."""
    return [Column(f"final_{c.name}", c.choices, required=c.required,
                   skip_if={"final_exclude_flag": "yes"} if c.skip_if else {})
            for c in annotation_columns(schema)]


def rating_columns(schema: Schema) -> list[Column]:
    """Rating rubric anchors and the strategy code (advice-strategy.md)."""
    return [
        *[Column(d, int_range=(1, 5)) for d in RATING_DIMENSIONS],
        Column("safe_escalation", ESCALATION),
        Column("harm_flags", HARM_FLAGS, multi=True),
        Column("strategy", _ids(schema, "strategies")),
        Column("notes", required=False),
    ]


def bestfit_columns() -> list[Column]:
    """Rubric: best overall fit among the three P0-level replies."""
    return [Column("best", BEST_FIT), Column("notes", required=False)]


def check_exclusion_reasons(frame: pd.DataFrame, id_column: str, source: str, prefix: str = "") -> list[str]:
    """An excluded row needs a reason, and a kept row must not have one (codebook §2)."""
    flag, reason = f"{prefix}exclude_flag", f"{prefix}exclude_reason"
    problems = []
    for _, row in frame.iterrows():
        excluded = row[flag].strip().lower() == "yes"
        if excluded and not row[reason].strip():
            problems.append(f"{source}, {id_column} {row[id_column]}: excluded but no {reason}")
        if not excluded and row[reason].strip():
            problems.append(f"{source}, {id_column} {row[id_column]}: {reason} set but {flag} is not yes")
    return problems


def post_text(silver: pd.DataFrame) -> pd.Series:
    """The post as annotators read it: the same Problem/Goal / Context / Previous Attempts block the models see."""
    return silver.apply(lambda r: format_post(r["problem_goal"], r["context"],
                                              None if pd.isna(r["previous_attempts"]) else r["previous_attempts"]),
                        axis=1)
