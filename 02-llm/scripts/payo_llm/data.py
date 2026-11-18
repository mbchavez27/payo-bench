"""Load posts (silver layer) and adjudicated gold labels."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from payo_llm import ROOT
from payo_llm.schema import Schema

SILVER_PATH = ROOT / "01-preprocessing" / "silver" / "advice_pairs.parquet"


@dataclass(frozen=True)
class Post:
    """The parts of a silver post that the prompts use."""

    post_id: str
    problem_goal: str
    context: str
    previous_attempts: str | None


@dataclass(frozen=True)
class GoldLabel:
    """The adjudicated emotion and cultural concept for one post (P-Oracle input)."""

    emotion: str
    cultural_concept: str


def read_post_ids(path: Path) -> list[str]:
    """
    Read the ordered list of post ids for a run.

    Args:
        path: CSV with a `post_id` column.

    Returns:
        Post ids in file order, without duplicates.
    """
    ids = pd.read_csv(path, dtype={"post_id": str})["post_id"].dropna().str.strip()
    if ids.duplicated().any():
        raise ValueError(f"{path}: duplicate post_id values")
    return ids.tolist()


def load_posts(post_ids: list[str], silver_path: Path = SILVER_PATH) -> list[Post]:
    """
    Fetch posts from the silver table, keeping the order of `post_ids`.

    Args:
        post_ids: Ids to load; every id must exist in silver.
        silver_path: Path to advice_pairs.parquet.

    Returns:
        One Post per id.
    """
    table = pd.read_parquet(silver_path).set_index("post_id")
    missing = [pid for pid in post_ids if pid not in table.index]
    if missing:
        raise KeyError(f"{len(missing)} post ids not in silver, e.g. {missing[:3]}")
    rows = table.loc[post_ids]
    return [
        Post(
            post_id=pid,
            problem_goal=row["problem_goal"],
            context=row["context"],
            previous_attempts=None if pd.isna(row["previous_attempts"]) else row["previous_attempts"],
        )
        for pid, row in rows.iterrows()
    ]


def load_gold(path: Path, schema: Schema) -> dict[str, GoldLabel]:
    """
    Load adjudicated labels for the P-Oracle condition.

    Rows that are excluded, or have no emotion yet, are skipped. Every label must be a schema id,
    so a typo in the sheet stops the run instead of reaching a prompt.

    Args:
        path: CSV with the annotation-template columns.
        schema: The label schema, for validating ids.

    Returns:
        Gold labels keyed by post id.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"P-Oracle needs adjudicated gold labels at {path}. "
            "Annotate the posts (02-llm/codebook/codebook.md) before running advice-oracle."
        )
    sheet = pd.read_csv(path, dtype=str).fillna("")
    sheet = sheet[(sheet["exclude_flag"].str.lower() != "yes") & (sheet["primary_emotion"] != "")]
    gold: dict[str, GoldLabel] = {}
    for row in sheet.itertuples():
        emotion = row.primary_emotion.strip().lower()
        concept = (row.cultural_concept.strip().lower() or "none")
        if emotion not in schema.emotion_ids:
            raise ValueError(f"{path}: post {row.post_id} has unknown emotion {emotion!r}")
        if concept not in schema.concept_ids:
            raise ValueError(f"{path}: post {row.post_id} has unknown concept {concept!r}")
        gold[row.post_id.strip()] = GoldLabel(emotion=emotion, cultural_concept=concept)
    return gold
