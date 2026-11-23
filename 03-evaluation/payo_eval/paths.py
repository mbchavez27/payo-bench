"""
Where every file lives, and the workflow config.

Each location can be redirected with an environment variable, so the full chain can run on synthetic
data (`workflow.py smoke`) without touching the real gold layer.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from payo_llm import ROOT
from payo_llm.data import SILVER_PATH as REAL_SILVER

EVAL_DIR = ROOT / "03-evaluation"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    """All file locations for one environment (real or synthetic)."""

    gold: Path
    work: Path
    silver: Path
    results: Path
    figures: Path
    config: Path
    pilot_posts: Path
    pilot_dir: Path

    # Gold layer: ids and labels only (committable); model outputs are JSONL (gitignored)
    @property
    def sample(self) -> Path:
        return self.gold / "sample_posts.csv"

    @property
    def er(self) -> Path:
        return self.gold / "emotion-recognition"

    @property
    def prescreen(self) -> Path:
        return self.er / "prescreen.csv"

    @property
    def assignment(self) -> Path:
        return self.er / "annotation_assignment.csv"

    @property
    def annotations_raw(self) -> Path:
        return self.er / "annotations_raw.csv"

    @property
    def adjudication(self) -> Path:
        return self.er / "adjudication.csv"

    @property
    def adjudicated(self) -> Path:
        return self.er / "annotations_adjudicated.csv"

    @property
    def advice(self) -> Path:
        return self.gold / "advice-generation"

    @property
    def ratings_dir(self) -> Path:
        return self.gold / "author-ratings"

    @property
    def items_key(self) -> Path:
        return self.ratings_dir / "items_key.csv"

    @property
    def bestfit_key(self) -> Path:
        return self.ratings_dir / "bestfit_key.csv"

    @property
    def rating_assignment(self) -> Path:
        return self.ratings_dir / "rating_assignment.csv"

    @property
    def ratings(self) -> Path:
        return self.ratings_dir / "ratings.csv"

    @property
    def bestfit(self) -> Path:
        return self.ratings_dir / "bestfit.csv"

    # Working sheets for people: contain post text, so they stay out of git
    def sheet(self, step: str, name: str) -> Path:
        return self.work / step / f"{name}.xlsx"


def get_paths() -> Paths:
    """Resolve paths, honouring the PAYO_* overrides."""
    gold = _env_path("PAYO_GOLD_DIR", ROOT / "02-llm" / "gold")
    return Paths(
        gold=gold,
        work=_env_path("PAYO_WORK_DIR", ROOT / "02-llm" / "work"),
        silver=_env_path("PAYO_SILVER_PATH", REAL_SILVER),
        results=_env_path("PAYO_RESULTS_DIR", EVAL_DIR / "results"),
        figures=_env_path("PAYO_FIGURES_DIR", EVAL_DIR / "figures"),
        config=_env_path("PAYO_EVAL_CONFIG", EVAL_DIR / "config.toml"),
        pilot_posts=_env_path("PAYO_PILOT_POSTS", ROOT / "02-llm" / "pilot" / "pilot_posts.csv"),
        pilot_dir=_env_path("PAYO_PILOT_DIR", ROOT / "02-llm" / "pilot"),
    )


def load_config(paths: Paths | None = None) -> dict[str, Any]:
    """Read config.toml."""
    paths = paths or get_paths()
    return tomllib.loads(paths.config.read_text(encoding="utf-8"))


def roster(cfg: dict[str, Any], role: str) -> list[str]:
    """
    Return a roster list from the config, or stop with a clear message if it is empty.

    Args:
        cfg: The loaded config.
        role: "annotators" or "raters".

    Returns:
        The ids, in config order.
    """
    names = [str(n).strip() for n in cfg["roster"].get(role, []) if str(n).strip()]
    if not names:
        raise SystemExit(f"config.toml [roster] {role} is empty: add the team's ids first.")
    if len(set(names)) != len(names):
        raise SystemExit(f"config.toml [roster] {role} has duplicate ids.")
    return names
