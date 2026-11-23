"""The whole workflow on synthetic data (no notebooks; `workflow.py smoke` also runs those)."""

import os

import pandas as pd

from payo_eval.synthetic import smoke


def test_smoke_chain(tmp_path):
    before = dict(os.environ)
    try:
        base = smoke(tmp_path / "run")
        gold = base / "gold"
        adjudicated = pd.read_csv(gold / "emotion-recognition" / "annotations_adjudicated.csv", dtype=str)
        key = pd.read_csv(gold / "author-ratings" / "items_key.csv", dtype=str)
        ratings = pd.read_csv(gold / "author-ratings" / "ratings.csv")
        assert {"emotion_final", "emotion_type"} <= set(adjudicated.columns)
        assert key["post_id"].nunique() == 80 and len(key) == 400
        assert set(ratings["item_id"]) == set(key["item_id"])
        assert (base / "pilot" / "pilot_annotations.csv").exists()
    finally:
        os.environ.clear()
        os.environ.update(before)
