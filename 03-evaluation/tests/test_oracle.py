"""P-Oracle from the pre-generated grid, and the rated-set candidates (analysis plan v1.4)."""

import json

import pandas as pd
import pytest

from payo_eval.oracle import grid_condition, select_oracle
from payo_eval.sampling import rating_candidates
from payo_llm.schema import load_schema


def write_grid(folder, model, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"replies_{model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def gold_csv(path, rows):
    cols = ["post_id", "exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def grid_rows(pid, labels, model):
    return [{"post_id": pid, "model": model, "task": "advice-oracle-grid", "condition": grid_condition(e, c),
             "seed": 42, "raw_output": f"{pid}:{e}:{c}", "error": None} for e, c in labels]


def test_select_oracle_picks_the_gold_pair(tmp_path):
    grid, advice = tmp_path / "advice" / "oracle_grid", tmp_path / "advice"
    labels = [("tampo", "hiya"), ("tampo", "none"), ("worry", "none")]
    for model in ("gemma", "sealion"):
        write_grid(grid, model, grid_rows("p1", labels, model) + grid_rows("p2", labels, model)
                   + grid_rows("p3", labels, model))
    gold_csv(tmp_path / "gold.csv", [["p1", "no", "", "tampo", ""], ["p2", "no", "", "worry", "none"],
                                     ["p3", "yes", "self_harm_crisis", "", ""]])
    written = select_oracle(grid, advice, tmp_path / "gold.csv", {"p1", "p2", "p3"}, load_schema())
    assert written == {"gemma": 2, "sealion": 2}
    out = [json.loads(line) for line in (advice / "replies_gemma_oracle.jsonl").read_text().splitlines()]
    assert {r["post_id"]: r["raw_output"] for r in out} == {"p1": "p1:tampo:none", "p2": "p2:worry:none"}
    assert {r["task"] for r in out} == {"advice-oracle"} and {r["condition"] for r in out} == {"P-Oracle"}


def test_select_oracle_stops_on_a_missing_pair(tmp_path):
    grid = tmp_path / "advice" / "oracle_grid"
    for model in ("gemma", "sealion"):
        write_grid(grid, model, grid_rows("p1", [("worry", "none")], model))
    gold_csv(tmp_path / "gold.csv", [["p1", "no", "", "tampo", "hiya"]])
    with pytest.raises(SystemExit, match="no successful reply"):
        select_oracle(grid, tmp_path / "advice", tmp_path / "gold.csv", {"p1"}, load_schema())


def test_rating_candidates_size_strata_and_determinism():
    sample = pd.DataFrame({"post_id": [f"p{i:03d}" for i in range(236)],
                           "stratum": ["culture/family"] * 100 + ["culture/romantic"] * 58 + ["control/other"] * 78})
    drawn = rating_candidates(sample, 71, seed=5)
    assert len(drawn) == 71 and drawn["post_id"].is_unique
    assert drawn["candidate_rank"].tolist() == list(range(1, 72))
    share = drawn["stratum"].value_counts()
    for stratum, n in {"culture/family": 100, "culture/romantic": 58, "control/other": 78}.items():
        assert abs(share[stratum] - 71 * n / 236) <= 1                # proportional (largest remainder)
    assert drawn.equals(rating_candidates(sample, 71, seed=5))
