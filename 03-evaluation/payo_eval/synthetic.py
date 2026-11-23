"""
A synthetic version of the whole study, for the smoke test and the tests. Never used for results.

`smoke()` builds a fake silver table, runs every workflow step on it with the sheets filled in by code
(standing in for the team), writes fake model outputs, and runs the analysis notebooks, all inside one
folder. It proves the chain works end to end before any real data exists.
"""

import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from payo_llm.schema import load_schema
from payo_eval import steps
from payo_eval.paths import EVAL_DIR, get_paths, load_config
from payo_eval.sheets import MAIN_SHEET

SEED = 7
EMOTION_WEIGHTS = {"hiya": .10, "tampo": .06, "sama_ng_loob": .08, "pikon": .02, "utang_na_loob_guilt": .08,
                   "pakikisama_strain": .03, "worry": .18, "sadness": .15, "anger": .07, "fear": .03,
                   "burnout": .08, "jealousy": .05, "none_practical": .07}
DOMAINS = ("family", "romantic", "friendship", "work_school", "money", "other")
CONCEPTS = ("utang_na_loob", "hiya", "pakikisama", "amor_propio", "family_obligation", "respect_for_elders",
            "bahala_na")
WORDS = "ang kasi po talaga yung family ko sobrang hirap help please work pera bahay mama papa".split()


def fake_silver(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Silver-shaped posts with random Taglish-ish filler text."""
    def text(k: int) -> str:
        return " ".join(rng.choice(WORDS, size=k))
    rows = []
    for i in range(n):
        rows.append({"post_id": f"s{i:04d}", "flair": "Advice", "problem_goal": text(rng.integers(10, 40)),
                     "context": text(rng.integers(40, 200)),
                     "previous_attempts": text(rng.integers(5, 30)) if rng.random() < .6 else None,
                     "created_at": "2025-01-01", "reply_comment_id": f"c{i:04d}",
                     "reply_body": text(rng.integers(12, 90)) + ("\n\nEdit: thanks!" if rng.random() < .1 else ""),
                     "reply_score": int(rng.integers(1, 300))})
    return pd.DataFrame(rows)


def truth_table(ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """The 'real' labels the fake team converges on."""
    schema = load_schema()
    reasons = [r if isinstance(r, str) else r["id"] for r in schema.raw["exclusion_reasons"]]
    emotions, weights = list(EMOTION_WEIGHTS), np.array(list(EMOTION_WEIGHTS.values()))
    rows = []
    for pid in ids:
        excluded = rng.random() < .05
        emotion = rng.choice(emotions, p=weights / weights.sum())
        culture = schema.emotion(emotion)["type"] == "culture_bound"
        rows.append({"post_id": pid, "exclude_flag": "yes" if excluded else "no",
                     "exclude_reason": rng.choice(reasons) if excluded else "",
                     "primary_emotion": "" if excluded else emotion,
                     "cultural_concept": "" if excluded else (rng.choice(CONCEPTS) if culture or rng.random() < .2 else "none"),
                     "expression": "" if excluded else rng.choice(["explicit", "implicit"]),
                     "domain": "" if excluded else rng.choice(DOMAINS),
                     "need_type": "" if excluded else rng.choice(["emotional", "practical", "both"])})
    return pd.DataFrame(rows).set_index("post_id")


def fill_sheet(path: Path, fill: Callable[[dict], dict]) -> None:
    """Write values into a sheet's input columns, the way a person would."""
    book = load_workbook(path)
    sheet = book[MAIN_SHEET]
    header = [c.value for c in sheet[1]]
    for row in sheet.iter_rows(min_row=2):
        values = {h: c.value for h, c in zip(header, row, strict=True)}
        for column, value in fill(values).items():
            row[header.index(column)].value = value
    book.save(path)


def annotate(truth: pd.DataFrame, rng: np.random.Generator, agreement: float = .85) -> Callable[[dict], dict]:
    """An annotator who matches the truth most of the time."""
    emotions = list(EMOTION_WEIGHTS)

    def fill(row: dict) -> dict:
        t = truth.loc[row["post_id"]].to_dict()
        out = dict(t)
        if t["exclude_flag"] == "no" and rng.random() > agreement:
            out["primary_emotion"] = rng.choice(emotions)
        if t["exclude_flag"] == "no" and rng.random() > .9:
            out["expression"] = "implicit" if t["expression"] == "explicit" else "explicit"
        return out
    return fill


def adjudicate_fill(truth: pd.DataFrame) -> Callable[[dict], dict]:
    """The adjudicator restores the truth on every field."""
    return lambda row: {f"final_{k}": v for k, v in truth.loc[row["post_id"]].to_dict().items()}


def prescreen_fill(truth: pd.DataFrame, rng: np.random.Generator) -> Callable[[dict], dict]:
    """A quick screener: mostly right on exclusion and domain, noisy on concept_likely."""
    def fill(row: dict) -> dict:
        t = truth.loc[row["post_id"]]
        if t["exclude_flag"] == "yes" and rng.random() < .7:
            return {"exclude_flag": "yes", "exclude_reason": t["exclude_reason"]}
        concept = t["cultural_concept"] not in ("", "none")
        likely = concept if rng.random() < .8 else not concept
        domain = t["domain"] if t["domain"] and rng.random() < .8 else rng.choice(DOMAINS)
        return {"exclude_flag": "no", "concept_likely": "yes" if likely else "no", "domain_guess": domain}
    return fill


def write_model_outputs(gold: Path, sample: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Fake Task A and Task B outputs (and variance runs) in the run.py layout."""
    schema = load_schema()
    emotions = list(EMOTION_WEIGHTS)
    correct: dict[tuple[str, str], bool] = {}
    (gold / "emotion-recognition").mkdir(parents=True, exist_ok=True)
    (gold / "advice-generation").mkdir(parents=True, exist_ok=True)
    (gold / "variance").mkdir(parents=True, exist_ok=True)

    def recognition(pid: str, model: str, seed: int) -> dict:
        t = truth.loc[pid]
        emotion = t["primary_emotion"] or "worry"
        cb = schema.emotion(emotion)["type"] == "culture_bound"
        p = {"gemma": .45 if cb else .65, "sealion": .62 if cb else .66}[model] - (.1 if t["expression"] == "implicit" else 0)
        right = rng.random() < p
        guess = emotion if right else rng.choice([e for e in emotions if e != emotion])
        valid = rng.random() > .02
        alias = rng.random() < .05
        return {"post_id": pid, "model": model, "task": "recognition", "condition": "none", "seed": seed,
                "emotion": guess if valid else None, "cultural_concept": t["cultural_concept"] or "none",
                "confidence": int(rng.choice([85, 90, 95])), "valid": valid,
                "invalid_reason": None if valid else "unknown_emotion", "concept_alias": alias,
                "raw_output": "{}", "n_prompt_tokens": 1500, "truncated": False, "error": None, "latency_s": 1.0}

    def advice(pid: str, model: str, condition: str, seed: int) -> dict:
        words = int(rng.integers(120, 190))
        return {"post_id": pid, "model": model, "task": "advice-p0" if condition == "P0" else "advice-oracle",
                "condition": condition, "seed": seed, "raw_output": " ".join(rng.choice(WORDS, size=words)),
                "reply_words": words, "n_prompt_tokens": 900 if condition == "P0" else 1050, "truncated": False,
                "error": None, "latency_s": 8.0}

    kept = [p for p in sample["post_id"] if truth.loc[p, "exclude_flag"] == "no"]
    for model in ("gemma", "sealion"):
        recs = [recognition(p, model, 42) for p in sample["post_id"]]
        for r in recs:
            correct[(r["post_id"], model)] = r["emotion"] == truth.loc[r["post_id"], "primary_emotion"]
        _jsonl(gold / "emotion-recognition" / f"predictions_{model}.jsonl", recs)
        _jsonl(gold / "advice-generation" / f"replies_{model}_p0.jsonl", [advice(p, model, "P0", 42) for p in sample["post_id"]])
        _jsonl(gold / "advice-generation" / f"replies_{model}_oracle.jsonl", [advice(p, model, "P-Oracle", 42) for p in kept])
        for seed in (1, 2, 3):
            first = list(sample["post_id"][:30])
            _jsonl(gold / "variance" / f"emotion-recognition_predictions_{model}_seed{seed}.jsonl",
                   [recognition(p, model, seed) for p in first])
            _jsonl(gold / "variance" / f"advice-generation_replies_{model}_p0_seed{seed}.jsonl",
                   [advice(p, model, "P0", seed) for p in first])
    return correct


def _jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def rate(key: pd.DataFrame, truth: pd.DataFrame, correct: dict, rater: str, rng: np.random.Generator) -> Callable[[dict], dict]:
    """A rater whose scores follow the hypotheses, plus noise and a personal offset."""
    schema = load_schema()
    key = key.set_index("item_id")
    base = {"community": (3.6, 3.8, 3.0, 3.8), "gemma": (3.4, 3.0, 3.5, 3.2), "sealion": (3.5, 3.3, 3.5, 3.6)}
    offset = rng.normal(0, .2)
    risky = {p for p in key["post_id"].unique() if rng.random() < .25}

    def fill(row: dict) -> dict:
        item = key.loc[row["item_id"]]
        emotion = truth.loc[item["post_id"], "primary_emotion"]
        cb = schema.emotion(emotion)["type"] == "culture_bound"
        means = np.array(base[item["source"]], dtype=float) + offset
        if item["condition"] == "P-Oracle":
            means += .3 + (.2 if cb else 0)
        if item["source"] != "community" and correct.get((item["post_id"], item["source"])):
            means += .2
        scores = np.clip(np.rint(means + rng.normal(0, .7, 4)), 1, 5).astype(int)
        if item["post_id"] in risky:
            escalation = "appropriate" if rng.random() < .7 else "missing"
        else:
            escalation = "excessive" if rng.random() < .03 else "not_needed"
        p_auto = .3 if item["source"] == "community" else .5
        strategy = rng.choice(["autonomy", "relational", "neutral_practical"], p=[p_auto, (1 - p_auto) * .6, (1 - p_auto) * .4])
        return {"empathy": int(scores[0]), "cultural_fit": int(scores[1]), "helpfulness": int(scores[2]),
                "register_tone": int(scores[3]), "safe_escalation": escalation,
                "harm_flags": "none" if rng.random() < .95 else "harmful_advice", "strategy": strategy}
    return fill


def synthetic_config(path: Path) -> None:
    """The real config with a fake roster and a decided pakikisama_strain merge."""
    text = (EVAL_DIR / "config.toml").read_text(encoding="utf-8")
    text = re.sub(r'^annotators = .*$', 'annotators = ["ann1", "ann2", "ann3"]', text, flags=re.M)
    text = re.sub(r'^raters = .*$', 'raters = ["ann1", "ann2", "ann3"]', text, flags=re.M)
    text = re.sub(r'^adjudicator = .*$', 'adjudicator = "adj"', text, flags=re.M)
    text = re.sub(r'^screener = .*$', 'screener = "scr"', text, flags=re.M)
    text = re.sub(r'^n = 900', 'n = 600', text, flags=re.M)
    text = re.sub(r'^pakikisama_strain = ""', 'pakikisama_strain = "hiya"', text, flags=re.M)
    text = text.replace("[merges]\n", '[merges]\nburnout = "worry"\njealousy = "anger"\nsama_ng_loob = "hiya"\n'
                        'utang_na_loob_guilt = "hiya"\nanger = "sadness"\nnone_practical = "worry"\n', 1)
    text = re.sub(r'^bootstrap_b = \d+', 'bootstrap_b = 200', text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def smoke(folder: Path | None, run_notebooks: Callable[[Path], None] | None = None) -> Path:
    """
    Run the whole workflow on synthetic data inside `folder` (a new temp folder if None).

    Args:
        folder: Where to put everything; must be empty or not exist.
        run_notebooks: Called with the notebook output folder once the gold layer is complete.

    Returns:
        The folder used.
    """
    base = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="payo-smoke-"))
    if base.exists() and any(base.iterdir()):
        raise SystemExit(f"{base} is not empty.")
    base.mkdir(parents=True, exist_ok=True)
    env = {"PAYO_GOLD_DIR": base / "gold", "PAYO_WORK_DIR": base / "work", "PAYO_SILVER_PATH": base / "silver.parquet",
           "PAYO_RESULTS_DIR": base / "results", "PAYO_FIGURES_DIR": base / "figures",
           "PAYO_EVAL_CONFIG": base / "config.toml", "PAYO_PILOT_DIR": base / "pilot",
           "PAYO_PILOT_POSTS": base / "pilot" / "pilot_posts.csv"}
    for key, value in env.items():
        os.environ[key] = str(value)
    rng = np.random.default_rng(SEED)
    silver = fake_silver(700, rng)
    silver.to_parquet(env["PAYO_SILVER_PATH"], index=False)
    synthetic_config(env["PAYO_EVAL_CONFIG"])
    (base / "pilot").mkdir()
    silver[["post_id"]].iloc[:20].to_csv(env["PAYO_PILOT_POSTS"], index=False)
    truth = truth_table(silver["post_id"].tolist(), rng)
    paths, cfg = get_paths(), load_config()
    people = cfg["roster"]["annotators"]

    print("== pilot")
    steps.pilot_sheets()
    for person in people:
        fill_sheet(paths.sheet("pilot", person), annotate(truth, rng))
    steps.pilot_merge()
    if paths.sheet("pilot", "adjudication").exists():
        fill_sheet(paths.sheet("pilot", "adjudication"), adjudicate_fill(truth))
    steps.pilot_adjudicate()

    print("== pre-screen and sample")
    steps.prescreen_sheet()
    fill_sheet(paths.sheet("prescreen", cfg["roster"]["screener"]), prescreen_fill(truth, rng))
    steps.sample()
    for person in people:
        fill_sheet(paths.sheet("annotation", person), annotate(truth, rng))

    print("== adjudication and finalization")
    steps.adjudication_sheet()
    if paths.sheet("annotation", "adjudication").exists():
        fill_sheet(paths.sheet("annotation", "adjudication"), adjudicate_fill(truth))
    steps.finalize_labels(accept_kappa=True)

    print("== fake full run and rating")
    sample = pd.read_csv(paths.sample, dtype={"post_id": str})
    correct = write_model_outputs(paths.gold, sample, truth, rng)
    steps.rating_sheets()
    key = pd.read_csv(paths.items_key, dtype=str)
    for rater in cfg["roster"]["raters"]:
        if paths.sheet("rating", rater).exists():
            fill_sheet(paths.sheet("rating", rater), rate(key, truth, correct, rater, rng))
            fill_sheet(paths.sheet("rating", f"{rater}_bestfit"), lambda row: {"best": rng.choice(["A", "B", "C"])})
    steps.collect_rating_sheets()
    steps.status()

    if run_notebooks:
        print("== notebooks")
        run_notebooks(base / "notebooks")
    print(f"\nSmoke run complete in {base}")
    return base
