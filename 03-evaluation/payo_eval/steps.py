"""
The workflow steps behind `03-evaluation/workflow.py`. Each step is one gate between two human tasks.

Every step checks its inputs, refuses to overwrite work that people may already have started on, and
prints what to do next.
"""

import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from payo_llm.data import read_post_ids
from payo_llm.pilot import load_outputs
from payo_llm.schema import load_schema
from payo_eval import load
from payo_eval.annotation import collect, disagreements, finalize, read_decisions, adjudicate
from payo_eval.instruments import (LABEL_FIELDS, adjudication_columns, annotation_columns, bestfit_columns,
                                   check_exclusion_reasons, post_text, prescreen_columns, rating_columns)
from payo_eval.paths import Paths, get_paths, load_config, roster
from payo_eval.rating import bestfit_key, build_items, collect_ratings, rater_frames
from payo_eval.oracle import select_oracle
from payo_eval.sampling import (assign, draw_prescreen_pool, keyword_prescreen, n_to_draw, rated_set,
                                rating_candidates, stratified_sample)
from payo_eval.sheets import normalize, read_sheet, validate, write_sheet
from payo_eval.report import finalization_report

ADJ_FIELDS_TEXT = list(LABEL_FIELDS)


def _context() -> tuple[Paths, dict, object, pd.DataFrame]:
    paths = get_paths()
    cfg = load_config(paths)
    silver = pd.read_parquet(paths.silver)
    silver["post_id"] = silver["post_id"].astype(str)
    return paths, cfg, load_schema(), silver.set_index("post_id", drop=False)


def _refuse_if_exists(*targets, force: bool = False) -> None:
    existing = [str(t) for t in targets if t.exists()]
    if existing and not force:
        raise SystemExit("Already done (would overwrite): " + ", ".join(existing) +
                         "\nRerun with --force only if nobody has started the next human step.")


def _post_frame(silver: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    rows = silver.loc[ids]
    return pd.DataFrame({"post_id": ids, "post": post_text(rows).to_numpy()})


def _write_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _shuffled(ids: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.permutation(len(ids))]


def _seed_for(name: str, base: int) -> int:
    return base + sum(ord(ch) for ch in name)


# ---------------------------------------------------------------- pilot

def pilot_sheets(force: bool = False) -> None:
    """Every annotator gets the same pilot posts (codebook §8 step 1)."""
    paths, cfg, schema, silver = _context()
    ids = read_post_ids(paths.pilot_posts)
    people = roster(cfg, "annotators")
    for person in people:
        frame = _post_frame(silver, _shuffled(ids, _seed_for(person, cfg["seed"])))
        prefill = pd.DataFrame({"exclude_flag": ["no"] * len(frame)})
        path = write_sheet(paths.sheet("pilot", person), frame, annotation_columns(schema), ["post"], prefill, force)
        print(f"Wrote {path}")
    print("Next: each annotator labels their sheet with 02-llm/codebook/codebook.md, then run `pilot-merge`.")


def pilot_merge(force: bool = False) -> None:
    """Stack the pilot sheets for pilot_report.py and write the adjudication sheet for the disagreements."""
    paths, cfg, schema, silver = _context()
    sheets = {p: read_sheet(paths.sheet("pilot", p)) for p in roster(cfg, "annotators")}
    raw = collect(sheets, schema)
    _write_csv(raw, paths.pilot_dir / "pilot_annotations_raw.csv")
    _adjudication_sheet(paths.sheet("pilot", "adjudication"), raw, silver, schema, force)
    print(f"Wrote {paths.pilot_dir / 'pilot_annotations_raw.csv'}\n"
          "Next: `uv run python 02-llm/scripts/pilot_report.py --annotations 02-llm/pilot/pilot_annotations_raw.csv`,"
          " discuss, fill the adjudication sheet, then run `pilot-adjudicate`.")


def pilot_adjudicate() -> None:
    """One agreed row per pilot post in pilot_annotations.csv (the pilot P-Oracle input)."""
    paths, _, schema, _ = _context()
    raw = pd.read_csv(paths.pilot_dir / "pilot_annotations_raw.csv", dtype=str).fillna("")
    labels = adjudicate(raw, _decisions(paths.sheet("pilot", "adjudication"), raw, schema))
    out = labels.assign(annotator="adjudicated", notes=labels["label_source"])
    _write_csv(out[["post_id", "annotator", *LABEL_FIELDS, "notes"]], paths.pilot_dir / "pilot_annotations.csv")
    print(f"Wrote {paths.pilot_dir / 'pilot_annotations.csv'}\n"
          "Next: `ollama serve`, then `uv run python 02-llm/scripts/run.py --profile pilot --task advice-oracle --model both`.")


def _adjudication_sheet(path, raw: pd.DataFrame, silver: pd.DataFrame, schema, force: bool) -> int:
    table = disagreements(raw)
    if table.empty:
        print("No disagreements: nothing to adjudicate.")
        return 0
    frame = _post_frame(silver, table["post_id"].tolist())
    for field in ADJ_FIELDS_TEXT:
        frame[field] = table[field].to_numpy()
    prefill = pd.DataFrame({f"final_{f}": table[f"agreed_{f}"].to_numpy() for f in LABEL_FIELDS})
    write_sheet(path, frame, adjudication_columns(schema), ["post"], prefill, force)
    print(f"Wrote {path}: {len(table)} posts to adjudicate (agreed fields are prefilled).")
    return len(table)


def _decisions(path, raw: pd.DataFrame, schema) -> pd.DataFrame | None:
    if disagreements(raw).empty:
        return None
    return read_decisions(read_sheet(path), schema)


# ---------------------------------------------------------------- A. pre-screen

def prescreen_sheet(force: bool = False) -> None:
    """Draw the pre-screen pool (pilot posts excluded) and write the screener's sheet (manual method only)."""
    paths, cfg, schema, silver = _context()
    if _prescreen_method(cfg) == "keywords":
        raise SystemExit('[prescreen] method = "keywords": no sheet is needed, run `sample` directly. '
                         'Set method = "manual" in config.toml to pre-screen by hand.')
    exclude = set(read_post_ids(paths.pilot_posts)) if paths.pilot_posts.exists() else set()
    ids = draw_prescreen_pool(silver["post_id"].tolist(), exclude, cfg["prescreen"]["n"], cfg["seed"])
    screener = cfg["roster"].get("screener") or "screener"
    frame = _post_frame(silver, ids)
    prefill = pd.DataFrame({"exclude_flag": ["no"] * len(frame)})
    path = write_sheet(paths.sheet("prescreen", screener), frame, prescreen_columns(schema), ["post"], prefill, force)
    print(f"Wrote {path} ({len(ids)} posts).\nNext: pre-screen with codebook §0, then run `sample`.")


# ---------------------------------------------------------------- B. sample + annotation sheets

def _prescreen_method(cfg: dict) -> str:
    method = cfg["prescreen"].get("method", "manual")
    if method not in ("keywords", "manual"):
        raise SystemExit(f'[prescreen] method must be "keywords" or "manual", not {method!r}.')
    return method


def _screened(paths: Paths, cfg: dict, schema, silver: pd.DataFrame) -> pd.DataFrame:
    """The pre-screen table: keyword-tagged in code, or read back from the screener's sheet."""
    if _prescreen_method(cfg) == "keywords":
        exclude = set(read_post_ids(paths.pilot_posts)) if paths.pilot_posts.exists() else set()
        return keyword_prescreen(silver.reset_index(drop=True), exclude,
                                 cfg["prescreen"]["keywords"]).assign(screener="keywords")
    screener = cfg["roster"].get("screener") or "screener"
    sheet = read_sheet(paths.sheet("prescreen", screener))
    columns = prescreen_columns(schema)
    problems = validate(sheet, columns, "post_id", "prescreen") or check_exclusion_reasons(sheet, "post_id", "prescreen")
    if problems:
        raise SystemExit("Pre-screen sheet has problems:\n  " + "\n  ".join(problems))
    return normalize(sheet, columns)[["post_id", "exclude_flag", "exclude_reason", "concept_likely",
                                      "domain_guess", "notes"]].assign(screener=screener)


def sample(force: bool = False) -> None:
    """Stratified sample from the pre-screen (keywords or the screener's sheet), the assignment, and the sheets."""
    paths, cfg, schema, silver = _context()
    _refuse_if_exists(paths.sample, paths.assignment, force=force)
    screened = _screened(paths, cfg, schema, silver)
    _write_csv(screened, paths.prescreen)

    s = cfg["sample"]
    drawn = stratified_sample(screened, n_to_draw(s["n_target"], s["expected_retention"]), s["concept_share"],
                              s["domain_floor"], cfg["seed"])
    words = (silver["problem_goal"].fillna("") + " " + silver["context"].fillna("") + " " +
             silver["previous_attempts"].fillna("")).str.split().str.len()
    drawn["post_length_words"] = drawn["post_id"].map(words).astype(int)
    _write_csv(drawn, paths.sample)

    people = roster(cfg, "annotators")
    adjudicator = cfg["roster"].get("adjudicator", "")
    plan = assign(drawn["post_id"].tolist(), people, s["double_share"], cfg["seed"],
                  not_on_doubles={adjudicator} if adjudicator else None)
    plan = plan.rename(columns={"person": "annotator"})
    _write_csv(plan[["post_id", "annotator", "slot"]], paths.assignment)
    for person in people:
        ids = _shuffled(plan.loc[plan["annotator"] == person, "post_id"].tolist(), _seed_for(person, cfg["seed"]))
        frame = _post_frame(silver, ids)
        prefill = pd.DataFrame({"exclude_flag": ["no"] * len(frame)})
        write_sheet(paths.sheet("annotation", person), frame, annotation_columns(schema), ["post"], prefill, force)
    print(drawn.groupby(["arm", "domain_guess"]).size().rename("posts").to_string())
    print(f"\nWrote {paths.sample} ({len(drawn)} posts), {paths.assignment} "
          f"({int((plan['slot'] == 2).sum())} double-annotated), and one sheet per annotator in {paths.work / 'annotation'}.\n"
          "Next: `rating-candidates`, then all GPU work (pipeline.py, RUNBOOK §2; nobody opens the outputs until\n"
          "`finalize-labels`). Annotation (RUNBOOK §3) can start any time, blind.")


# ---------------------------------------------------------------- B2. rated-set candidates

def rating_candidates_step(force: bool = False) -> None:
    """Draw the rated-set candidates now, so the P-Oracle grid can run before annotation (plan v1.4)."""
    paths, cfg, _, _ = _context()
    _refuse_if_exists(paths.rating_candidates, force=force)
    sample = pd.read_csv(paths.sample, dtype=str)
    n = n_to_draw(cfg["rating"]["n_eval"], cfg["sample"]["expected_retention"])
    drawn = rating_candidates(sample, n, cfg["seed"] + 11)
    _write_csv(drawn, paths.rating_candidates)
    print(drawn.groupby("stratum").size().rename("candidates").to_string())
    print(f"\nWrote {paths.rating_candidates} ({len(drawn)} candidates for {cfg['rating']['n_eval']} rated posts).\n"
          "Next: the GPU runs (pipeline.py), including the P-Oracle grid for these posts.")


def _select_oracle_if_grid(paths: Paths, schema) -> None:
    """After the labels are frozen: pick each candidate's P-Oracle reply from the pre-generated grid."""
    if not paths.oracle_grid.exists():
        print("No P-Oracle grid found: run `run.py --task advice-oracle` after this, or the grid, then rerun.")
        return
    candidates = set(pd.read_csv(paths.rating_candidates, dtype=str)["post_id"])
    written = select_oracle(paths.oracle_grid, paths.advice, paths.adjudicated, candidates, schema)
    print("P-Oracle replies selected from the grid: " + ", ".join(f"{m} {n}" for m, n in written.items()))


# ---------------------------------------------------------------- C. adjudication

def adjudication_sheet(force: bool = False) -> None:
    """Collect the annotation sheets and write the adjudicator's sheet (disagreements only)."""
    paths, cfg, schema, silver = _context()
    plan = pd.read_csv(paths.assignment, dtype=str).rename(columns={"annotator": "person"})
    sheets = {p: read_sheet(paths.sheet("annotation", p)) for p in sorted(plan["person"].unique())}
    raw = collect(sheets, schema, plan)
    _write_csv(raw, paths.annotations_raw)
    print(f"Wrote {paths.annotations_raw} ({len(raw)} rows).")
    n = _adjudication_sheet(paths.sheet("annotation", "adjudication"), raw, silver, schema, force)
    adjudicator = cfg["roster"].get("adjudicator") or "the adjudicator"
    print(f"Next: {adjudicator} fills the adjudication sheet, then run `finalize-labels`." if n
          else "Next: run `finalize-labels`.")


# ---------------------------------------------------------------- D. finalize

def finalize_labels(accept_kappa: bool = False) -> None:
    """Gold labels, merges and the κ gate (analysis plan §2), plus the report."""
    paths, cfg, schema, _ = _context()
    raw = pd.read_csv(paths.annotations_raw, dtype=str).fillna("")
    decisions = _decisions(paths.sheet("annotation", "adjudication"), raw, schema)
    if decisions is not None:
        _write_csv(decisions, paths.adjudication)
    result = finalize(raw, decisions, schema, cfg.get("merges", {}), cfg["labels"]["min_label"],
                      cfg["labels"]["kappa_gate"])
    paths.results.mkdir(parents=True, exist_ok=True)
    report = finalization_report(result, cfg, accepted=accept_kappa)
    (paths.results / "label_finalization.md").write_text(report, encoding="utf-8")
    print(report)
    if result.gate_failures and not accept_kappa:
        raise SystemExit("κ gate failed (analysis plan §2): revise and re-adjudicate those labels, add a merge, or "
                         "rerun with --accept-kappa and log the reason in analysis-plan.md §11.")
    cols = ["post_id", *LABEL_FIELDS, "label_source", "emotion_final", "emotion_type"]
    _write_csv(result.labels[cols], paths.adjudicated)
    (paths.er / "merges.json").write_text(json.dumps(result.merges, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {paths.adjudicated} and merges.json.")
    _select_oracle_if_grid(paths, schema)
    print("Next: `rating-sheets`.")


# ---------------------------------------------------------------- E. rating sheets

def rating_sheets(force: bool = False) -> None:
    """The rated set, blinded items, rater assignment and every rater's two sheets."""
    paths, cfg, schema, silver = _context()
    _refuse_if_exists(paths.items_key, paths.rating_assignment, force=force)
    gold = load.labels(paths)
    outputs = load_outputs(paths.gold)
    ok = outputs[(outputs["task"].isin(["advice-p0", "advice-oracle"])) & (outputs["error"].isna() | (outputs["error"] == ""))]
    replies = {(m, c): g[["post_id", "raw_output", "reply_words"]] for (m, c), g in ok.groupby(["model", "condition"])}
    have_all = set(gold["post_id"])
    for table in replies.values():
        have_all &= set(table["post_id"])
    if len(replies) < 4:
        raise SystemExit("The full run needs P0 and P-Oracle replies for both models before rating sheets can be made.")
    n_eval = cfg["rating"]["n_eval"]
    if paths.rating_candidates.exists():  # v1.4: the first n_eval candidates kept after annotation
        order = pd.read_csv(paths.rating_candidates, dtype=str)["post_id"]
        kept = gold.set_index("post_id")
        ids = [p for p in order if p in kept.index and p in have_all][:n_eval]
        if len(ids) < n_eval:
            print(f"Warning: only {len(ids)} of {len(order)} candidates survived annotation (wanted {n_eval}); "
                  "rating all of them. Log this in analysis-plan.md §11.")
        rated = kept.loc[ids].reset_index()
        rated["stratum"] = rated["emotion_type"] + "/" + rated["domain"]
    else:
        rated = rated_set(gold[gold["post_id"].isin(have_all)], n_eval, cfg["seed"])
    _write_csv(rated[["post_id", "emotion_type", "domain", "stratum"]], paths.ratings_dir / "rated_posts.csv")

    raw = pd.read_csv(paths.annotations_raw, dtype=str).fillna("")
    adjudicated = set(pd.read_csv(paths.adjudicated, dtype=str).query("label_source == 'adjudicated'")["post_id"])
    adjudicator = cfg["roster"].get("adjudicator", "")
    avoid = {pid: set(group["annotator"]) | ({adjudicator} if pid in adjudicated and adjudicator else set())
             for pid, group in raw.groupby("post_id")}
    plan = assign(rated["post_id"].tolist(), roster(cfg, "raters"), cfg["rating"]["double_share"], cfg["seed"] + 7, avoid)
    plan = plan.rename(columns={"person": "rater"})
    _write_csv(plan, paths.rating_assignment)

    community = silver.loc[rated["post_id"], ["post_id", "reply_body", "reply_score"]].reset_index(drop=True)
    key, texts = build_items(rated[["post_id"]], community, replies, cfg["seed"])
    bf_key = bestfit_key(key, cfg["seed"])
    _write_csv(key, paths.items_key)
    _write_csv(bf_key, paths.bestfit_key)
    posts = post_text(silver.loc[rated["post_id"]])
    for rater in sorted(plan["rater"].unique()):
        frame, best = rater_frames(rater, plan.rename(columns={"rater": "person"}), key, texts, posts, bf_key)
        write_sheet(paths.sheet("rating", rater), frame, rating_columns(schema), ["post", "reply"], overwrite=force)
        write_sheet(paths.sheet("rating", f"{rater}_bestfit"), best, bestfit_columns(),
                    ["post", "reply_A", "reply_B", "reply_C"], overwrite=force)
    conflicts = int(plan["conflict"].sum())
    print(f"Rated set: {len(rated)} posts, {len(key)} items; {int((plan['slot'] == 2).sum())} posts double-rated; "
          f"{conflicts} assignments where a rater had annotated the post (roster too small to avoid).\n"
          f"Wrote the key to {paths.items_key} (raters must not open it) and sheets to {paths.work / 'rating'}.\n"
          "Next: rate with 02-llm/codebook/rating-rubric.md (rating sheet first, then the best-fit sheet), "
          "then run `collect-ratings`.")


# ---------------------------------------------------------------- F. collect

def collect_rating_sheets() -> None:
    """Validate every rater's sheets and write ratings.csv and bestfit.csv."""
    paths, _, schema, _ = _context()
    plan = pd.read_csv(paths.rating_assignment, dtype=str)
    plan["slot"] = plan["slot"].astype(int)
    key = pd.read_csv(paths.items_key, dtype=str)
    bf_key = pd.read_csv(paths.bestfit_key, dtype=str)
    raters = sorted(plan["rater"].unique())
    sheets = {r: read_sheet(paths.sheet("rating", r)) for r in raters}
    bests = {r: read_sheet(paths.sheet("rating", f"{r}_bestfit")) for r in raters}
    ratings, best = collect_ratings(sheets, bests, schema, plan.rename(columns={"rater": "person"}), key, bf_key)
    _write_csv(ratings, paths.ratings)
    _write_csv(best, paths.bestfit)
    print(f"Wrote {paths.ratings} ({len(ratings)} ratings) and {paths.bestfit} ({len(best)} choices).\n"
          "Next: run the notebooks 01-07 in 03-evaluation (or `workflow.py analyze`).")


# ---------------------------------------------------------------- status

def status() -> None:
    """Print which artifacts exist and the next step."""
    paths = get_paths()
    cfg = load_config(paths)
    manual = _prescreen_method(cfg) == "manual"
    steps: list[tuple[str, Callable[[], bool], str]] = [
        *([("pre-screen sheet", lambda: any((paths.work / "prescreen").glob("*.xlsx")), "prescreen-sheet")]
          if manual else []),
        ("sample + annotation sheets", lambda: paths.sample.exists(),
         "pre-screen, then sample" if manual else "sample (keyword pre-screen)"),
        ("rated-set candidates", lambda: paths.rating_candidates.exists(), "rating-candidates"),
        ("GPU: recognition + P0 + P-Oracle grid",
         lambda: len(list(paths.advice.glob("replies_*_p0.jsonl"))) >= 2
         and len(list(paths.er.glob("predictions_*.jsonl"))) >= 2 and len(list(paths.oracle_grid.glob("*.jsonl"))) >= 2,
         "pipeline.py on the GPU PC, then copy 02-llm/gold/ back here"),
        ("collected annotations", lambda: paths.annotations_raw.exists(), "annotate blind, then adjudication-sheet"),
        ("gold labels + P-Oracle selected", lambda: paths.adjudicated.exists()
         and len(list(paths.advice.glob("replies_*_oracle.jsonl"))) >= 2, "adjudicate, then finalize-labels"),
        ("rating sheets", lambda: paths.items_key.exists(), "rating-sheets"),
        ("ratings", lambda: paths.ratings.exists(), "rate, then collect-ratings"),
    ]
    next_step = None
    for name, done, how in steps:
        ok = bool(done())
        print(f"[{'x' if ok else ' '}] {name}")
        if not ok and next_step is None:
            next_step = how
    print(f"\nNext: {next_step}" if next_step else "\nAll data steps done: run the analysis notebooks.")
