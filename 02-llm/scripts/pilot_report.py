"""
Pilot health report: Task A validity, label collapse, reply length, timing, retention and label balance.

Shows no accuracy against gold labels: the pilot is not a result, and model-vs-gold numbers come only
from the full run (03-evaluation/analysis-plan.md). Works after any subset of tasks has run.

Writes:
    <out-dir>/pilot_report.md   (02-llm/pilot/outputs/ is gitignored)

Usage:
    uv run python 02-llm/scripts/pilot_report.py
    uv run python 02-llm/scripts/pilot_report.py --out-dir <folder> --annotations <csv>
"""

import argparse
import math
import tomllib
from pathlib import Path

import pandas as pd

from payo_llm import ROOT
from payo_llm.data import SILVER_PATH, read_post_ids
from payo_llm.pilot import (annotation_summary, advice_health, completeness, flags, load_outputs,
                            recognition_health, render_table, timing)
from payo_llm.schema import load_schema

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
N_TARGET = 300
MIN_LABEL = 15


def parse_args(profile: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / profile["out_dir"], help="run output folder")
    parser.add_argument("--annotations", type=Path, default=ROOT / profile["gold"], help="pilot annotation sheet")
    parser.add_argument("--posts", type=Path, default=ROOT / profile["posts"], help="pilot post ids")
    return parser.parse_args()


def oracle_ids(sheet: pd.DataFrame) -> list[str]:
    """Posts that get a P-Oracle prompt: labeled and not excluded (same rule as data.load_gold)."""
    sheet = sheet.fillna("")
    keep = (sheet["exclude_flag"].str.strip().str.lower() != "yes") & (sheet["primary_emotion"].str.strip() != "")
    return sorted(set(sheet.loc[keep, "post_id"].str.strip()))


def main() -> None:
    cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    profile = cfg["profiles"]["pilot"]
    args = parse_args(profile)
    schema = load_schema()

    post_ids = read_post_ids(args.posts)
    sheet = pd.read_csv(args.annotations, dtype=str) if args.annotations.exists() else pd.DataFrame(
        columns=["post_id", "annotator", "exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept",
                 "expression", "domain", "need_type"])
    silver = pd.read_parquet(SILVER_PATH, columns=["post_id", "reply_body"])
    community_words = silver.set_index("post_id")["reply_body"].fillna("").str.split().str.len()

    outputs = load_outputs(args.out_dir)
    expected = {"recognition": post_ids, "advice-p0": post_ids, "advice-oracle": oracle_ids(sheet)}
    complete = completeness(outputs, expected)
    rec_summary, rec_labels, same_label = recognition_health(outputs)
    adv_summary, adv_diff = advice_health(outputs, community_words)
    times = timing(outputs, profile["num_ctx"])
    ann = annotation_summary(sheet, schema, n_silver=len(silver), n_target=N_TARGET, min_label=MIN_LABEL)
    warnings = flags(complete, rec_summary, adv_summary, times, ann, n_silver=len(silver), min_label=MIN_LABEL)

    lines = [
        "# Pilot report",
        "",
        f"Posts: {len(post_ids)} · outputs: `{args.out_dir}` · annotations: `{args.annotations}`",
        "",
        "The pilot is **not a result**: different backend and quantization from the full run, and no "
        "accuracy is computed. With 30 posts every share is rough (about ±18 points).",
        "",
        "## Flags",
        "",
        *([f"- {w}" for w in warnings] or ["- none"]),
        "",
        "## Completeness",
        "",
        render_table(complete),
        "",
        "## Task A: output health (no gold)",
        "",
    ]
    if rec_summary.empty:
        lines += ["_Task A not run yet._", ""]
    else:
        lines += [render_table(rec_summary), "", "Predicted emotion counts:", "", render_table(rec_labels), ""]
        if same_label is not None:
            lines += [f"Gemma and SEA-LION give the same emotion on {same_label:.0%} of posts where both are valid "
                      "(descriptive; not agreement with gold).", ""]
    lines += ["## Task B: reply length", ""]
    if adv_summary.empty:
        lines += ["_Task B not run yet._", ""]
    else:
        lines += [render_table(adv_summary), ""]
        if not adv_diff.empty:
            lines += ["P-Oracle minus P0 words, same post (prompt-length confound):", "", render_table(adv_diff), ""]
    lines += ["## Timing and prompt length", "", render_table(times) if not times.empty else "_nothing run yet_", ""]

    lines += ["## Annotations: retention and label balance", ""]
    if not ann.n_labeled:
        lines += [f"_Not labeled yet ({ann.n_rows} rows in the sheet)._", ""]
    else:
        low, high = ann.retention_ci
        needed = f"{ann.posts_needed}" if ann.posts_needed else "n/a"
        lines += [
            f"Labeled rows: {ann.n_labeled} of {ann.n_rows} · excluded: {ann.n_excluded} · "
            f"retention: {ann.retention:.0%} (95% CI {low:.0%}–{high:.0%}) · "
            f"silver posts needed for {N_TARGET}: {needed} of {len(silver)}",
            "",
            render_table(ann.exclude_reasons),
            "",
            "Emotion balance (kept rows). `expected_in_300_random` assumes a simple random sample; the "
            f"stratified sample can oversample rare labels up to `pool_estimate`. `below_min`: pool under {MIN_LABEL}.",
            "",
            render_table(ann.emotions),
            "",
            render_table(ann.emotion_types),
            "",
            f"Cultural concept present: {ann.concept_present:.0%}" if not math.isnan(ann.concept_present) else "",
            "",
        ]
        for table in ann.categories.values():
            lines += [render_table(table), ""]
        if not ann.agreement.empty:
            lines += ["### Annotator agreement (posts with more than one annotator)", "",
                      render_table(ann.agreement), "", render_table(ann.disagreements), ""]

    report = "\n".join(lines)
    print(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "pilot_report.md"
    target.write_text(report, encoding="utf-8")
    print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
