"""
One command from preprocessing to the model runs: does every automatic step that can run now, then stops
and says what a person has to do next.

    uv run python pipeline.py                 # Mac or GPU PC: run what's possible, print the next step
    uv run python pipeline.py --gpu-mem 0.8   # GPU PC: pass vLLM options through to every model run
    uv run python pipeline.py --no-trial      # skip the 3-post trial before the first model run

Stages, in order (each one is skipped when its output already exists):
  1. silver      01-preprocessing notebook: bronze -> silver/advice_pairs.parquet, checked against MANIFEST.json
  2. sample      03-evaluation/workflow.py sample: keyword pre-screen, 236 posts, annotation sheets
     candidates  03-evaluation/workflow.py rating-candidates: 71 rated-set candidates
  3. GPU         02-llm/scripts/run.py, both models (needs an NVIDIA GPU), all before any human step:
                 recognition + P0 advice, their variance runs, and the P-Oracle grid (every emotion x concept
                 for each candidate; `finalize-labels` later picks the reply matching the gold label)
After the GPU stage, people annotate and rate (RUNBOOK.md); no more GPU work is needed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SILVER = ROOT / "01-preprocessing" / "silver" / "advice_pairs.parquet"
SILVER_MANIFEST = ROOT / "01-preprocessing" / "silver" / "MANIFEST.json"
BRONZE = ROOT / "00-data" / "bronze"
NAME_REVIEW = ROOT / "01-preprocessing" / "pii" / "name_review.csv"
NOTEBOOK = ROOT / "01-preprocessing" / "01-preprocessing.ipynb"
GOLD = ROOT / "02-llm" / "gold"
SAMPLE = GOLD / "sample_posts.csv"
CANDIDATES = GOLD / "author-ratings" / "rating_candidates.csv"
TRIAL_DIR = Path("/tmp/payo-trial")


def say(text: str) -> None:
    print(f"\n== {text}", flush=True)


def run(argv: list[str]) -> None:
    """Run one existing command; stop the pipeline if it fails."""
    print("$ " + " ".join(argv), flush=True)
    if subprocess.run(argv, cwd=ROOT, check=False).returncode != 0:
        sys.exit(f"\nStopped: `{' '.join(argv)}` failed. Fix the message above and rerun pipeline.py (it resumes).")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_silver() -> None:
    """Build silver from bronze if it is missing, and check it matches the committed manifest."""
    expected = json.loads(SILVER_MANIFEST.read_text())["files"]["advice_pairs.parquet"]["sha256"]
    if not SILVER.exists():
        if not (any(BRONZE.glob("*/posts.jsonl")) and NAME_REVIEW.exists()):
            sys.exit("Stopped: no silver data on this machine. Copy 01-preprocessing/silver/advice_pairs.parquet from "
                     "the main machine (it is gitignored), then rerun. Rebuilding needs the bronze dump and the "
                     "private pii/name_review.csv, and would not match the committed sample without them.")
        say("1. silver: running the preprocessing notebook (10-15 min the first time)")
        run(["uv", "run", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=-1", str(NOTEBOOK)])
    if sha256(SILVER) != expected:
        sys.exit("Stopped: silver/advice_pairs.parquet doesn't match silver/MANIFEST.json. The sample and every "
                 "later file were drawn from the committed version: copy that file from the main machine.")
    say("1. silver: ok (matches MANIFEST.json)")


def stage_sample() -> None:
    if SAMPLE.exists():
        say(f"2. sample: ok ({SAMPLE.relative_to(ROOT)} exists)")
        return
    say("2. sample: drawing the study sample and the annotation sheets")
    run(["uv", "run", "python", "03-evaluation/workflow.py", "sample"])
    print("\nCommit and push 02-llm/gold/sample_posts.csv and 02-llm/gold/emotion-recognition/{prescreen,"
          "annotation_assignment}.csv, then send each person their sheet from 02-llm/work/annotation/.")


def stage_candidates() -> None:
    if CANDIDATES.exists():
        say(f"2. candidates: ok ({CANDIDATES.relative_to(ROOT)} exists)")
        return
    say("2. candidates: drawing the rated-set candidates (the P-Oracle grid runs on these)")
    run(["uv", "run", "python", "03-evaluation/workflow.py", "rating-candidates"])
    print(f"\nCommit and push {CANDIDATES.relative_to(ROOT)} so the GPU PC uses the same candidates.")


def has_cuda() -> bool:
    probe = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run(["uv", "run", "python", "-c", probe], cwd=ROOT, capture_output=True).returncode == 0


def outputs(pattern: str) -> int:
    return len(list(GOLD.glob(pattern)))


def stage_models(extra: list[str], trial: bool) -> None:
    """All GPU work, in one session: check, 3-post trial, recognition + P0, variance, P-Oracle grid. All resume."""
    base = ["uv", "run", "python", "02-llm/scripts/run.py", "--profile", "full", "--model", "both", *extra]
    if not has_cuda():
        say("3. GPU: no NVIDIA GPU here")
        print("Run this same command on the GPU PC (Linux or WSL2, RTX 30-series or newer, 8 GB+):\n"
              "  git pull && uv sync --extra full && cp .env.example .env   # HF_TOKEN in .env\n"
              "  uv run python pipeline.py\n"
              "Copy 01-preprocessing/silver/advice_pairs.parquet to it first (gitignored).")
        return
    say("3. GPU: GPU and model access check")
    run([*base, "--check"])
    run1 = outputs("emotion-recognition/predictions_*.jsonl") < 2 or outputs("advice-generation/replies_*_p0.jsonl") < 2
    if run1 and trial and not TRIAL_DIR.exists():
        say("3. trial: 3 posts into /tmp/payo-trial (throwaway), to confirm both models load and fit")
        run([*base, "--task", "recognition,advice-p0", "--limit", "3", "--out-dir", str(TRIAL_DIR)])
    say("3. recognition + P0 advice (resumes if interrupted). Don't open the outputs until finalize-labels")
    run([*base, "--task", "recognition,advice-p0"])
    run([*base, "--task", "recognition,advice-p0", "--variance"])
    say("3. P-Oracle grid: every emotion x concept for each candidate (the longest step; resumes)")
    run([*base, "--task", "advice-oracle-grid"])
    print("\nGPU work complete. Copy 02-llm/gold/ (the JSONL files, MANIFEST.json, run_record.json) to the main "
          "machine without opening them. People can now annotate (RUNBOOK §3); `finalize-labels` selects the "
          "P-Oracle replies, and no more GPU work is needed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu-mem", help="passed to run.py (use the same value for every run)")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp8", "bnb4"], help="passed to run.py")
    parser.add_argument("--no-trial", action="store_true", help="skip the 3-post trial run")
    args = parser.parse_args()
    extra = [*(["--gpu-mem", args.gpu_mem] if args.gpu_mem else []),
             *(["--precision", args.precision] if args.precision else [])]
    stage_silver()
    stage_sample()
    stage_candidates()
    stage_models(extra, trial=not args.no_trial)
    print("\nWhere the whole study is: uv run python 03-evaluation/workflow.py status")


if __name__ == "__main__":
    main()
