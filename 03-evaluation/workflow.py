"""
The human-in-the-loop workflow, one command per gate. See RUNBOOK.md for the full order.

    uv run python 03-evaluation/workflow.py status
    uv run python 03-evaluation/workflow.py pilot-sheets        # every annotator labels the pilot posts
    uv run python 03-evaluation/workflow.py pilot-merge         # -> pilot_report.py, adjudication sheet
    uv run python 03-evaluation/workflow.py pilot-adjudicate    # -> pilot_annotations.csv (pilot P-Oracle)
    uv run python 03-evaluation/workflow.py prescreen-sheet     # A. pre-screen pool
    uv run python 03-evaluation/workflow.py sample              # B. stratified sample + annotation sheets
    uv run python 03-evaluation/workflow.py adjudication-sheet  # C. collect annotations, disagreements
    uv run python 03-evaluation/workflow.py finalize-labels     # D. gold labels, merges, κ gate
    uv run python 03-evaluation/workflow.py rating-sheets       # E. after the full run: blinded rating sheets
    uv run python 03-evaluation/workflow.py collect-ratings     # F. ratings.csv, bestfit.csv
    uv run python 03-evaluation/workflow.py analyze             # run notebooks 01-07 headless
    uv run python 03-evaluation/workflow.py smoke               # whole chain on synthetic data

Sheets go to 02-llm/work/ (gitignored: they contain post text). Ids and labels go to 02-llm/gold/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from payo_eval import steps

EVAL_DIR = Path(__file__).resolve().parent
NOTEBOOKS = ["01-label-finalization", "02-emotion-recognition", "03-advice-quality", "04-strategy-drift",
             "05-linking", "06-reliability", "07-multiple-comparisons"]


def run_notebooks(out_dir: Path | None = None) -> None:
    """Execute every analysis notebook top to bottom (in place, or into out_dir)."""
    for name in NOTEBOOKS:
        path = EVAL_DIR / f"{name}.ipynb"
        argv = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(path),
                "--ExecutePreprocessor.timeout=1800"]
        argv += ["--output-dir", str(out_dir)] if out_dir else ["--inplace"]
        print(f"Running {name} ...", flush=True)
        subprocess.run(argv, check=True, cwd=EVAL_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "pilot-sheets", "pilot-merge", "pilot-adjudicate",
                                            "prescreen-sheet", "sample", "adjudication-sheet", "finalize-labels",
                                            "rating-sheets", "collect-ratings", "analyze", "smoke"])
    parser.add_argument("--force", action="store_true", help="overwrite existing sheets/outputs of this step")
    parser.add_argument("--accept-kappa", action="store_true", help="finalize-labels: continue past the κ gate")
    parser.add_argument("--dir", type=Path, help="smoke: folder for the synthetic run (default: a temp folder)")
    args = parser.parse_args()

    match args.command:
        case "status":
            steps.status()
        case "pilot-sheets":
            steps.pilot_sheets(args.force)
        case "pilot-merge":
            steps.pilot_merge(args.force)
        case "pilot-adjudicate":
            steps.pilot_adjudicate()
        case "prescreen-sheet":
            steps.prescreen_sheet(args.force)
        case "sample":
            steps.sample(args.force)
        case "adjudication-sheet":
            steps.adjudication_sheet(args.force)
        case "finalize-labels":
            steps.finalize_labels(args.accept_kappa)
        case "rating-sheets":
            steps.rating_sheets(args.force)
        case "collect-ratings":
            steps.collect_rating_sheets()
        case "analyze":
            run_notebooks()
        case "smoke":
            from payo_eval.synthetic import smoke  # noqa: PLC0415

            smoke(args.dir, lambda out: run_notebooks(out))


if __name__ == "__main__":
    main()
