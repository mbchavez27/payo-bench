"""
Draw the 30 pilot posts from the silver table.

Writes:
    02-llm/pilot/pilot_posts.csv         post ids only (committed)
    02-llm/pilot/pilot_annotations.csv   annotation sheet with post ids prefilled (gitignored);
                                         label it with 02-llm/codebook/codebook.md so the pilot
                                         P-Oracle run has gold labels

Pilot posts are not part of the final ~300 (codebook §8); the sampling notebook excludes these ids.

Usage:
    uv run python 02-llm/scripts/make_pilot_sample.py
"""

import pandas as pd

from payo_llm import ROOT
from payo_llm.data import SILVER_PATH

SEED = 42
N_POSTS = 30
PILOT_DIR = ROOT / "02-llm" / "pilot"
TEMPLATE = ROOT / "02-llm" / "codebook" / "annotation-template.csv"


def main() -> None:
    ids_path = PILOT_DIR / "pilot_posts.csv"
    sheet_path = PILOT_DIR / "pilot_annotations.csv"
    if ids_path.exists() or sheet_path.exists():
        raise SystemExit(f"Pilot files already exist in {PILOT_DIR}; delete them first to redraw.")

    # Simple random sample: there is no pre-screen yet, and the pilot's job is to check retention and label balance
    silver = pd.read_parquet(SILVER_PATH, columns=["post_id"])
    sample = silver.sample(N_POSTS, random_state=SEED).reset_index(drop=True)

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(ids_path, index=False)
    columns = pd.read_csv(TEMPLATE, nrows=0).columns
    sheet = pd.DataFrame("", index=range(N_POSTS), columns=columns)
    sheet["post_id"] = sample["post_id"]
    sheet.to_csv(sheet_path, index=False)
    print(f"Wrote {N_POSTS} pilot posts to {ids_path.relative_to(ROOT)} and {sheet_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
