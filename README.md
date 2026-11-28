# payo-bench
Payo Benchmark: Gemma vs. SEA-LION on Filipino emotion recognition and empathic advice generation, rated blind by the authors.

**What to do next:** run `uv run python pipeline.py` (below), then follow [`RUNBOOK.md`](RUNBOOK.md). Every remaining step is either filling in a sheet or running one command.

## Usage

### One command

```bash
uv sync                        # once (on the GPU PC: uv sync --extra full, and put HF_TOKEN in .env)
uv run python pipeline.py
```

`pipeline.py` runs every automatic step that can run on this machine, skips what is already done, and stops at the next human step with instructions. Rerun it any time; it resumes.

| # | Stage | What it runs | Where |
| --- | --- | --- | --- |
| 1 | Silver | `01-preprocessing/01-preprocessing.ipynb` (bronze → `silver/advice_pairs.parquet`), only if silver is missing; then checks it against `silver/MANIFEST.json` | Main machine (needs the bronze dump and the private `pii/name_review.csv`). Other machines copy the parquet file instead |
| 2 | Sample | `03-evaluation/workflow.py sample` (keyword pre-screen, 236 stratified posts, one annotation sheet per person) and `rating-candidates` (71 rated-set candidates) | Any machine; done once, then committed |
| 3 | All GPU work | `02-llm/scripts/run.py --profile full`: GPU check, a 3-post trial, emotion recognition + P0 advice (both models) with variance runs, and the **P-Oracle grid** (each candidate × every emotion × concept; `finalize-labels` later picks the reply matching the gold label) | GPU PC: Linux or WSL2, NVIDIA RTX 30-series or newer, 8 GB+ (FP8 is picked automatically under 11 GB). Free alternative: Kaggle 2× T4 with `--precision fp32 --tp 2` (`02-llm/scripts/kaggle_full_run.ipynb`) |

On a small or busy GPU, pass vLLM options through: `uv run python pipeline.py --gpu-mem 0.8`. Use the same flags for every run.

### Typical flow

All GPU work happens first, in one session. People are only needed after it.

```bash
# Main machine (Mac)
uv run python pipeline.py                       # silver ok, sample + candidates ok, "no GPU here"
# GPU PC: copy 01-preprocessing/silver/advice_pairs.parquet over first (it is gitignored)
git pull && uv sync --extra full && cp .env.example .env
uv run python pipeline.py                       # all GPU work; copy 02-llm/gold/ back, unopened
# People: annotate blind; then on the main machine:
uv run python 03-evaluation/workflow.py adjudication-sheet   # Ethan fills it
uv run python 03-evaluation/workflow.py finalize-labels      # labels frozen + P-Oracle selected
uv run python 03-evaluation/workflow.py rating-sheets        # rate, then collect-ratings
uv run python 03-evaluation/workflow.py analyze              # every statistical test
```

`uv run python 03-evaluation/workflow.py status` shows where the study is. [`RUNBOOK.md`](RUNBOOK.md) has every step, including the human ones; each stage's README has the details ([`01-preprocessing`](01-preprocessing/README.md), [`02-llm/scripts`](02-llm/scripts/README.md), [`03-evaluation`](03-evaluation/README.md)).

## Repository layout

The repo reads top to bottom in pipeline order. Each medallion layer sits inside the stage that produces it.

```
00-data/bronze/            Raw r/adviceph dump (immutable)
01-preprocessing/silver/   Template posts paired with their top-voted reply
02-llm/gold/               Frozen benchmark artifacts + Gemma / SEA-LION outputs
02-llm/work/               Sheets for annotators and raters (gitignored; contain post text)
03-evaluation/             Workflow commands, statistics, notebooks
```

| Layer  | Location                   | Rule                                        |
| ------ | -------------------------- | ------------------------------------------- |
| Bronze | `00-data/bronze/`          | Never edited                                |
| Silver | `01-preprocessing/silver/` | Regenerable from bronze by code             |
| Gold   | `02-llm/gold/`             | Versioned and frozen; what the paper reports on |

Data files are not in git. Each layer has a committed `MANIFEST.json` (sha256 + row counts) for checking your copy. See [`00-data/README.md`](00-data/README.md) for the full data contract.
