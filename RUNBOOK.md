# Runbook: from here to the paper

Every remaining step is either **a person filling in a sheet** or **one command**. Run the steps in order; each command checks its inputs, refuses to overwrite started work, and prints the next step. `uv run python 03-evaluation/workflow.py status` shows where you are.

Sheets are Excel files in `02-llm/work/` (gitignored, because they contain post text). Open them in Excel, or upload them to Google Sheets and download them back as `.xlsx` with the **same file name**. Fill only the pale-yellow columns, and use the dropdowns. Ids and labels go to `02-llm/gold/` (committed).

Shorthand: `wf` = `uv run python 03-evaluation/workflow.py`.

## 0. Once

| Who | Do | Check |
| --- | --- | --- |
| Team | Resolve every `TO-VERIFY` in `02-llm/codebook/codebook.md`, then set the codebook, `label-schema.json`, `rating-rubric.md` and `advice-strategy.md` to v1 | No `TO-VERIFY` left |
| Team | Check `[roster]` in `03-evaluation/config.toml`. Set now: all five annotate and rate (`max`, `airon`, `jayhan`, `jerome`, `ethan`), `ethan` adjudicates, `max` pre-screens. Change it before `wf sample` if needed | `wf status` runs |
| Anyone | `uv sync` | `uv run pytest` passes |

## 1. Pilot labels and pilot P-Oracle

| # | Who | Do |
| --- | --- | --- |
| 1.1 | Command | `wf pilot-sheets` → one sheet per annotator in `02-llm/work/pilot/` (the same 30 posts for everyone) |
| 1.2 | Each annotator | Label your sheet alone, with the codebook |
| 1.3 | Command | `wf pilot-merge` → `02-llm/pilot/pilot_annotations_raw.csv` + `work/pilot/adjudication.xlsx` |
| 1.4 | Command | `uv run python 02-llm/scripts/pilot_report.py --annotations 02-llm/pilot/pilot_annotations_raw.csv`: retention, label balance, disagreements |
| 1.5 | Team | Discuss every disagreement; revise the codebook (log it in its §10). The adjudicator fills the blank `final_*` cells in the adjudication sheet |
| 1.6 | Command | `wf pilot-adjudicate` → `02-llm/pilot/pilot_annotations.csv` (one row per post) |
| 1.7 | Command | `ollama serve` (other terminal), then `uv run python 02-llm/scripts/run.py --profile pilot --task advice-oracle --model both`, then `pilot_report.py` again |

Pilot labels are never part of the study set.

## 2. Pre-screen and sample

| # | Who | Do |
| --- | --- | --- |
| 2.1 | Command | `wf prescreen-sheet` → `02-llm/work/prescreen/<screener>.xlsx` (900 random posts, pilot excluded) |
| 2.2 | Screener | About 30 s per post: `exclude_flag` (+ reason), `concept_likely`, `domain_guess` (codebook §0) |
| 2.3 | Command | `wf sample` → `02-llm/gold/sample_posts.csv` (~334 posts, stratified), the assignment (25% hidden overlap), and one annotation sheet per annotator in `work/annotation/` |

## 3. Annotation and gold labels

| # | Who | Do |
| --- | --- | --- |
| 3.1 | Each annotator | Label your sheet alone (codebook §1-§7). Don't read replies or open the thread |
| 3.2 | Command | `wf adjudication-sheet` → `annotations_raw.csv` + `work/annotation/adjudication.xlsx` (only the double-annotated disagreements; agreed fields prefilled) |
| 3.3 | Adjudicator | Fill the blank `final_*` cells |
| 3.4 | Command | `wf finalize-labels` → `annotations_adjudicated.csv`, `merges.json`, `03-evaluation/results/label_finalization.md` |

If 3.4 stops:
- **"no merge target"**: a label has fewer than 15 posts. Set its target under `[merges]` in `config.toml` (e.g. `pakikisama_strain = "hiya"`), log it in codebook §10, and rerun.
- **"κ gate failed"**: revise the label's definition and re-adjudicate it, add a merge, or rerun with `--accept-kappa` and log the reason in `analysis-plan.md` §11.

Labels are now frozen. Nobody looks at model output before this point.

## 4. Full run (Linux + NVIDIA GPU)

Follow "Full run" in `02-llm/scripts/README.md`. In short:

```bash
uv sync --extra full && cp .env.example .env     # HF_TOKEN; accept the Gemma license first
uv run python 02-llm/scripts/run.py --profile full --check --model both
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle --model both
uv run python 02-llm/scripts/run.py --profile full --task all --model both --variance
```

Copy `02-llm/gold/` (JSONL + `MANIFEST.json` + `run_record.json`) back to the main machine, and fill the run-record table in `02-llm/README.md`.

## 5. Rating

| # | Who | Do |
| --- | --- | --- |
| 5.1 | Command | `wf rating-sheets` → 80 rated posts × 5 blinded replies; per rater `work/rating/<rater>.xlsx` and `<rater>_bestfit.xlsx`. The key (`02-llm/gold/author-ratings/items_key.csv`) must stay closed until 5.3 |
| 5.2 | Each rater | Rating sheet first (rubric anchors + strategy code for every reply), then the best-fit sheet (A/B/C per post) |
| 5.3 | Command | `wf collect-ratings` → `ratings.csv`, `bestfit.csv`. It lists every invalid or empty cell; fix them and rerun |

## 6. Analysis

```bash
wf analyze        # runs notebooks 01-07 in 03-evaluation/ in place
```

Tables go to `03-evaluation/results/` and figures to `03-evaluation/figures/`. `07-multiple-comparisons` holds the Holm-corrected table of every confirmatory test. Write the findings cell at the end of each notebook.

## Safety nets

- `wf smoke` runs this whole runbook on synthetic data in a temp folder (sheets filled by code). Run it after any code change.
- `--force` on a step regenerates its outputs. Use it only if nobody has started the next human step: regenerated sheets have new posts or item ids.
- Sheets are validated on the way back in. A typo stops the step with the row and column, and nothing half-valid is written.
