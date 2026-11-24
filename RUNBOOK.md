# Runbook: from here to the paper

Every remaining step is either **a person filling in a sheet** or **one command**. Run the steps in order; each command checks its inputs, refuses to overwrite started work, and prints the next step. `uv run python 03-evaluation/workflow.py status` shows where you are.

The order follows analysis plan v1.3:
1. The models run first, on the sampled posts.
2. The team annotates blind.
3. The one task that needs your labels (P-Oracle) runs last.

This is the same pattern as an LLM-coded study validated by blind hand coding. Here the hand labels are the gold standard.

Sheets are Excel files in `02-llm/work/`, gitignored because they contain post text. Open them in Excel, or upload them to Google Sheets and download them back as `.xlsx` with the **same file name**. Fill only the pale-yellow columns and use the dropdowns. Ids and labels go to `02-llm/gold/` (committed).

Shorthand: `wf` = `uv run python 03-evaluation/workflow.py`.

## Blindness rule

Model outputs (`02-llm/gold/**/*.jsonl`) are gitignored. Whoever runs the GPU does not open, print or share them, and does not commit them, until step 3.3 (`finalize-labels`) has run. `run.py` prints only counts and timing. Annotators label from the post alone.

## 0. Once

| Who | Do | Check |
| --- | --- | --- |
| Done | Codebook, `label-schema.json`, `rating-rubric.md`, `advice-strategy.md` frozen at v1 (sources verified) | No `TO-VERIFY` left |
| Done | `[roster]` in `03-evaluation/config.toml`: `max`, `airon`, `jayhan`, `jerome`, `ethan` annotate and rate; `ethan` adjudicates | `wf status` runs |
| Anyone | `uv sync` | `uv run pytest` passes |

## 1. Sample (one command, no human step)

| # | Who | Do |
| --- | --- | --- |
| 1.1 | Command | `wf sample` → the keyword pre-screen of every silver post (`02-llm/gold/emotion-recognition/prescreen.csv`, matched words in `notes`), `02-llm/gold/sample_posts.csv` (236 posts, stratified by culture-likely × domain), the assignment (25% hidden overlap), and one annotation sheet per annotator in `work/annotation/` |
| 1.2 | Team | Commit and push `sample_posts.csv`, `prescreen.csv` and `annotation_assignment.csv` (ids only), so the GPU machine has the same sample |

## 2. GPU run 1 and annotation (in parallel)

**2a. GPU run 1 (Linux + NVIDIA GPU)**: recognition + P0 advice. These don't need labels. Follow "Full run" in `02-llm/scripts/README.md`:

```bash
uv sync --extra full && cp .env.example .env     # HF_TOKEN; accept the Gemma license first
uv run python 02-llm/scripts/run.py --profile full --check --model both
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both --variance
```

Keep the outputs on that machine, or in a private folder, unopened (blindness rule).

**2b. Annotation (all 5)**

| # | Who | Do |
| --- | --- | --- |
| 2.1 | Each annotator | Label your sheet alone (codebook §1-§7), about 56 posts each. Don't read the replies, open the thread, or look at any model output |

## 3. Gold labels

| # | Who | Do |
| --- | --- | --- |
| 3.1 | Command | `wf adjudication-sheet` → `annotations_raw.csv` + `work/annotation/adjudication.xlsx` (only the double-annotated disagreements; agreed fields prefilled) |
| 3.2 | Ethan | Fill the blank `final_*` cells |
| 3.3 | Command | `wf finalize-labels` → `annotations_adjudicated.csv`, `merges.json`, `03-evaluation/results/label_finalization.md` |

If 3.3 stops:
- **"no merge target"**: a label has fewer than 15 posts. Set its target under `[merges]` in `config.toml` (e.g. `pakikisama_strain = "hiya"`), log it in codebook §10, and rerun.
- **"κ gate failed"**: revise the label's definition and re-adjudicate it, add a merge, or rerun with `--accept-kappa` and log the reason in `analysis-plan.md` §11.

**Labels are now frozen.** From here on, the model outputs may be opened.

## 4. GPU run 2: P-Oracle

```bash
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle --model both
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle --model both --variance
```

Copy `02-llm/gold/` (JSONL + `MANIFEST.json` + `run_record.json`) back to the main machine, and fill in the run-record table in `02-llm/README.md`.

## 5. Rating

| # | Who | Do |
| --- | --- | --- |
| 5.1 | Command | `wf rating-sheets` → 60 rated posts × 5 blinded replies; per rater `work/rating/<rater>.xlsx` and `<rater>_bestfit.xlsx`. The key (`02-llm/gold/author-ratings/items_key.csv`) stays closed until 5.3 |
| 5.2 | Each rater | Rating sheet first (rubric anchors + strategy code for every reply), then the best-fit sheet (A/B/C per post). About 75 replies each |
| 5.3 | Command | `wf collect-ratings` → `ratings.csv`, `bestfit.csv`. It lists every invalid or empty cell; fix them and rerun |

## 6. Analysis

```bash
wf analyze        # runs notebooks 01-07 in 03-evaluation/ in place
```

Tables go to `03-evaluation/results/` and figures to `03-evaluation/figures/`. `07-multiple-comparisons` holds the Holm-corrected table of every confirmatory test. Write the findings cell at the end of each notebook.

## Optional (not used in v1.3)

- **Shared pilot:** `wf pilot-sheets` → label → `wf pilot-merge` → `pilot_report.py --annotations …` → `wf pilot-adjudicate`. Pilot posts are always excluded from the sample.
- **Manual pre-screen:** set `[prescreen] method = "manual"`, run `wf prescreen-sheet`, have the screener fill it, then run `wf sample`.

## Safety nets

- `wf smoke` runs this whole runbook on synthetic data in a temp folder, with sheets filled by code. Run it after any code change.
- `--force` on a step regenerates its outputs. Use it only if nobody has started the next human step: regenerated sheets have new posts or item ids.
- Sheets are validated on the way back in. A typo stops the step with the row and column, and nothing half-valid is written.
