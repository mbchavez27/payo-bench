# Runbook: from here to the paper

Every remaining step is either **a person filling in a sheet** or **one command**. Run the steps in order; each command checks its inputs, refuses to overwrite started work, and prints the next step. `uv run python 03-evaluation/workflow.py status` shows where you are.

The order follows analysis plan v1.4:
1. **All GPU work runs first**, in one session. This includes P-Oracle, pre-generated for every possible label.
2. **Then people**: blind annotation, then rating.

This is the same pattern as an LLM-coded study validated by blind hand coding. Here the hand labels are the gold standard.

Sheets are Excel files in `02-llm/work/`, gitignored because they contain post text. Open them in Excel, or upload them to Google Sheets and download them back as `.xlsx` with the **same file name**. Fill only the pale-yellow columns and use the dropdowns. Ids and labels go to `02-llm/gold/` (committed).

Shorthand: `wf` = `uv run python 03-evaluation/workflow.py`.

**Shortcut:** `uv run python pipeline.py` runs every automatic step from preprocessing to the end of the GPU work (§1, §2) that can run on the current machine, and stops at the next human step.

## Blindness rule

Model outputs (`02-llm/gold/**/*.jsonl`) are gitignored. Whoever runs the GPU does not open, print or share them, and does not commit them, until step 4.3 (`finalize-labels`) has run. `run.py` prints only counts and timing. Annotators label from the post alone.

## 0. Once

| Who | Do | Check |
| --- | --- | --- |
| Done | Codebook, `label-schema.json`, `rating-rubric.md`, `advice-strategy.md` frozen at v1 (sources verified) | No `TO-VERIFY` left |
| Done | `[roster]` in `03-evaluation/config.toml`: `max`, `airon`, `jayhan`, `jerome`, `ethan` annotate and rate; `ethan` adjudicates | `wf status` runs |
| Anyone | `uv sync` | `uv run pytest` passes |

## 1. Sample (commands, no human step)

| # | Who | Do |
| --- | --- | --- |
| 1.1 | Command | `wf sample` → the keyword pre-screen of every silver post (`02-llm/gold/emotion-recognition/prescreen.csv`, matched words in `notes`), `02-llm/gold/sample_posts.csv` (236 posts, stratified by culture-likely × domain), the assignment (25% hidden overlap), and one annotation sheet per annotator in `work/annotation/` ✅ |
| 1.2 | Command | `wf rating-candidates` → `02-llm/gold/author-ratings/rating_candidates.csv` (71 posts; the rated set is the first 60 kept after annotation) ✅ |
| 1.3 | Team | Commit and push those CSVs (ids only), so the GPU machine uses the same posts ✅ |

## 2. All GPU work, in one session (no human step)

On the GPU PC (Linux or WSL2, RTX 30-series or newer, 8 GB+), with `01-preprocessing/silver/advice_pairs.parquet` copied over:

```bash
git pull && uv sync --extra full && cp .env.example .env     # HF_TOKEN; accept the Gemma license first
uv run python pipeline.py                                     # add e.g. --gpu-mem 0.8 if the check asks for it
```

`pipeline.py` runs, in order:
1. the GPU check;
2. a 3-post trial in a throwaway folder;
3. recognition + P0 for both models;
4. their variance runs;
5. the **P-Oracle grid**: each candidate × every emotion × concept, about 14.8k replies, the longest step.

Every step resumes if interrupted. Setup and fixes are in "Small GPU" in `02-llm/scripts/README.md`.

When it finishes, copy `02-llm/gold/` (JSONL, `MANIFEST.json`, `run_record.json`) to the main machine **without opening the outputs**. Fill in the run-record table in `02-llm/README.md`. No more GPU work is needed.

## 3. Annotation (all 5)

| # | Who | Do |
| --- | --- | --- |
| 3.1 | Each annotator | Label your sheet alone (codebook §1-§7), about 59 posts each. Don't read the replies, open the thread, or look at any model output. This can start before or during step 2 |

## 4. Gold labels

| # | Who | Do |
| --- | --- | --- |
| 4.1 | Command | `wf adjudication-sheet` → `annotations_raw.csv` + `work/annotation/adjudication.xlsx` (only the double-annotated disagreements; agreed fields prefilled) |
| 4.2 | Ethan | Fill the blank `final_*` cells |
| 4.3 | Command | `wf finalize-labels` → `annotations_adjudicated.csv`, `merges.json`, `03-evaluation/results/label_finalization.md`, and the **P-Oracle replies selected from the grid** by each post's gold label |

If 4.3 stops:
- **"no merge target"**: a label has fewer than 15 posts. Set its target under `[merges]` in `config.toml` (e.g. `pakikisama_strain = "hiya"`), log it in codebook §10, and rerun.
- **"κ gate failed"**: revise the label's definition and re-adjudicate it, add a merge, or rerun with `--accept-kappa` and log the reason in `analysis-plan.md` §11.
- **"grid has no successful reply"**: a grid generation failed. Rerun `pipeline.py` on the GPU PC (it resumes), copy the grid back, and rerun.

**Labels are now frozen.** From here on, the model outputs may be opened.

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
