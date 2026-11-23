# Gold layer — datasheet

Frozen, task-ready benchmark artifacts. Results in the paper are computed from these files. Any change bumps the version and updates `MANIFEST.json`.

| Folder                 | Contents                                                          |
| ---------------------- | ----------------------------------------------------------------- |
| Path | Contents | Written by |
| --- | --- | --- |
| `sample_posts.csv` | the stratified sample: post id, arm, domain guess, stratum, post length | `workflow.py sample` |
| `emotion-recognition/label-schema.json` | label ids and definitions (the single source) | hand-edited, versioned |
| `emotion-recognition/prescreen.csv` | the screener's labels for the pre-screen pool | `sample` |
| `emotion-recognition/annotation_assignment.csv` | who labels which post (slot 2 = hidden overlap) | `sample` |
| `emotion-recognition/annotations_raw.csv` | every annotator's labels | `adjudication-sheet` |
| `emotion-recognition/adjudication.csv` | the adjudicator's decisions | `finalize-labels` |
| `emotion-recognition/annotations_adjudicated.csv` | gold labels (+ `emotion_final`, `emotion_type`); the P-Oracle input | `finalize-labels` |
| `emotion-recognition/merges.json` | the scoring-time merge map | `finalize-labels` |
| `emotion-recognition/predictions_<model>.jsonl` | Task A outputs | `run.py` (gitignored) |
| `advice-generation/replies_<model>_{p0,oracle}.jsonl` | Task B outputs | `run.py` (gitignored) |
| `author-ratings/rated_posts.csv`, `rating_assignment.csv` | the 80 rated posts and who rates them | `rating-sheets` |
| `author-ratings/items_key.csv`, `bestfit_key.csv` | the blinding keys (raters don't open them) | `rating-sheets` |
| `author-ratings/ratings.csv`, `bestfit.csv` | rubric scores, strategy codes, best-fit choices | `collect-ratings` |

## Datasheet (fill in before release)

- **Version:**
- **Composition:** number of instances per task and split; rated subset size (`N_eval`) and double-rated share
- **Label schema:** see `emotion-recognition/label-schema.json`
- **Provenance:** every row keeps its Reddit `id` / `comment_id` back to bronze
- **PII:** no author information; emails, URLs, phone numbers, handles and reviewed person names are replaced with placeholders (`[EMAIL]`, `[URL]`, `[PHONE]`, `[HANDLE]`, `u/[USER]`, `[NAME]`)
- **Intended use:**
- **License:**
