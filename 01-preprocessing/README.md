# 01-preprocessing — Bronze → Silver

Turns raw bronze data (`00-data/bronze/`) into one clean table in `silver/`. Silver can always be rebuilt by running the notebook top to bottom:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01-preprocessing/01-preprocessing.ipynb
```

The first run takes about 10–15 minutes because of name detection. Later runs reuse the NER cache and take under a minute.

## Output: `silver/advice_pairs.parquet`

One row per r/adviceph post, paired with its top-voted human reply.

| Column              | Type           | Source / notes                                        |
| ------------------- | -------------- | ----------------------------------------------------- |
| `post_id`           | string         | Reddit post ID (primary key, traces back to bronze)   |
| `flair`             | string \| null | Post flair from the flair bronze snapshot             |
| `problem_goal`      | string         | Parsed from the post's **Problem/Goal** header        |
| `context`           | string         | Parsed from the post's **Context** header             |
| `previous_attempts` | string \| null | Parsed from **Previous Attempts** (optional)          |
| `created_at`        | timestamp[UTC] | Post creation time                                    |
| `reply_comment_id`  | string         | Reddit comment ID of the top reply (traces to bronze) |
| `reply_body`        | string         | Top reply text                                        |
| `reply_score`       | int64          | Top reply score (snapshot at scrape time)             |

No author information is stored. All text columns are PII-masked (see [Cleaning](#cleaning)).

## Inclusion criteria

1. **Live post:** selftext is not `[removed]`, `[deleted]`, or empty.
2. **Not removed by AutoModerator:** the post has no AutoModerator removal notice. Removal reasons include wrong post format, links in the post, a body under 400 characters, and low-karma or low-quality accounts. AutoModerator's standard rules notice, which appears on nearly every post, is ignored.
3. **Template present:** both Problem/Goal and Context headers are found and non-empty. Headers must start a line; spelling variants of Previous Attempts are accepted.
4. **Not a duplicate:** see below.
5. **Eligible reply:** top-level comment, not AutoModerator, not `[removed]`/`[deleted]`, not written by the original poster.
6. **Top reply:** highest score; ties go to the earliest reply.
7. **Substantive:** the top reply has at least 10 words. There's no fallback to a lower-voted reply.

Language is not filtered.

## Cleaning

### Deduplication

Two posts are duplicates when their Problem/Goal and Context match after lowercasing and collapsing punctuation and whitespace. The earliest post is kept, since later copies are resubmissions. The notebook also checks that no two silver rows share a reply.

### PII removal

**Regex** replaces structured identifiers with typed placeholders:

| PII                                                        | Placeholder |
| ---------------------------------------------------------- | ----------- |
| Email address                                              | `[EMAIL]`   |
| URL (incl. bare `facebook.com/…` etc.)                     | `[URL]`     |
| PH mobile (`09xx…`, `+63 9xx…`) and Metro Manila landline   | `[PHONE]`   |
| `@handle`                                                  | `[HANDLE]`  |
| Reddit `u/username`                                        | `u/[USER]`  |

**Person names** become `[NAME]`, in three stages:

1. **Detect.** Candidates come from calamanCy NER (`tl_calamancy_md` 0.2.0, pinned to Hugging Face commit `4960655`) plus a cue regex: a capitalized word after *si, ni, kay, sina, nina, kina, Ate, Kuya, Tita, Tito, Lola, Lolo, Ma'am, Sir, Mr./Ms./Mrs., Dr., Atty., named, name is, pangalan*.
2. **Auto-prefill.** Each candidate gets a suggested `replace`/`keep`. It's kept if it's a kinship term, honorific, religious term or brand; if it isn't name-shaped (emoji, initials, ALL CAPS); or if it's a common word (its lowercase form appears ≥ 2 times in the corpus).
3. **Manual pass.** A reviewer goes through `pii/name_review.csv` (see below). Approved names are replaced everywhere they occur as a whole word.

The notebook then checks that no PII pattern, approved name, or bronze username is left in any text column.

### Doing the manual pass

1. Open `01-preprocessing/pii/name_review.csv`. It has one row per candidate, with its `source` (`ner`, `cue`, or both), `count`, and an `example` snippet.
2. For each row, set `decision` to `replace` (a private person's name) or `keep` (not a name, or a public figure, brand, drug, place, or title). Put your initials in `reviewed_by` and add a `note` if useful.
3. Rerun the notebook. Your decisions are kept; only new candidates get `reviewed_by = auto`.

`pii/` holds real names (`name_review.csv`, `ner_cache.parquet`), so it's gitignored. Its sha256 is recorded in `silver/MANIFEST.json`.

## Attrition

From `silver/attrition.csv`:

| Step                  | Posts | Reason                                                 |
| --------------------- | ----: | ------------------------------------------------------ |
| bronze                | 3,786 | Raw dump                                               |
| live posts            | 3,223 | selftext not [removed]/[deleted]/empty                 |
| not auto-removed      | 3,205 | No AutoModerator removal notice                        |
| template present      | 2,692 | Problem/Goal and Context both non-empty                |
| deduplicated          | 2,660 | Earliest copy of identical Problem/Goal + Context kept |
| has eligible reply    | 2,214 | ≥ 1 human top-level reply not by OP                    |
| substantive top reply | 1,881 | Top reply ≥ 10 words                                   |

## PII replacements

From `silver/pii_counts.csv` (before the manual pass; `[NAME]` will change once it's done):

| Placeholder | Replacements |
| ----------- | -----------: |
| `[EMAIL]`   | 2            |
| `[URL]`     | 12           |
| `[PHONE]`   | 2            |
| `[HANDLE]`  | 2            |
| `u/[USER]`  | 0            |
| `[NAME]`    | 212          |

## Known gaps

- `flair` is null until `00-data/collect_flair.py` has been run (it needs Reddit API credentials in `.env`). Rerun the notebook afterwards to fill it in.
- Posts that put every header on one line aren't parsed and are excluded.
- Names that neither NER nor a cue word catches can remain. A name that is also a common word (e.g. *May*) is kept unless a reviewer marks it.
