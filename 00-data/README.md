# 00-data — Bronze layer

Raw data, exactly as collected. **Never edit files here.** Every later layer is regenerated from this one by code.

## Medallion contract (whole repo)

| Layer  | Location                   | Written by             | Format         | Rule                                                      |
| ------ | -------------------------- | ---------------------- | -------------- | --------------------------------------------------------- |
| Bronze | `00-data/bronze/`          | collection only        | original jsonl | Immutable; a new pull goes in a new dated folder          |
| Silver | `01-preprocessing/silver/` | preprocessing notebook | parquet        | Fully regenerable from bronze by code; no manual edits    |
| Gold   | `02-llm/gold/`             | LLM scripts            | jsonl          | Frozen per version; any change bumps version and manifest |

Rules that apply to every layer:

- **Stable keys.** Reddit `id` (posts) and `comment_id` (comments) are carried through every layer, so any gold row can be traced back to its bronze source.
- **PII boundary.** Reddit usernames (`author`) exist only in bronze. Silver and gold store no author information, and PII in text is replaced with placeholders (see `01-preprocessing/README.md`).
- **Manifests.** Each layer has a `MANIFEST.json` with sha256, row count, and byte size per file. Data files are gitignored; manifests are committed so anyone can check they have the same data.

## Bronze contents

### `bronze/reddit-adviceph_2026-11-06/`

Source: Reddit, r/adviceph. Folder date = date the dump was added to this repo (update if the actual scrape date differs).

**`posts.jsonl`**: 3,786 rows, one submission per line

| Field          | Type | Notes                         |
| -------------- | ---- | ----------------------------- |
| `id`           | str  | Reddit post ID (primary key)  |
| `subreddit`    | str  |                               |
| `title`        | str  |                               |
| `selftext`     | str  | Post body                     |
| `author`       | str  | **PII** (bronze only)         |
| `score`        | int  |                               |
| `num_comments` | int  |                               |
| `created_utc`  | int  | Unix timestamp                |
| `created_iso`  | str  | ISO-8601 timestamp            |
| `url`          | str  |                               |
| `permalink`    | str  |                               |

**`comments.jsonl`**: 62,992 rows, one comment per line

| Field                | Type      | Notes                                         |
| -------------------- | --------- | --------------------------------------------- |
| `comment_id`         | str       | Reddit comment ID (primary key)               |
| `post_id`            | str       | FK → `posts.id`                               |
| `parent_id`          | str       | Parent post or comment ID                     |
| `parent_kind`        | str       | `post` or `comment`                           |
| `kind`               | str       | Reddit thing kind (e.g. `t1`)                 |
| `depth`              | int       | 0 = top-level reply to the post               |
| `path`               | list      | Ancestor comment IDs                          |
| `subreddit`          | str       |                                               |
| `author`             | str       | **PII** (bronze only); includes AutoModerator |
| `body`               | str       |                                               |
| `score`              | int       |                                               |
| `created_utc`        | int       |                                               |
| `created_iso`        | str       |                                               |
| `permalink`          | str       |                                               |
| `collapsed_children` | list      |                                               |
| `collapsed_count`    | int\|null |                                               |

### `bronze/reddit-adviceph-flair_<date>/` (to be collected)

Flair is missing from the original dump. `00-data/collect_flair.py` fetches it from Reddit's OAuth API (`/api/info`) for every post ID and writes it here as a separate snapshot. The script won't overwrite an existing snapshot.

```bash
cp .env.example .env   # fill in REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET from a Reddit "script" app
uv run python 00-data/collect_flair.py
```

**`flair.jsonl`**: one row per post Reddit returns

| Field                    | Type        | Notes                              |
| ------------------------ | ----------- | ---------------------------------- |
| `id`                     | str         | FK → `posts.id`                    |
| `link_flair_text`        | str \| null | Flair label                        |
| `link_flair_template_id` | str \| null | Flair template ID                  |
| `retrieved_at`           | str         | ISO-8601 time of the API call      |

Flair is collected later than the posts, and moderators can change flair, so it's a later snapshot.

## Verifying your copy

```bash
cd 00-data/bronze
shasum -a 256 reddit-adviceph_2026-11-06/*.jsonl   # compare against MANIFEST.json
wc -l reddit-adviceph_2026-11-06/*.jsonl
```
