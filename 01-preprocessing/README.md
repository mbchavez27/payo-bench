# 01-preprocessing — Bronze → Silver

Turns raw bronze data (`00-data/bronze/`) into clean, conformed tables in `silver/`. Silver can always be rebuilt by running the preprocessing notebook from top to bottom.

## Planned outputs (`silver/`)

| File               | Description                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `posts.parquet`    | Posts with removed/deleted/empty ones dropped, normalized text, language tags                 |
| `comments.parquet` | Bot/AutoModerator comments removed, `author` replaced with salted-hash `author_id`            |
| `threads.parquet`  | Each post joined with its top-level human advice comments                                     |
| `MANIFEST.json`    | sha256, rows, bytes per file, plus the notebook and git commit that produced them             |

## Attrition table

Fill this in as the pipeline is built. The paper's data section reports it.

| Step | Posts in | Posts out | Comments in | Comments out | Reason |
| ---- | -------- | --------- | ----------- | ------------ | ------ |
| Bronze | 3,786 | — | 62,992 | — | Raw dump |
