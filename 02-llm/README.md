# 02-llm — Silver → Gold

Runs Gemma and SEA-LION on the silver data and writes frozen benchmark artifacts to `gold/`.

## Layout

- `codebook/`: annotation codebook (`codebook.md`), advice-strategy codebook, judge rubric, annotation sheet template
- `prompts/`: versioned prompt templates (one file per task per version; see `prompts/README.md`)
- `gold/`: the gold layer (see `gold/README.md`). Label ids and one-line definitions live in `gold/emotion-recognition/label-schema.json`, the single source shared by the codebook and the prompts.

## Run record (fill in per run)

Record these for every run so it can be reproduced:

| Field           | Gemma | SEA-LION |
| --------------- | ----- | -------- |
| Model ID        |       |          |
| Revision / hash |       |          |
| Temperature     |       |          |
| Top-p           |       |          |
| Max new tokens  |       |          |
| Seed            |       |          |
| Prompt version  |       |          |
| Prompt files    | `emotion-recognition_v1`, `advice-p0_v1`, `advice-oracle_v1` | same |
