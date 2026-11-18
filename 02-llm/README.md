# 02-llm — Silver → Gold

Runs Gemma and SEA-LION on the silver data for the two study tasks (emotion recognition and advice generation) and writes frozen benchmark artifacts to `gold/`. Everything else in this stage (pre-screen, annotation, rating) is done by the authors.

## Layout

- `codebook/`: author instruments, no LLM: annotation codebook (`codebook.md`, including the pre-screen), advice-strategy codebook, rating rubric, and the sheet templates (pre-screen, annotation, rating, best fit)
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
