# 02-llm — Silver → Gold

Runs Gemma and SEA-LION on the silver data and writes frozen benchmark artifacts to `gold/`.

## Layout

- `prompts/`: versioned prompt templates (one file per task per version)
- `gold/`: the gold layer (see `gold/README.md`)

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
