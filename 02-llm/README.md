# 02-llm — Silver → Gold

Runs Gemma and SEA-LION on the silver data for the two study tasks (emotion recognition and advice generation) and writes frozen benchmark artifacts to `gold/`. Everything else in this stage (pre-screen, annotation, rating) is done by the authors.

## Layout

- `codebook/`: author instruments, no LLM: annotation codebook (`codebook.md`, including the pre-screen), advice-strategy codebook, rating rubric, and the sheet templates (pre-screen, annotation, rating, best fit)
- `scripts/`: runners for the two LLM tasks: Ollama for the 30-post pilot on a Mac, vLLM for the full run on a CUDA GPU (see `scripts/README.md`)
- `pilot/`: pilot post ids (committed); pilot outputs and pilot labels stay local
- `prompts/`: versioned prompt templates (one file per task per version; see `prompts/README.md`)
- `work/`: Excel sheets for the team, made and read back by `03-evaluation/workflow.py` (gitignored; see `../RUNBOOK.md`)
- `gold/`: the gold layer (see `gold/README.md`). Label ids and one-line definitions live in `gold/emotion-recognition/label-schema.json`, the single source shared by the codebook and the prompts.

## Run record (fill in per run)

Record these for every run so it can be reproduced. `scripts/run.py` writes all of them to `run_record.json` in the run's output folder; copy them here for the final run:

| Field           | Gemma | SEA-LION |
| --------------- | ----- | -------- |
| Model ID        |       |          |
| Revision / hash |       |          |
| Temperature     |       |          |
| Top-p           |       |          |
| Max new tokens  |       |          |
| Seed            |       |          |
| Backend         |       |          |
| Precision       |       |          |
| Prompt version  |       |          |
| Prompt files    | `emotion-recognition_v3`, `advice-p0_v2`, `advice-oracle_v2` | same |
