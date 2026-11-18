# Prompts

LLMs are used **only for the two study tasks**: emotion recognition (Task A) and advice generation (Task B). Scoring, strategy coding and pre-screening are done by the authors (see `../codebook/`), and all statistics are in `../../03-evaluation/`.

One file per task per version (`<task>_v<N>.md`). A released version is never edited; any change creates `_v<N+1>`.

## File format

Each file has YAML front-matter (`task`, `version`, `placeholders`, `output_format`) followed by the prompt body. **Only the body is sent to the model.**

- Placeholders are `{{name}}` and are filled by plain string replacement (not `str.format`, because the bodies contain JSON braces).
- The whole body is sent as **one user turn with no system prompt**. Gemma 3 has no system role, and Gemma-SEA-LION-v4 uses the same chat template, so both models get the exact same string.
- `{{emotion_list}}` and `{{concept_list}}` are rendered from `../gold/emotion-recognition/label-schema.json` in schema order, one line per item: `- <id>: <definition>`.

## Shared placeholders

| Placeholder          | Rendered as                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `{{post}}`           | `Problem/Goal: ...` / `Context: ...` / `Previous Attempts: ...` (the last line only if not null), separated by blank lines. Flair is never included. |
| `{{reply_language}}` | `reply_language` from the schema                                                                  |

## Files

| File                        | Task                                         | Runs on         |
| --------------------------- | -------------------------------------------- | --------------- |
| `emotion-recognition_v1.md` | Task A: emotion + cultural concept + confidence | Gemma, SEA-LION |
| `advice-p0_v1.md`           | Task B, baseline: post only                  | Gemma, SEA-LION |
| `advice-oracle_v1.md`       | Task B, oracle: post + gold emotion and concept | Gemma, SEA-LION |

Scale: ~300 posts × 2 models = 600 recognition outputs; ~300 posts × 2 models × 2 conditions = 1,200 replies.
