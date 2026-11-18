# Prompts

One file per task per version (`<task>_v<N>.md`). A released version is never edited; any change creates `_v<N+1>`.

## File format

Each file has YAML front-matter (`task`, `version`, `placeholders`, `output_format`) followed by the prompt body. **Only the body is sent to the model.**

- Placeholders are `{{name}}` and are filled by plain string replacement (not `str.format`, because the bodies contain JSON braces).
- The whole body is sent as **one user turn with no system prompt**. Gemma 3 has no system role, and Gemma-SEA-LION-v4 uses the same chat template, so both models get the exact same string.
- Lists (`{{emotion_list}}`, `{{concept_list}}`, ...) are rendered from `../gold/emotion-recognition/label-schema.json` in schema order, one line per item: `- <id>: <definition>`.

## Shared placeholders

| Placeholder         | Rendered as                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `{{post}}`          | `Problem/Goal: ...` / `Context: ...` / `Previous Attempts: ...` (the last line only if not null), separated by blank lines. Flair is never included. |
| `{{reply_language}}`| `reply_language` from the schema                                                                  |
| `{{reply}}`, `{{reply_a}}`, `{{reply_b}}` | Reply text. Community replies are cleaned first (see `../codebook/judge-rubric.md`). |

## Files

| File                        | Task                                      | Run by      |
| --------------------------- | ----------------------------------------- | ----------- |
| `prescreen_v1.md`           | Exclusion flags + cultural-concept presence, for sampling | judge model |
| `emotion-recognition_v1.md` | Task A: emotion + concept + confidence    | Gemma, SEA-LION |
| `advice-p0_v1.md`           | Task B, baseline: post only               | Gemma, SEA-LION |
| `advice-oracle_v1.md`       | Task B, oracle: post + gold emotion/concept | Gemma, SEA-LION |
| `judge-rating_v1.md`        | Rubric scores for one reply               | judge model |
| `judge-pairwise_v1.md`      | Best overall fit, A vs. B                 | judge model |
| `strategy-coding_v1.md`     | Dominant advice strategy (drift)          | judge model |
