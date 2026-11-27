# Prompts

LLMs are used **only for the two study tasks**: emotion recognition (Task A) and advice generation (Task B). Scoring, strategy coding and pre-screening are done by the authors (see `../codebook/`), and all statistics are in `../../03-evaluation/`.

One file per task per version (`<task>_v<N>.md`). A released version is never edited; any change creates `_v<N+1>`.

## File format

Each file has YAML front-matter (`task`, `version`, `placeholders`, `output_format`) followed by the prompt body. **Only the body is sent to the model.**

- Placeholders are `{{name}}` and are filled by plain string replacement (not `str.format`, because the bodies contain JSON braces).
- The whole body is sent as **one user turn with no system prompt** (Gemma 3 has no system role). Both models get the same user content; each backend wraps it in that model's **own chat template** (SEA-LION ships its own). The scripts save each template to `chat_templates/<model>.txt` and log its sha256.
- `{{emotion_list}}` and `{{concept_list}}` are rendered from `../gold/emotion-recognition/label-schema.json` in schema order, one line per item: `- <id>: <definition>`.

## Shared placeholders

| Placeholder          | Rendered as                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `{{post}}`           | `Problem/Goal: ...` / `Context: ...` / `Previous Attempts: ...` (the last line only if not null), separated by blank lines. Flair is never included. |
| `{{reply_language}}` | `reply_language` from the schema                                                                  |

## Files

| File                        | Task                                         | Runs on         |
| --------------------------- | -------------------------------------------- | --------------- |
| `emotion-recognition_v1.md` | Task A: emotion + cultural concept + confidence (pilot only; superseded) | Gemma, SEA-LION |
| `emotion-recognition_v2.md` | Task A, as v1 plus concept parsing rule (pilot 2026-11-23; superseded) | Gemma, SEA-LION |
| `emotion-recognition_v3.md` | Task A, schema v2 (konsensya added, pakikisama_strain retired; pilot 2026-11-27; superseded) | Gemma, SEA-LION |
| `emotion-recognition_v4.md` | Task A, as v3 plus a rule that `none_practical` (not `none`) is the no-emotion id | Gemma, SEA-LION |
| `advice-p0_v1.md`           | Task B, baseline: post only (~150w, pilot only; superseded) | Gemma, SEA-LION |
| `advice-p0_v2.md`           | Task B, baseline: post only (calibrated to ~90 words, max 110; pilot 2026-11-27; superseded) | Gemma, SEA-LION |
| `advice-p0_v3.md`           | Task B, baseline: as v2 with a firm 110-word cap, no formula opener/sign-off/emoji, no persona claims (pilot 2026-11-27; superseded) | Gemma, SEA-LION |
| `advice-p0_v4.md`           | Task B, baseline: one paragraph of 4–5 sentences, Taglish first sentence (pilot 2026-11-27, stopped after 16 replies; superseded) | Gemma, SEA-LION |
| `advice-p0_v5.md`           | Task B, baseline: v2 wording plus a no-emoji / no-persona line. Length, openers and formatting tells (markdown, dashes, quotes, spacing, "Honestly,", stock sign-offs; v1.9) are normalized by `scripts/payo_llm/cleanup.py` on every rater-facing reply | Gemma, SEA-LION |
| `advice-oracle_v1.md`       | Task B, oracle: post + gold emotion and concept (~150w; superseded) | Gemma, SEA-LION |
| `advice-oracle_v2.md`       | Task B, oracle: post + gold emotion and concept (calibrated to ~90 words, max 110; superseded) | Gemma, SEA-LION |
| `advice-oracle_v3.md`       | Task B, oracle: advice-p0_v3 plus the unchanged gold-label Background block (superseded) | Gemma, SEA-LION |
| `advice-oracle_v4.md`       | Task B, oracle: advice-p0_v4 plus the unchanged gold-label Background block (never run; superseded) | Gemma, SEA-LION |
| `advice-oracle_v5.md`       | Task B, oracle: advice-p0_v5 plus the unchanged gold-label Background block | Gemma, SEA-LION |

Scale: ~300 posts × 2 models = 600 recognition outputs; ~300 posts × 2 models × 2 conditions = 1,200 replies.
