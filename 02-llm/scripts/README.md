# 02-llm/scripts — running the two LLM tasks

These scripts run **only** the two study tasks on Gemma and SEA-LION:

| Task | `--task` | Prompt |
| --- | --- | --- |
| A. Emotion recognition | `recognition` | `prompts/emotion-recognition_v1.md` |
| B. Advice, baseline | `advice-p0` | `prompts/advice-p0_v1.md` |
| B. Advice, oracle | `advice-oracle` | `prompts/advice-oracle_v1.md` (adjudicated gold labels only, never Task A output) |

There are two profiles:

| | `pilot` | `full` |
| --- | --- | --- |
| Machine | MacBook (Apple Silicon, 16 GB) | Linux + NVIDIA GPU (8 GB+) |
| Backend | Ollama (vLLM doesn't run on macOS) | vLLM |
| Weights | GGUF Q8_0 (`config.toml` → `ollama_tag`) | HF weights at a pinned revision (`hf_id`, `hf_revision`) |
| Precision | Q8_0 | `auto`: bf16 if VRAM ≥ 11 GB, else FP8 weight-only |
| Posts | 30 (`02-llm/pilot/pilot_posts.csv`) | the ~236-post sample (`02-llm/gold/sample_posts.csv`) |
| Output | `02-llm/pilot/outputs/` (local only) | `02-llm/gold/` + `MANIFEST.json` |

**The pilot is not a result.** Its backend and quantization differ from the full run. Use it to check prompts, parsing, invalid-output rates, reply length, timing and label balance. Never analyze its outputs as findings.

## Pilot on the Mac (Ollama)

```bash
brew install ollama            # or install the Ollama app from ollama.com
ollama serve                   # skip if the app is running
ollama pull hf.co/ggml-org/gemma-3-4b-it-GGUF:Q8_0
ollama pull hf.co/aisingapore/Gemma-SEA-LION-v4-4B-VL-GGUF:Q8_0

uv sync
uv run python 02-llm/scripts/make_pilot_sample.py        # once: 30 post ids + annotation sheet
uv run python 02-llm/scripts/run.py --profile pilot --check
uv run python 02-llm/scripts/run.py --profile pilot --task recognition,advice-p0 --model both
```

For the pilot **P-Oracle** run, first label the pilot posts. Use `03-evaluation/workflow.py pilot-sheets`, then `pilot-merge`, then `pilot-adjudicate` (RUNBOOK §1), which writes `02-llm/pilot/pilot_annotations.csv`. Then run:

```bash
uv run python 02-llm/scripts/run.py --profile pilot --task advice-oracle --model both
```

Posts without a label are skipped and counted.

### Pilot report

```bash
uv run python 02-llm/scripts/pilot_report.py      # prints and writes 02-llm/pilot/outputs/pilot_report.md
```

You can run it after any subset of tasks; tasks that haven't run yet are shown as "not run". It reports:

- completeness and errors;
- Task A validity, the predicted-label spread (label collapse) and confidence spread;
- reply length against the 180-word cap, truncation, and the community reply length on the same posts;
- the P-Oracle − P0 length gap;
- latency and prompt length against `num_ctx`;
- from `pilot_annotations.csv`: retention, the silver posts needed for 300, per-label balance with a pool estimate against the merge threshold, and annotator disagreements (when a post has more than one annotator row).

A **Flags** list at the top collects every threshold that tripped (the constants are in `payo_llm/pilot.py`).

**It never compares model output with the gold labels.** The pilot is not a result, so accuracy comes only from the full run.

Rough time on an M-series Mac: about 1–2 minutes per post per model for all three tasks, so roughly 1–2 hours for the whole pilot.

## Full run on the GPU box (vLLM)

```bash
uv sync --extra full           # installs vllm + bitsandbytes (Linux only)
cp .env.example .env           # set HF_TOKEN; accept the Gemma license on Hugging Face first
uv run python 02-llm/scripts/run.py --profile full --check --model both

uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both
# after annotation + adjudication (gold/emotion-recognition/annotations_adjudicated.csv):
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle --model both
# variance estimate (sampled decoding, 3 seeds, first 30 posts):
uv run python 02-llm/scripts/run.py --profile full --task all --model both --variance
```

- **Precision:**
  - `auto` picks the precision once and uses it for both models.
  - If FP8 fails to load on your card, rerun with `--precision bnb4`, the bitsandbytes 4-bit fallback.
  - Whatever you choose, use the same value for every full-run command, and record it.
- **`sample_posts.csv`:** this comes from `03-evaluation/workflow.py sample`, the stratified ~236 posts with the pilot ids excluded. Recognition and P0 advice need only this file, so they can run before annotation (keep the outputs unopened until `finalize-labels`). P-Oracle needs `annotations_adjudicated.csv` too.

## Behavior shared by both profiles

- **One model at a time.** `--model both` runs Gemma and then SEA-LION in separate processes, so memory is fully released in between.
- **Resumable.** Every finished generation is appended to its JSONL file straight away. Rerunning the same command skips what's done and retries anything that ended with an `error`.
- **Greedy decoding** for the main run (`temperature = 0`, seed 42). `--variance` uses `temperature = 0.7`, `top_p = 0.95` and seeds 1–3, on the first 30 posts.
- **Prompt lengths.** A prompt that doesn't fit the context is not cut down. It is recorded with `error = "prompt_too_long"`.
- **Task A parsing.** The first JSON object is read and compared with the exact schema ids. `valid = false` means the emotion is not a schema label, and it counts as incorrect. Concept and confidence problems are listed in `invalid_reason`. One fixed exception, set before the full run: in the concept field only, the emotion ids in the schema's `concept_aliases` (`pakikisama_strain`, `utang_na_loob_guilt`) are read as the concept they name, and the record gets `concept_alias = true`. The pilot showed both models putting these ids in the concept field even when the prompt told them not to.
- **Run record.** `run_record.json` records each model's backend, exact weights (HF revision or Ollama digest), precision, decoding, prompt file hashes, library versions and git commit. The chat template each model used is saved to `chat_templates/<model>.txt`.

## Output files

```
<out>/emotion-recognition/predictions_<model>.jsonl
<out>/advice-generation/replies_<model>_p0.jsonl
<out>/advice-generation/replies_<model>_oracle.jsonl
<out>/variance/...
<out>/run_record.json
<out>/chat_templates/<model>.txt
```

Each JSONL line holds:

| Group | Fields |
| --- | --- |
| Identity | `post_id`, `model`, `task`, `condition`, `seed` |
| Backend and model | `backend`, `model_ref`, `precision` |
| Prompt | `prompt_version`, `prompt_sha256`, `rendered_sha256` |
| Settings | `decoding` |
| Output | `raw_output`, `n_prompt_tokens`, `n_output_tokens`, `finish_reason`, `truncated` |
| Timing and errors | `latency_s`, `error`, `created_at` |
| Task A only | `emotion`, `cultural_concept`, `confidence`, `valid`, `invalid_reason`, `concept_alias` |
| Task B only | `reply_words` |

Ollama counts only the prompt tokens it actually evaluated, so a cached prefix can make `n_prompt_tokens` lower than the true prompt length.

## Tests

```bash
uv run pytest          # rendering, parsing and backends (mocked); no model needed
```
