# 02-llm/scripts — running the two LLM tasks

These scripts run **only** the two study tasks on Gemma and SEA-LION:

| Task | `--task` | Prompt |
| --- | --- | --- |
| A. Emotion recognition | `recognition` | `prompts/emotion-recognition_v4.md` |
| B. Advice, baseline | `advice-p0` | `prompts/advice-p0_v5.md` |
| B. Advice, oracle | `advice-oracle` | `prompts/advice-oracle_v5.md` (adjudicated gold labels only, never Task A output) |

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
- reply length against the 110-word cap of the v2 advice prompts (and replies under half the 90-word target), truncation, and the community reply length on the same posts; an `oracle_grid/` folder, if present, is summarized as one "P-Oracle grid" row per model;
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
  - `fp32` is never picked by `auto`. It is the T4 route: `--precision fp32 --tp 2` on Kaggle's 2× T4 (below).
  - Whatever you choose, use the same value for every full-run command, and record it.
- **`sample_posts.csv`:** this comes from `03-evaluation/workflow.py sample`, the stratified ~236 posts with the pilot ids excluded. Recognition and P0 advice need only this file, so they can run before annotation (keep the outputs unopened until `finalize-labels`). P-Oracle needs `annotations_adjudicated.csv` too.

## Small GPU (8–11 GB), the team's setup

This works with any **RTX 30-series or newer** card that has **8–11 GB**. `auto` picks **FP8 weight-only** because the VRAM is under 11 GB. That is the study precision: report it as "FP8 weight-only (vLLM)". Cards older than RTX 30 (GTX, RTX 20xx, T4) and cards under 8 GB are stopped by `--check`.

**Windows:** vLLM needs WSL2. Once, in PowerShell as admin, run `wsl --install -d Ubuntu-24.04` and restart. Install the normal NVIDIA Windows driver; don't install a Linux driver inside WSL. In Ubuntu, `nvidia-smi` should show the card. Then install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`), clone the repo **inside the Linux home folder** (`cd ~ && git clone …`, not under `/mnt/c`, which is slow and has broken vLLM's startup on the team's PC), copy `.env` and `advice_pairs.parquet` into it, and continue below.

The `full` profile sets `enforce_eager = true` and `max_num_seqs = 16` (`config.toml`). They change only vLLM's working memory, not the weights, precision, context length or decoding, and are what lets FP8 Gemma-3-4B fit its 8k KV cache on an 8 GB card. Both are saved under `engine` in `run_record.json`.

```bash
uv sync --extra full && cp .env.example .env       # HF_TOKEN; accept the Gemma license on Hugging Face first
nvidia-smi                                          # card, driver, CUDA 12.x
uv run python 02-llm/scripts/run.py --profile full --check --model both
#   prints the GPU, compute capability, total/free VRAM and the precision, and stops on a problem

# Trial: 3 posts + 1 grid post into a throwaway folder (doesn't touch 02-llm/gold). Confirms both models
# load and fit and that every prompt runs. These are study posts: don't open the trial outputs either.
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both \
    --limit 3 --out-dir /tmp/payo-trial
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle-grid --model both \
    --limit 1 --out-dir /tmp/payo-trial
# Real runs, all before annotation (rerunning resumes). The root pipeline.py runs exactly these:
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both
uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model both --variance
uv run python 02-llm/scripts/run.py --profile full --task advice-oracle-grid --model both   # 71 posts x 96, the longest
```

If something fails:

| Message | Fix |
| --- | --- |
| `only X of Y GB free ... pass --gpu-mem 0.8` | The display or another app uses VRAM. Close browsers and games, or pass the suggested `--gpu-mem` on **every** command |
| `Engine core initialization failed`, with `determine_available_memory` / `_initialize_kv_caches` or "No available memory for the cache blocks" above it | Not enough VRAM left for the 8k KV cache. Close every other GPU app, rerun with the highest `--gpu-mem` the check accepts, then `rm -rf /tmp/payo-trial` and retry. If it still fails, use `--precision bnb4` for **every** run and log it in `02-llm/README.md` |
| CUDA out of memory while loading | Rerun with `--gpu-mem 0.85`. If it still fails, move the files in `02-llm/gold/` away and use `--precision bnb4` (4-bit) for **every** run; log it in `02-llm/README.md` |
| `already has outputs at precision ...` | One file got two precisions. Use the same `--precision` as before, or move the file away to start over |
| `compute capability 7.5 < 8.0` on a T4 | Use Kaggle's **GPU T4 x2** with `--precision fp32 --tp 2` (section above) |
| `fp32 needs about 28 GB across GPUs` | One T4 isn't enough: pick **GPU T4 x2** and pass `--tp 2` |
| `To serve at least one request with the model's max seq len (8192) ... KV cache` on 2× T4 | Pass the Kaggle flags (`--gpu-mem 0.95 --max-batched-tokens 1024`) |
| `gated repo` / 401 | Accept the Gemma license on Hugging Face with the account whose token is in `.env` |

Don't open or print the JSONL outputs until `finalize-labels` (RUNBOOK, blindness rule). The run prints only counts.

## Free GPU: Kaggle 2× T4 (FP32)

With no RTX 30-series card, use Kaggle's free **GPU T4 x2** (30 GPU-hours a week, 12-hour sessions) through `kaggle_full_run.ipynb`. It runs `pipeline.py --precision fp32 --tp 2 --gpu-mem 0.95 --max-batched-tokens 1024 --max-num-seqs 32 --cuda-graphs decode`: the weights are loaded unquantized in float32 and each model is split across both GPUs (tensor parallel 2). `tensor_parallel_size` is saved under `engine` in `run_record.json`.

Why not something else:

- **Free Colab** has a single T4 (15 GB). FP32 weights alone are 17.2 GB, so they don't fit.
- **BF16 and FP8** need compute capability 8.0. The T4 is 7.5.
- **FP16** fits, but Gemma 3 was trained in BF16 and gives non-finite logits in FP16 (both models here are Gemma 3).

Steps (the notebook's first cell has the details):

1. Upload `advice_pairs.parquet` as a **private** Kaggle dataset, and add `HF_TOKEN` under Add-ons → Secrets.
2. Import `02-llm/scripts/kaggle_full_run.ipynb`, set Accelerator **GPU T4 x2** and Internet on, and pin `COMMIT`.
3. Use **Save Version → Save & Run All**. The clone lives in `/tmp`, and only `02-llm/gold/` is copied to the saved output.
4. **Resume:** FP32 on a T4 is slow, and the P-Oracle grid may need 2–3 sessions. Attach the previous version's output as an input, set `PREVIOUS`, and run again. Finished generations are skipped.
5. Download `gold/` to the main machine **without opening it**.

On a T4, vLLM's Triton attention kernel needs 80 KB of shared memory for Gemma 3 at FP32, but the T4 has 64 KB. A small vLLM plugin (`payo_llm/fp32_attention.py`, registered in `pyproject.toml`) makes vLLM use 16-token attention tiles for FP32 only (48 KB). The math is the same; only the blocking of the attention sum changes. BF16 and FP8 runs are untouched. `run_record.json` records it as `engine.fp32_attention_tile`.

At FP32, vLLM's default 8,192-token prefill chunk reserves ~3.7 GB of activations per GPU and leaves too little for an 8k KV cache, so the Kaggle flags also set a smaller prefill chunk (`--max-batched-tokens 1024`; longer prompts are read in several chunks) and `--gpu-mem 0.95`. For speed, `--max-num-seqs 32` lets more requests run at once, and `--cuda-graphs decode` replays each decode step as a captured CUDA graph (torch.compile stays off, so the same kernels run with less launch overhead). With the earlier eager settings, one P-Oracle grid post took ~8.5 min per model. All of these are recorded under `engine` and change how fast the work runs, not what is computed.

**Splitting across two accounts:** the P-Oracle grid is ~95% of the work. To halve the wall time, run `--model gemma` on one Kaggle account and `--model sealion` on another at the same time (the notebook's `MODEL` setting), with the same `COMMIT` and flags. Then copy one `gold/` into `02-llm/gold/` and run `uv run python 02-llm/scripts/merge_gold.py <the other gold folder>`. It refuses to merge if the precision or engine settings differ, or if a model appears on both sides, and it rewrites `MANIFEST.json`.

Use `--precision fp32 --tp 2 --gpu-mem 0.95 --max-batched-tokens 1024 --max-num-seqs 32 --cuda-graphs decode` for **every** full-run command, in every session and on both machines. The precision guard stops a file from mixing precisions. Before the real runs, check the trial's `latency_s` to estimate how many sessions you'll need. If vLLM hits another T4-specific failure, log the fix in `02-llm/README.md`.

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
<out>/advice-generation/replies_<model>_oracle.jsonl      # selected from the grid by finalize-labels (v1.4)
<out>/advice-generation/oracle_grid/replies_<model>.jsonl # advice-oracle-grid: every emotion x concept per candidate
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
