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

**Study precision (decided 2026-11-28, before any full-run output):** if the full run uses Kaggle's free 2× T4, it is **FP32 (vLLM, tensor parallel 2)** for every run (`--precision fp32 --tp 2 --gpu-mem 0.95 --max-batched-tokens 1024 --max-num-seqs 32 --cuda-graphs decode`, 16-token FP32 attention tiles; `scripts/README.md`, "Free GPU"). FP16 was ruled out because Gemma 3 gives non-finite logits in it. On an RTX 30-series card or newer with 8–11 GB, it stays FP8 weight-only. Whichever hardware runs first fixes the precision for the whole study.

Record these for every run so it can be reproduced. `scripts/run.py` writes all of them to `run_record.json` in the run's output folder; copy them here for the final run:

| Field           | Gemma | SEA-LION |
| --------------- | ----- | -------- |
| Model ID        | `google/gemma-3-4b-it` | `aisingapore/Gemma-SEA-LION-v4-4B-VL` |
| Revision / hash | `093f9f388b31de276ce2de164bdc2081324b9767` | `b4fcc3f0f2217761ad217dff1755f3ab3f3a8d82` |
| Chat template sha256 | `7de1c58e…88dd3c4` | `70210738…b219d` |
| Temperature     | 0.0 (variance: 0.7) | same |
| Top-p           | 1.0 (variance: 0.95) | same |
| Max new tokens  | recognition 64, advice 320 | same |
| Seed            | 42 (variance: 1, 2, 3 on 30 posts) | same |
| Backend         | vLLM 0.34.0, torch 2.14.0+cu130, Kaggle 2× Tesla T4, TP 2 | same |
| Precision       | fp32 | fp32 |
| Prompt version  | recognition v4, P0 v5, P-Oracle v5 | same |
| Prompt files    | `emotion-recognition_v4`, `advice-p0_v5`, `advice-oracle_v5` | same |

Final run: Kaggle, 2026-11-28, code at commit `c8d1e9c` (the record says `-dirty` because the notebook replaced the clone's `02-llm/gold` folder with a symlink to `/kaggle/working/gold`; no code was changed). Engine: `max_model_len` 8192, `gpu_memory_utilization` 0.95, `max_num_seqs` 32, `max_num_batched_tokens` 1024, CUDA graphs for decode, 16-token fp32 attention tiles. Gemma ran in session 1 and SEA-LION in session 2 (resumed from session 1's output). Outputs: recognition and P0 on 236 posts, variance on 30 posts, P-Oracle grid of 6,816 replies per model (71 candidates × 96).
