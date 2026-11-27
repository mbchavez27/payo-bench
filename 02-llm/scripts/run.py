"""
Run the two LLM tasks: emotion recognition (Task A) and advice generation (Task B, P0 and P-Oracle).

Examples:
    uv run python 02-llm/scripts/run.py --profile pilot --check
    uv run python 02-llm/scripts/run.py --profile pilot --task all --model both
    uv run python 02-llm/scripts/run.py --profile full --task recognition,advice-p0 --model gemma
    uv run python 02-llm/scripts/run.py --profile full --task all --model both --variance

See 02-llm/scripts/README.md for setup.
"""

import argparse
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from payo_llm import ROOT
from payo_llm.backends import (Backend, Decoding, Generation, OllamaBackend, VLLMBackend, check_gpu, cuda_info,
                               cuda_vram_gb, select_precision)
from payo_llm.data import GoldLabel, Post, load_gold, load_posts, read_post_ids
from payo_llm.parsing import parse_recognition
from payo_llm.prompts import PromptTemplate, format_post, load_prompt, render, sha256_text
from payo_llm.records import append_jsonl, done_keys, existing_precisions, now_iso, sha256_file, update_run_record, write_manifest
from payo_llm.schema import SCHEMA_PATH, Schema, load_schema

CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
PROMPT_DIR = ROOT / "02-llm" / "prompts"
TASKS = ("recognition", "advice-p0", "advice-oracle", "advice-oracle-grid")
DEFAULT_TASKS = ("recognition", "advice-p0", "advice-oracle")  # what --task all means
PROMPT_FILES = {
    "recognition": "emotion-recognition_v4.md",
    "advice-p0": "advice-p0_v5.md",
    "advice-oracle": "advice-oracle_v5.md",
    "advice-oracle-grid": "advice-oracle_v5.md",  # the same prompt, for every emotion x concept (v1.4)
}
CONDITIONS = {"recognition": "none", "advice-p0": "P0", "advice-oracle": "P-Oracle"}


def grid_condition(emotion: str, concept: str) -> str:
    """Condition of one pre-generated P-Oracle reply: unique per label pair, so resume keys stay unique."""
    return f"P-Oracle:{emotion}|{concept}"


@dataclass(frozen=True)
class Job:
    """One prompt to send: a post under one task and condition."""

    post: Post
    task: str
    prompt: str
    template: PromptTemplate
    condition: str = ""
    oracle_label: tuple[str, str] | None = None  # (emotion, concept) given in a grid prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, choices=["pilot", "full"])
    parser.add_argument("--task", default="all", help="comma-separated: recognition,advice-p0,advice-oracle, or all")
    parser.add_argument("--model", default="both", choices=["gemma", "sealion", "both"])
    parser.add_argument("--posts", type=Path, help="CSV with a post_id column (overrides the profile)")
    parser.add_argument("--limit", type=int, help="only the first N posts (smoke tests)")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp8", "bnb4"], help="full profile only")
    parser.add_argument("--gpu-mem", type=float, help="full profile: override gpu_memory_utilization (0-1)")
    parser.add_argument("--out-dir", type=Path, help="write outputs here instead of the profile's folder (trial runs)")
    parser.add_argument("--variance", action="store_true", help="sampled runs on a subset, for variance")
    parser.add_argument("--check", action="store_true", help="preflight only: no generation")
    args = parser.parse_args()
    args.tasks = list(DEFAULT_TASKS) if args.task == "all" else [t.strip() for t in args.task.split(",")]
    unknown = set(args.tasks) - set(TASKS)
    if unknown:
        parser.error(f"unknown task(s): {sorted(unknown)}")
    if args.variance and "advice-oracle-grid" in args.tasks:
        parser.error("advice-oracle-grid has no variance run (analysis plan v1.4)")
    return args


def child_argv(args: argparse.Namespace, model: str) -> list[str]:
    """Rebuild the command line for one model, so each model runs in a fresh process."""
    argv = [sys.executable, str(Path(__file__).resolve()), "--profile", args.profile,
            "--task", ",".join(args.tasks), "--model", model]
    if args.posts:
        argv += ["--posts", str(args.posts)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.precision:
        argv += ["--precision", args.precision]
    if args.gpu_mem:
        argv += ["--gpu-mem", str(args.gpu_mem)]
    if args.out_dir:
        argv += ["--out-dir", str(args.out_dir)]
    if args.variance:
        argv.append("--variance")
    if args.check:
        argv.append("--check")
    return argv


def resolve_precision(profile: dict[str, Any], override: str | None) -> str:
    """Pick the vLLM precision once; the same value is used for both models."""
    requested = override or profile["precision"]
    vram = cuda_vram_gb() if requested == "auto" else None
    precision = select_precision(requested, vram)
    if vram is not None:
        print(f"GPU VRAM {vram:.1f} GB -> precision {precision}")
    return precision


def make_backend(cfg: dict[str, Any], profile: dict[str, Any], model: str, precision: str | None) -> Backend:
    spec = cfg["models"][model]
    if profile["backend"] == "ollama":
        return OllamaBackend(spec["ollama_tag"], profile["host"], profile["num_ctx"])
    return VLLMBackend(spec["hf_id"], spec["hf_revision"], precision or "bf16", profile["max_model_len"],
                       profile["gpu_memory_utilization"], cfg["seed"], **engine_options(profile))


def engine_options(profile: dict[str, Any]) -> dict[str, Any]:
    """vLLM memory options from the profile (defaults are vLLM's own)."""
    return {"enforce_eager": bool(profile.get("enforce_eager", False)),
            "max_num_seqs": int(profile.get("max_num_seqs", 256))}


def build_jobs(task: str, posts: list[Post], schema: Schema, template: PromptTemplate,
               gold: dict[str, GoldLabel] | None) -> tuple[list[Job], int]:
    """
    Render the prompt for every post in a task.

    Returns:
        The jobs, and how many posts were skipped for lacking a gold label (P-Oracle only).
        advice-oracle-grid makes one job per post x emotion x concept, in post order (prefix-cache friendly).
    """
    shared = {
        "emotion_list": schema.render_list("emotions"),
        "concept_list": schema.render_list("cultural_concepts"),
        "reply_language": schema.reply_language,
    }
    jobs, skipped = [], 0
    for post in posts:
        values = {**shared, "post": format_post(post.problem_goal, post.context, post.previous_attempts)}
        if task == "advice-oracle-grid":
            for emotion_id in schema.emotion_ids:
                for concept_id in schema.concept_ids:
                    emotion, concept = schema.emotion(emotion_id), schema.concept(concept_id)
                    cell = values | {"emotion_name": emotion["name"], "emotion_definition": emotion["definition"],
                                     "concept_name": concept["name"], "concept_definition": concept["definition"]}
                    jobs.append(Job(post, task, render(template, cell), template,
                                    grid_condition(emotion_id, concept_id), (emotion_id, concept_id)))
            continue
        if task == "advice-oracle":
            label = gold.get(post.post_id) if gold else None
            if label is None:
                skipped += 1
                continue
            # Gold labels only: the model's own Task A output is never used here
            emotion, concept = schema.emotion(label.emotion), schema.concept(label.cultural_concept)
            values |= {"emotion_name": emotion["name"], "emotion_definition": emotion["definition"],
                       "concept_name": concept["name"], "concept_definition": concept["definition"]}
        jobs.append(Job(post, task, render(template, values), template, CONDITIONS[task]))
    return jobs, skipped


def output_path(out_dir: Path, task: str, model: str, seed: int | None) -> Path:
    if task == "recognition":
        name = f"emotion-recognition/predictions_{model}"
    elif task == "advice-oracle-grid":
        name = f"advice-generation/oracle_grid/replies_{model}"
    else:
        name = f"advice-generation/replies_{model}_{'p0' if task == 'advice-p0' else 'oracle'}"
    if seed is not None:
        return out_dir / "variance" / f"{name.replace('/', '_')}_seed{seed}.jsonl"
    return out_dir / f"{name}.jsonl"


def to_record(job: Job, gen: Generation, backend: Backend, model: str, decoding: Decoding,
              schema: Schema, latency: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "post_id": job.post.post_id,
        "model": model,
        "task": job.task,
        "condition": job.condition,
        "seed": decoding.seed,
        "backend": backend.name,
        "model_ref": backend.model_ref,
        "precision": backend.precision,
        "prompt_version": job.template.name,
        "prompt_sha256": job.template.sha256,
        "rendered_sha256": sha256_text(job.prompt),
        "decoding": decoding.as_dict(),
        "raw_output": gen.text,
        "n_prompt_tokens": gen.n_prompt_tokens,
        "n_output_tokens": gen.n_output_tokens,
        "finish_reason": gen.finish_reason,
        "truncated": gen.truncated,
        "latency_s": round(latency, 3),
        "error": gen.error,
        "created_at": now_iso(),
    }
    if job.task == "recognition":
        parsed = parse_recognition(gen.text, schema.emotion_ids, schema.concept_ids, schema.concept_aliases)
        record |= {"emotion": parsed.emotion, "cultural_concept": parsed.cultural_concept,
                   "confidence": parsed.confidence, "valid": parsed.valid,
                   "invalid_reason": parsed.invalid_reason, "concept_alias": parsed.concept_alias}
    else:
        record["reply_words"] = len(gen.text.split())
    if job.oracle_label:
        record["oracle_emotion"], record["oracle_concept"] = job.oracle_label
    return record


def run_jobs(jobs: list[Job], backend: Backend, model: str, decoding: Decoding, schema: Schema,
             path: Path, batch_size: int) -> dict[str, int]:
    """Generate in batches, appending after each batch so an interrupted run can resume."""
    mixed = existing_precisions(path) - {backend.precision}
    if mixed:
        raise SystemExit(f"{path} already has outputs at precision {sorted(mixed)}, but this run is "
                         f"{backend.precision}. Every study run must use one precision: rerun with the same "
                         "--precision, or move that file away to start over.")
    done = done_keys(path)
    pending = [j for j in jobs if (j.post.post_id, model, j.task, j.condition, decoding.seed) not in done]
    stats = {"skipped_done": len(jobs) - len(pending), "written": 0, "errors": 0, "invalid": 0, "truncated": 0}
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        began = time.perf_counter()
        gens = backend.generate([j.prompt for j in batch], decoding)
        per_item = (time.perf_counter() - began) / len(batch)
        records = [to_record(j, g, backend, model, decoding, schema, per_item) for j, g in zip(batch, gens, strict=True)]
        append_jsonl(path, records)
        stats["written"] += len(records)
        stats["errors"] += sum(1 for r in records if r["error"])
        stats["invalid"] += sum(1 for r in records if r.get("valid") is False)
        stats["truncated"] += sum(1 for r in records if r["truncated"])
        print(f"  {path.name}: {start + len(batch)}/{len(pending)}", flush=True)
    return stats


def preflight(cfg: dict[str, Any], profile: dict[str, Any], model: str, precision_override: str | None) -> None:
    """Check the backend can serve the model, without generating anything."""
    if profile["backend"] == "ollama":
        backend = OllamaBackend(cfg["models"][model]["ollama_tag"], profile["host"], profile["num_ctx"])
        print(f"[ok] {model}: {backend.model_ref} ({backend.precision}), {backend.versions()}")
        return
    precision = resolve_precision(profile, precision_override)
    capability, total, free, name = cuda_info()
    print(f"GPU: {name}, compute capability {capability[0]}.{capability[1]}, {total:.1f} GB total, {free:.1f} GB free")
    problems = check_gpu(capability, total, free, profile["gpu_memory_utilization"])
    if problems:
        sys.exit("GPU check failed:\n  " + "\n  ".join(problems))
    import vllm  # noqa: PLC0415
    from huggingface_hub import model_info  # noqa: PLC0415

    spec = cfg["models"][model]
    info = model_info(spec["hf_id"], revision=spec["hf_revision"])  # fails if gated and no access
    print(f"[ok] {model}: {spec['hf_id']}@{info.sha[:12]} reachable; vllm {vllm.__version__}; precision {precision}")


def run_model(args: argparse.Namespace, cfg: dict[str, Any], model: str) -> None:
    profile = dict(cfg["profiles"][args.profile])
    if args.gpu_mem:
        profile["gpu_memory_utilization"] = args.gpu_mem
    if args.check:
        preflight(cfg, profile, model, args.precision)
        return

    schema = load_schema()
    grid = args.tasks == ["advice-oracle-grid"]
    if "advice-oracle-grid" in args.tasks and not grid:
        sys.exit("Run advice-oracle-grid on its own: it uses a different post list (the rated-set candidates).")
    if grid and not (args.posts or profile.get("oracle_grid_posts")):
        sys.exit(f"The {args.profile} profile has no oracle_grid_posts; pass --posts.")
    post_ids = read_post_ids(ROOT / (args.posts or profile["oracle_grid_posts" if grid else "posts"]))
    if args.limit:
        post_ids = post_ids[:args.limit]
    variance = cfg["decoding"]["variance"]
    if args.variance:
        post_ids = post_ids[:variance["n_posts"]]
    posts = load_posts(post_ids)
    templates = {task: load_prompt(PROMPT_DIR / PROMPT_FILES[task]) for task in args.tasks}
    gold = load_gold(ROOT / profile["gold"], schema) if "advice-oracle" in args.tasks else None

    precision = resolve_precision(profile, args.precision) if profile["backend"] == "vllm" else None
    out_dir = args.out_dir.resolve() if args.out_dir else ROOT / profile["out_dir"]
    print(f"Loading {model} ({profile['backend']}) for {len(posts)} posts: {', '.join(args.tasks)}")
    backend = make_backend(cfg, profile, model, precision)
    try:
        seeds = variance["seeds"] if args.variance else [cfg["seed"]]
        main = cfg["decoding"]["variance" if args.variance else "main"]
        for task in args.tasks:
            jobs, skipped = build_jobs(task, posts, schema, templates[task], gold)
            if skipped:
                print(f"  {task}: {skipped} posts skipped (no gold label)")
            max_new = cfg["max_new_tokens"]["recognition" if task == "recognition" else "advice"]
            for seed in seeds:
                decoding = Decoding(main["temperature"], main["top_p"], max_new, seed)
                path = output_path(out_dir, task, model, seed if args.variance else None)
                stats = run_jobs(jobs, backend, model, decoding, schema, path, profile["batch_size"])
                print(f"  {task} seed={seed}: {stats}")

        template_text = backend.chat_template or ""
        template_path = out_dir / "chat_templates" / f"{model}.txt"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(template_text, encoding="utf-8")
        update_run_record(out_dir / "run_record.json", model, {
            "profile": args.profile,
            "backend": backend.name,
            "model_ref": backend.model_ref,
            "precision": backend.precision,
            "chat_template_sha256": sha256_text(template_text),
            "decoding_main": cfg["decoding"]["main"],
            "decoding_variance": cfg["decoding"]["variance"],
            "max_new_tokens": cfg["max_new_tokens"],
            "seed": cfg["seed"],
            "prompt_files": {t.name: {"version": t.version, "sha256": t.sha256} for t in templates.values()},
            "schema_version": schema.version,
            "posts_file": str(args.posts or profile["oracle_grid_posts" if grid else "posts"]),
            "n_posts": len(posts),
            "versions": backend.versions(),
            **({"engine": {"max_model_len": profile["max_model_len"],
                           "gpu_memory_utilization": profile["gpu_memory_utilization"],
                           **engine_options(profile)}} if profile["backend"] == "vllm" else {}),
        }, run=",".join(args.tasks) + ("+variance" if args.variance else ""))
    finally:
        backend.close()


def manifest_inputs(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, str]:
    profile = cfg["profiles"][args.profile]
    paths = [SCHEMA_PATH, ROOT / profile["posts"], ROOT / profile.get("oracle_grid_posts", profile["posts"]),
             ROOT / profile["gold"],
             *([ROOT / args.posts] if args.posts else []),
             ROOT / "01-preprocessing" / "silver" / "MANIFEST.json",
             *(PROMPT_DIR / name for name in PROMPT_FILES.values())]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths if p.exists()}


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")  # HF_TOKEN for the gated Gemma weights (full run)
    cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if args.model == "both":
        # One process per model: GPU/unified memory is fully released between models
        for model in ("gemma", "sealion"):
            result = subprocess.run(child_argv(args, model), check=False)
            if result.returncode != 0:
                sys.exit(f"{model} failed (exit {result.returncode}); rerun to resume")
    else:
        run_model(args, cfg, args.model)

    if args.profile == "full" and not args.check and not args.out_dir:
        path = write_manifest(ROOT / cfg["profiles"]["full"]["out_dir"], manifest_inputs(args, cfg))
        print(f"Manifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
