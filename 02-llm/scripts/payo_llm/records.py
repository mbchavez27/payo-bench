"""Output files: JSONL records with resume support, the gold manifest, and the run record."""

import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from payo_llm import ROOT

KEY_FIELDS = ("post_id", "model", "task", "condition", "seed")


def record_key(record: dict[str, Any]) -> tuple:
    """The identity of one generation; a rerun skips keys already on disk."""
    return tuple(record[field] for field in KEY_FIELDS)


def done_keys(path: Path) -> set[tuple]:
    """
    Read the keys already written to a JSONL file.

    Records with an `error` are not counted as done, so they are retried on the next run.

    Args:
        path: JSONL output file (may not exist yet).

    Returns:
        Keys of successful records.
    """
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not record.get("error"):
                    keys.add(record_key(record))
    return keys


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records, one JSON object per line, and flush so a crash loses at most one batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Current commit hash, with "-dirty" if there are uncommitted changes."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(out_dir: Path, inputs: dict[str, str]) -> Path:
    """
    Write MANIFEST.json for every JSONL under out_dir, in the silver manifest's format.

    Args:
        out_dir: Output folder of the run (02-llm/gold for the full profile).
        inputs: Input files (repo-relative path -> sha256) the outputs depend on.

    Returns:
        Path of the manifest.
    """
    files = {}
    for path in sorted(out_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        files[str(path.relative_to(out_dir))] = {
            "sha256": sha256_file(path), "rows": rows, "bytes": path.stat().st_size}
    manifest = {
        "layer": "gold",
        "generated_at": now_iso(),
        "generated_by": "02-llm/scripts/run.py",
        "git_commit": git_commit(),
        "inputs": inputs,
        "files": files,
    }
    target = out_dir / "MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def update_run_record(path: Path, model: str, entry: dict[str, Any]) -> None:
    """
    Store one model's run settings in run_record.json (one entry per model, latest run wins).

    Args:
        path: run_record.json in the run's output folder.
        model: "gemma" or "sealion".
        entry: Settings to record (backend, model_ref, precision, decoding, prompt files, versions).
    """
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record[model] = {**entry, "git_commit": git_commit(), "updated_at": now_iso()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
