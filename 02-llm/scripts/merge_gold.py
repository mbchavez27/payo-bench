"""
Merge a second machine's gold/ folder into 02-llm/gold/, when the GPU work was split by model
(for example Gemma on one Kaggle account and SEA-LION on another, each with `pipeline.py --model ...`).

    uv run python 02-llm/scripts/merge_gold.py ~/Downloads/gold-sealion

- Output files are per model, so they are copied in. A file that exists on both sides with different
  content stops the merge (nothing is overwritten).
- run_record.json is keyed by model; the other side's models are added. The same model on both sides stops it.
- Both sides must have the same precision and engine settings, so the study has one setup.
- MANIFEST.json is rewritten over the merged JSONL files.

It never opens the outputs' contents beyond hashing them (blindness rule).
"""

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from payo_llm import ROOT
from payo_llm.records import write_manifest

GOLD = ROOT / "02-llm" / "gold"
RECORD = "run_record.json"
SETUP_KEYS = ("precision", "engine")  # must match across machines


def model_setups(record: dict[str, Any]) -> dict[str, str]:
    """One JSON string per model and run of the settings that must agree across machines."""
    return {f"{model} {run}": json.dumps({k: entry.get(k) for k in SETUP_KEYS}, sort_keys=True)
            for model, runs in record.items() for run, entry in runs.items()}


def check_compatible(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    """Problems that stop the merge (empty list = fine)."""
    problems = [f"{m} is in both run records" for m in sorted(set(ours) & set(theirs))]
    setups = {**model_setups(ours), **model_setups(theirs)}
    if len(set(setups.values())) > 1:
        problems.append("precision/engine settings differ between models: " + "; ".join(
            f"{m}: {s}" for m, s in sorted(setups.items())))
    return problems


def copy_outputs(src: Path, dst: Path) -> tuple[int, list[str]]:
    """Copy every file except run_record.json and MANIFEST.json. Returns (copied, conflicts)."""
    copied, conflicts = 0, []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        if rel.name in (RECORD, "MANIFEST.json"):
            continue
        target = dst / rel
        if target.exists():
            if not filecmp.cmp(path, target, shallow=False):
                conflicts.append(str(rel))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("other", type=Path, help="the other machine's gold/ folder")
    args = parser.parse_args()
    other = args.other.expanduser().resolve()
    if not (other / RECORD).exists():
        sys.exit(f"{other} has no {RECORD}: is it a gold/ folder from pipeline.py?")

    ours = json.loads((GOLD / RECORD).read_text(encoding="utf-8")) if (GOLD / RECORD).exists() else {}
    theirs = json.loads((other / RECORD).read_text(encoding="utf-8"))
    problems = check_compatible(ours, theirs)
    conflicting = [str(p.relative_to(other)) for p in other.rglob("*.jsonl")
                   if (GOLD / p.relative_to(other)).exists()
                   and not filecmp.cmp(p, GOLD / p.relative_to(other), shallow=False)]
    problems += [f"{rel} exists on both sides with different content" for rel in conflicting]
    if problems:
        sys.exit("Not merged:\n  " + "\n  ".join(problems))

    copied, conflicts = copy_outputs(other, GOLD)
    if conflicts:  # non-JSONL files such as chat templates
        sys.exit("Copied the outputs, but these differ on both sides; check them:\n  " + "\n  ".join(conflicts))
    (GOLD / RECORD).write_text(json.dumps({**ours, **theirs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sys.path.insert(0, str(ROOT / "02-llm" / "scripts"))
    import run  # noqa: PLC0415  (manifest inputs, same as a normal full run)
    import tomllib  # noqa: PLC0415

    cfg = tomllib.loads(run.CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = write_manifest(GOLD, run.manifest_inputs(argparse.Namespace(profile="full", posts=None), cfg))
    print(f"Merged {copied} files from {other}; models now: {sorted({**ours, **theirs})}. "
          f"Manifest: {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
