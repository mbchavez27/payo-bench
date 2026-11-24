"""
P-Oracle from the pre-generated grid (analysis plan v1.4).

The GPU writes one P-Oracle reply per candidate post for every emotion x concept pair before anyone
annotates (`run.py --task advice-oracle-grid`). Once the labels are frozen, this module picks, for each post,
the reply whose prompt carried that post's gold emotion and concept, and writes it where `run.py
--task advice-oracle` would have written it. Greedy decoding makes the two the same reply.
"""

import json
from pathlib import Path

from payo_llm.data import load_gold
from payo_llm.schema import Schema

MODELS = ("gemma", "sealion")


def grid_condition(emotion: str, concept: str) -> str:
    """Must match run.py's grid_condition."""
    return f"P-Oracle:{emotion}|{concept}"


def _read_grid(path: Path) -> dict[tuple[str, str], dict]:
    """(post_id, condition) -> the last successful record (reruns append retries)."""
    rows: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not record.get("error"):
                    rows[(record["post_id"], record["condition"])] = record
    return rows


def select_oracle(grid_dir: Path, advice_dir: Path, adjudicated: Path, candidates: set[str],
                  schema: Schema) -> dict[str, int]:
    """
    Write replies_<model>_oracle.jsonl from the grid, using each candidate's gold label.

    Args:
        grid_dir: advice-generation/oracle_grid (replies_<model>.jsonl from run.py).
        advice_dir: advice-generation, where replies_<model>_oracle.jsonl is written.
        adjudicated: annotations_adjudicated.csv (original, unmerged labels: the P-Oracle input, v1.2).
        candidates: Rated-set candidate post ids.
        schema: The label schema (validates the gold ids, as run.py does).

    Returns:
        Replies written per model.

    Raises:
        SystemExit: A grid file is missing, or a needed label pair has no successful reply.
    """
    gold = {pid: label for pid, label in load_gold(adjudicated, schema).items() if pid in candidates}
    written = {}
    for model in MODELS:
        path = grid_dir / f"replies_{model}.jsonl"
        if not path.exists():
            raise SystemExit(f"No P-Oracle grid for {model} at {path}: run pipeline.py on the GPU PC first.")
        grid = _read_grid(path)
        out, missing = [], []
        for pid in sorted(gold):
            label = gold[pid]
            record = grid.get((pid, grid_condition(label.emotion, label.cultural_concept)))
            if record is None:
                missing.append(f"{pid} ({label.emotion}, {label.cultural_concept})")
                continue
            out.append({**record, "task": "advice-oracle", "condition": "P-Oracle",
                        "oracle_source": "pregenerated_grid"})
        if missing:
            raise SystemExit(f"{model}: the grid has no successful reply for {len(missing)} gold label(s): "
                             + ", ".join(missing[:10]) + ". Rerun the grid on the GPU PC (it resumes).")
        advice_dir.mkdir(parents=True, exist_ok=True)
        with (advice_dir / f"replies_{model}_oracle.jsonl").open("w", encoding="utf-8") as handle:
            for record in out:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written[model] = len(out)
    return written
