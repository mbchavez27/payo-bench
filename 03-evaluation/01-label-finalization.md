---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 01 · Label finalization and pre-screen selection bias

Analysis plan §2 and §9 R1. Recomputes the finalization from the frozen gold files (the same code as `workflow.py finalize-labels`), so the paper's counts and κ come from one place.

```python
# Shared setup: every path comes from payo_eval.paths, so PAYO_* variables can point the notebook at synthetic data
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from payo_eval import analyses as A
from payo_eval import load
from payo_eval.paths import get_paths, load_config

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
```

```python
# Settings from 03-evaluation/config.toml; outputs go to results/01-label-finalization/ and figures/
PATHS = get_paths()
CFG = load_config(PATHS)
SEED = CFG["seed"]
B = CFG["analysis"]["bootstrap_b"]
RESULTS = PATHS.results / "01-label-finalization"
TESTS = PATHS.results / "tests"
FIGURES = PATHS.figures
for folder in (RESULTS, TESTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)


def save(table: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Write a result table as CSV and return it for display."""
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    return table
```

```python
# Load the gold annotation files and rerun finalization (deterministic, no new decisions)
from payo_llm.schema import load_schema
from payo_eval.annotation import finalize

schema = load_schema()
raw = pd.read_csv(PATHS.annotations_raw, dtype=str).fillna("")
decisions = pd.read_csv(PATHS.adjudication, dtype=str).fillna("") if PATHS.adjudication.exists() else None
result = finalize(raw, decisions, schema, CFG.get("merges", {}), CFG["labels"]["min_label"], CFG["labels"]["kappa_gate"])
print("Merges:", result.merges or "none")
```

## Counts before and after merging, and exclusions

```python
save(result.counts, "emotion_counts")
```

```python
save(result.exclusions, "exclusions")
```

## R1: agreement on the double-annotated posts

Cohen's κ pools first vs. second annotator across pairs. One-vs-rest κ is the §2 gate (κ < gate means revise, re-adjudicate or merge).

```python
save(result.field_kappa, "r1_field_kappa")
```

```python
save(result.label_kappa, "r1_label_kappa")
```

```python
print("Gate failures:", result.gate_failures or "none")
```

## Pre-screen selection bias (§2 step 3)

Computed on the sampled posts only (only they have gold labels).

```python
# Compare the screener's quick guesses with the gold labels of the same posts
prescreen = pd.read_csv(PATHS.prescreen, dtype=str).fillna("")
bias = A.prescreen_bias(prescreen, load.labels(PATHS))
save(bias["concept_share"], "prescreen_concept_share")
```

```python
save(bias["accuracy"], "prescreen_accuracy")
```

## Findings

_Fill in after the real run: the main estimates with their CIs, and anything flagged above._
