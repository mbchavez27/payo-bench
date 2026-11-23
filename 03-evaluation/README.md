# 03-evaluation: gold → results

All statistics are computed here, in code, from the frozen files in `02-llm/gold/`. LLMs run only the two study tasks. The authors pre-screen, annotate, rate and code strategies. No test result comes from an LLM. The order of work is in [`../RUNBOOK.md`](../RUNBOOK.md).

- [`analysis-plan.md`](analysis-plan.md): every metric, test and model, mapped to RQ1–RQ3 and H1a–H4b (v1.2).
- [`config.toml`](config.toml): roster, sample sizes, thresholds, merge map, margins.
- `workflow.py`: the human-in-the-loop steps (sheets out, sheets in) and `analyze` / `smoke`.
- `payo_eval/`: all logic, tested (`tests/`).

## `payo_eval` modules

| Module | Does |
| --- | --- |
| `paths` | every file location (overridable with `PAYO_*` variables), config loading |
| `sheets`, `instruments` | Excel sheets with dropdowns; column rules from the schema and rubric; strict read-back |
| `sampling` | pre-screen pool, stratified sample, balanced assignment with overlap, rated set |
| `annotation`, `report` | collect, disagreements, adjudication, merges, κ gate, finalization report |
| `rating` | blinded items and key, community-reply cleaning, rater sheets, rating collection |
| `load` | analysis-ready tables from gold (merges applied to gold and predictions) |
| `metrics`, `models`, `agreement` | accuracy/F1/bootstrap/ECE; GEE, MixedLM, contrasts, McNemar, Wilcoxon, TOST, Holm; κ, α |
| `analyses` | one function per plan section, used by the notebooks |
| `steps`, `synthetic` | the workflow steps; a synthetic study for `smoke` and the tests |

## Notebooks (one evaluation each)

| Notebook | Covers | Plan § |
| --- | --- | --- |
| `01-label-finalization.ipynb` | counts and merges, R1 κ, pre-screen selection bias | §2, §9 R1 |
| `02-emotion-recognition.ipynb` | RQ1, H1a, H1b, H3a, concept metrics, calibration | §3 |
| `03-advice-quality.ipynb` | H2a (with length and upvote checks), best fit, H3b, harm flags | §4.1, §5 |
| `04-strategy-drift.ipynb` | H2b, drift by domain and concept | §4.2 |
| `05-linking.ipynb` | H4a, H4b and the interpretation rule, prompt tokens | §6 |
| `06-reliability.ipynb` | R2, R3, R4 | §9 |
| `07-multiple-comparisons.ipynb` | Holm within F1–F4 (run last) | §10 |

All use `SEED = 42` from `config.toml`. Outputs go to `results/` and `figures/` (gitignored). `workflow.py analyze` runs them in order.
