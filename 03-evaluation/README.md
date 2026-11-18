# 03-evaluation — Gold → results

All statistics are computed here, in code, from the frozen files in `02-llm/gold/`. LLMs run only the two study tasks (emotion recognition and advice generation). The authors pre-screen, annotate, rate and code strategies. No test result comes from an LLM.

- [`analysis-plan.md`](analysis-plan.md): every metric, test and model, mapped to RQ1–RQ3 and H1a–H4b. Freeze it before viewing model output.

## Planned notebooks (one evaluation per notebook)

| Notebook | Covers | Plan § |
| --- | --- | --- |
| `00-sampling.ipynb` | stratified ~300 posts from the pre-screen, 25% double-annotation subset, `N_eval` rated set, double-rated items, blinded `item_id` key | §1.1, §2 |
| `01-label-finalization.ipynb` | merges, κ gate, pre-screen selection bias | §2, §9 R1 |
| `02-emotion-recognition.ipynb` | RQ1, H1a, H1b, H3a | §3 |
| `03-advice-quality.ipynb` | H2a, H3b, best fit | §4.1, §5 |
| `04-strategy-drift.ipynb` | H2b | §4.2, §9 R2 |
| `05-linking.ipynb` | H4a, H4b | §6 |
| `06-reliability.ipynb` | rating α, generation variance (and the survey, if it runs) | §7, §9 R3–R4 |

All notebooks use `SEED = 42`. Outputs go to `03-evaluation/results/` (tables) and `03-evaluation/figures/`.
