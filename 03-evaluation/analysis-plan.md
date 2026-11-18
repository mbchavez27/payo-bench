# Analysis plan

**Version:** v1-draft · **Freeze before any model output is viewed.** After freezing, any change is logged in §11 with its reason, and analyses not listed here are reported as exploratory.

This file holds every **statistical** decision in the study. LLMs run only the two study tasks (`02-llm/prompts/`: emotion recognition and advice generation). Pre-screening, annotation, rating and strategy coding are done by the **authors** (`02-llm/codebook/`). Every number in the paper is computed here, in code, from frozen gold files.

Study framing: diagnostic and exploratory (modest N). Effect sizes with 95% CIs are the main result. p-values are secondary and Holm-corrected within each family (§10).

---

## 1. Units and inputs

| Symbol | Meaning |
| --- | --- |
| post *i* | one annotated, non-excluded post (target ≈ 300) |
| rated set | `N_eval` posts drawn from the annotated set for author rating (default **80**; §1.1) |
| model *m* | `gemma` (google/gemma-3-4b-it) or `sealion` (aisingapore/Gemma-SEA-LION-v4-4B-VL) |
| source *s* | `community`, `gemma`, `sealion` |
| condition *c* | `P0`, `P-Oracle` (LLM replies only) |
| rater *r* | an author rating replies with `02-llm/codebook/rating-rubric.md` |
| `emotion_type` | `culture_bound` / `shared` / `none`, from the gold label via the schema (never annotated directly) |
| `expression` | `explicit` / `implicit` (gold) |
| `correct` | Task A emotion matches gold; `invalid` outputs count as incorrect |

Gold labels are the **adjudicated** annotations (codebook §8). Posts labeled `none_practical` are kept in the overall RQ1 metrics but **left out** of the culture-bound vs. shared contrasts (H1a, H3a, H4b), since they belong to neither group.

### 1.1 Rated set

- **Size:** `N_eval` posts, stratified by `emotion_type` × `domain`. Each post has 5 replies: community, Gemma-P0, SEA-LION-P0, Gemma-Oracle and SEA-LION-Oracle, so 5 × `N_eval` items (400 at the default).
- **Double rating:** about 25% of items are rated by a second author, for §9 R3. Each item's primary rating is from one author; in a double-rated item, both ratings enter the models, with rater as a random effect.
- **Workload:** `N_eval` is the main lever if time is short. Below about 60 posts, the H4 analyses (§6) should be reported as descriptive only.
- **Drawn in code:** `00-sampling.ipynb` selects the rated set, the double-rated items and the blinded `item_id` key, with `SEED = 42`.

## 2. Before analysis: sampling and label-set finalization

Done after annotation and **before** looking at any model output.

1. **Merge rule.** Any emotion label with fewer than 15 adjudicated posts is merged into its nearest label. Proposed merges: `pikon` → `anger`; `tampo` → `sama_ng_loob` (as "relational hurt"); `pakikisama_strain` → `hiya` or `burnout` (decided at adjudication); `fear` → `worry`. A merge keeps the target label's `emotion_type`. Report the counts before and after merging.
2. **Agreement gate.** A label whose one-vs-rest κ is below 0.40 on the double-annotated subset (§9 R1) is revised and re-adjudicated, or merged. Report κ before and after.
3. **Pre-screen selection bias.** The manual pre-screen (codebook §0) is used only for sampling. Report:
   - the share of annotated posts with `cultural_concept ≠ none`, split by pre-screen `concept_likely` = `yes` / `no`;
   - the pre-screen's sensitivity and specificity against the gold concept label;
   - agreement between `domain_guess` and the gold domain.

## 3. RQ1: emotion recognition (H1a, H1b, H3a)

**Data:** 600 Task A outputs (≈300 posts × 2 models). Ratings are not needed.

**Descriptives, per model:**
- accuracy, macro-F1, and per-class precision, recall and F1 for the emotion label;
- the same metrics for the cultural-concept label, as a secondary measure;
- the invalid-output rate;
- confusion matrices;
- a majority-class baseline (always predict the most frequent gold label).

**CIs:** percentile bootstrap that resamples **posts** (B = 2,000, `SEED = 42`), applied to accuracy and macro-F1, overall and by `emotion_type` and `expression`.

**Model (primary):** mixed-effects logistic regression

```
correct ~ emotion_type * model + expression + (1 | post)
```

| Hypothesis | Estimate | Direction |
| --- | --- | --- |
| H1a culture-bound < shared | OR of `emotion_type` (culture_bound vs shared), averaged over models | OR < 1 |
| H1b implicit < explicit | OR of `expression` (implicit vs explicit) | OR < 1 |
| H3a SEA-LION > Gemma on culture-bound | `sealion − gemma` contrast within `culture_bound` (from the interaction) | OR > 1 |

**Paired check:** McNemar test of Gemma vs. SEA-LION correctness on the same posts. It is run overall and on the culture-bound subset (H3a), with the effect reported as the discordant-pair OR.

**Calibration (descriptive):** a reliability diagram and expected calibration error (ECE) of `confidence / 100`, per model.

## 4. RQ2: advice quality vs. community (H2a, H2b)

### 4.1 Ratings (H2a)

**Data:** author ratings on the rated set, P0 replies plus community replies (3 × `N_eval` items). One ordinal mixed model per dimension *d* ∈ {empathy, cultural_fit, helpfulness}:

```
rating_d ~ source + (1 | post) + (1 | rater)
```

**Contrasts:** `community − gemma` and `community − sealion`.

| Hypothesis | Test |
| --- | --- |
| Community higher on cultural fit and empathy | Directional contrast, OR > 1 for community |
| LLM comparable or higher on helpfulness | **Non-inferiority** of each LLM vs. community: the lower 95% CI bound of the LLM − community mean difference is above −Δ (Δ in §8) |

**Sensitivity:** add `reply_length_words` (centered) as a covariate. Report whether any source effect changes direction or loses its CI. For community replies only, also report the correlation of `reply_score` (upvotes) with ratings, as the upvote confound.

**Best overall fit:** each rater picks one of the three P0-level replies per post. Report win rates per source, with Wilson CIs.

### 4.2 Strategy drift (H2b)

**Data:** the authors' strategy codes on the rated set, P0 LLM replies plus community.

**Drift** for model *m*:

```
drift_m = P(autonomy | m) − P(autonomy | community)
```

- `neutral_practical` replies stay in the denominator. As a sensitivity check, drift is recomputed with them excluded.
- **CI:** paired bootstrap over posts, since each post contributes a community reply and an LLM reply.
- **Test:** McNemar on per-post autonomy (LLM vs. community), per model. H2b predicts drift > 0.
- **By domain and by cultural concept:** report drift with bootstrap CIs, **descriptive only**, with no per-subgroup tests (to avoid multiplicity). Cells with fewer than 10 posts are flagged.

## 5. RQ3: model comparison on advice (H3b)

The same rating models as §4.1, P0 replies, with the `sealion − gemma` contrast.

| Hypothesis | Test |
| --- | --- |
| SEA-LION higher on cultural fit | Directional contrast, OR > 1 |
| SEA-LION higher on register and tone | Directional contrast, OR > 1, from `rating_register_tone ~ source + (1 | post) + (1 | rater)` |
| No advantage in safe escalation | **Equivalence (TOST)** on the rate of `missing` escalation among posts where raters marked escalation as needed, with margin ±Δ_esc (§8). If fewer than 20 such posts exist, report descriptively only and say so. |

Harm-flag rates per source are reported descriptively, with Wilson CIs.

## 6. Linking question (H4a, H4b)

### 6.1 Recognition → advice (H4a, observational)

P0 ratings on the rated set, with `correct` taken from the full Task A run, per dimension:

```
rating_d ~ correct * emotion_type + model + domain + expression + post_length_words + (1 | post) + (1 | rater)
```

H4a predicts a positive `correct` effect.

**Within-post contrast:** take the rated posts where exactly one model is correct. Compare the correct model's mean rating with the incorrect model's on the same post using a **Wilcoxon signed-rank** test, with rank-biserial r as the effect size. This holds the post fixed, but the models differ in more than correctness, so it is reported as an **association**. If fewer than 15 such posts exist, report descriptively only.

### 6.2 Oracle test (H4b, interventional)

Per rated post and model, Δ = mean rating(P-Oracle) − mean rating(P0).

- **Paired test:** Wilcoxon signed-rank on Δ, per model and per `emotion_type`, with rank-biserial r.
- **Model:** `rating_d ~ condition * emotion_type * model + (1 | post) + (1 | rater)`. H4b predicts the `condition × emotion_type` interaction to be positive for `culture_bound`.

**Interpretation rule (pre-specified):**

| Pattern | Reading |
| --- | --- |
| Mean Δ ≥ Δ_oracle, CI excludes 0 | Recognition is the bottleneck |
| CI of mean Δ lies within ±Δ_oracle | Response generation is the bottleneck |
| Δ larger for culture-bound than shared (interaction CI excludes 0) | Cultural concepts are the weak point |
| Anything else | Inconclusive; report as such |

Prompt-length confound: P-Oracle prompts are longer. Report the mean prompt tokens for each condition.

## 7. If the reader survey runs (optional)

The survey may be dropped for time. If it runs, on an 80-post subset of the rated set with P0 replies plus community:
- Fit `rating_d ~ source + (1 | post) + (1 | respondent)` for empathy, cultural fit and helpfulness, with the same contrasts as §4.1. Report them next to the author results.
- **Author–reader agreement:** per reply, the Spearman ρ between the author mean and the reader mean per dimension (bootstrap CI). Also agreement on best-fit choices (% agreement and Cohen's κ).
- Survey reliability: Krippendorff's α (ordinal) per dimension.

If it does not run, report this as a construct limitation: the ratings come from the authors, not from lay Filipino readers.

## 8. Pre-registered margins (fill in before freezing)

| Symbol | Used in | Proposed value | Rationale |
| --- | --- | --- | --- |
| Δ | H2a non-inferiority (helpfulness) | 0.5 points on the 1–5 scale | half a scale step |
| Δ_esc | H3b safe-escalation equivalence | 10 percentage points | |
| Δ_oracle | H4b "substantial" improvement | 0.5 points on the 1–5 scale | same as Δ |

## 9. Reliability

| Id | What | Statistic |
| --- | --- | --- |
| R1 | Annotation (≥ 25% double-annotated) | Cohen's κ per field (`primary_emotion`, `cultural_concept`, `expression`), plus one-vs-rest κ per emotion label; the gate is in §2, step 2 |
| R2 | Strategy coding (double-coded items) | Cohen's κ between the two authors |
| R3 | Author ratings (≈ 25% double-rated) | Krippendorff's α (ordinal) per dimension; percent agreement on safe escalation and best fit |
| R4 | Generation variance (sampled runs on a subset) | Agreement of Task A labels across samples, and the share of advice replies whose strategy code changes; compare with the size of the effects above |

If R3 α is below 0.67 for a dimension, results on that dimension are reported as tentative.

## 10. Multiple comparisons

Holm correction is applied **within each family**. Descriptive, subgroup and sensitivity analyses are not corrected and are labeled as such.

| Family | Tests |
| --- | --- |
| F1 Recognition | H1a, H1b, H3a (model contrast + McNemar) |
| F2 Advice vs. community | H2a (3 dimensions × 2 contrasts), H2b (2 models) |
| F3 Model comparison | H3b (cultural fit, register, escalation TOST) |
| F4 Linking | H4a (3 dimensions), H4b (3 dimensions) |
| F5 Survey (only if it runs) | §7 contrasts |

## 11. Changelog

| Version | Date | Change |
| --- | --- | --- |
| v1-draft | 2026-11-18 | First draft. LLMs run only Task A and Task B; ratings, strategy codes and pre-screen are by the authors; the LLM judge is removed; the reader survey is optional. |

## Threats specific to author rating

- **Raters know the label set and set the gold emotions.** Mitigations: blinded `item_id`, shuffled order, P0 and Oracle mixed together, raters skip posts they annotated where possible, and R3 α is reported.
- **Authors are not lay readers.** This is the reason for keeping §7 if time allows.
- **The rated set is smaller than the full run** (`N_eval` vs. ≈300), so H2–H4 have less power than RQ1. Report this alongside the effect sizes.

## Open decision: tooling for ordinal mixed models

Python has no mature ordinal model with random effects (`statsmodels` `OrderedModel` has none). Options:
- **R `ordinal::clmm`**, called from a small R script (the standard tool; recommended);
- **`bambi`** (Bayesian, cumulative family with random effects; adds PyMC);
- treating ratings as continuous in `statsmodels` `MixedLM`, as a sensitivity check only.

Decide this before writing the notebooks. The logistic models (§3, §4.2) can use `statsmodels` or `pymer4`.
