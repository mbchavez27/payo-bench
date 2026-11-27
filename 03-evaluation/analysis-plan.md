# Analysis plan

**Version:** v1 · **Frozen 2026-11-18, before any model output was viewed.** After freezing, any change is logged in §11 with its reason, and analyses not listed here are reported as exploratory.

This file holds every **statistical** decision in the study. LLMs run only the two study tasks (`02-llm/prompts/`: emotion recognition and advice generation). Pre-screening, annotation, rating and strategy coding are done by the **authors** (`02-llm/codebook/`). Every number in the paper is computed here, in code, from frozen gold files.

Study framing: diagnostic and exploratory (modest N). Effect sizes with 95% CIs are the main result. p-values are secondary and Holm-corrected within each family (§10).

---

## 1. Units and inputs

| Symbol | Meaning |
| --- | --- |
| post *i* | one annotated, non-excluded post (target ≈ 200 since v1.3; was 300) |
| rated set | `N_eval` = **60** posts drawn from the annotated set for author rating (80 until v1.3; §1.1) |
| model *m* | `gemma` (google/gemma-3-4b-it) or `sealion` (aisingapore/Gemma-SEA-LION-v4-4B-VL) |
| source *s* | `community`, `gemma`, `sealion` |
| condition *c* | `P0`, `P-Oracle` (LLM replies only) |
| rater *r* | an author rating replies with `02-llm/codebook/rating-rubric.md` |
| `emotion_type` | `culture_bound` / `shared` / `none`, from the gold label via the schema (never annotated directly) |
| `expression` | `explicit` / `implicit` (gold) |
| `correct` | Task A emotion matches gold; `invalid` outputs count as incorrect |

Gold labels are the **adjudicated** annotations (codebook §8). Posts labeled `none_practical` are kept in the overall RQ1 metrics but **left out** of the culture-bound vs. shared contrasts (H1a, H3a, H4b), since they belong to neither group.

### 1.1 Rated set

- **Size:** `N_eval` posts, stratified by `emotion_type` × `domain`. Each post has 5 replies: community, Gemma-P0, SEA-LION-P0, Gemma-Oracle and SEA-LION-Oracle, so 5 × `N_eval` items (300 at the default).
- **Double rating:** about 25% of items are rated by a second author, for §9 R3. Each item's primary rating is from one author; in a double-rated item, both ratings enter the models, with rater as a random effect.
- **Workload:** `N_eval` is the main lever if time is short. Below about 60 posts, the H4 analyses (§6) should be reported as descriptive only.
- **Drawn in code:** `00-sampling.ipynb` selects the rated set, the double-rated items and the blinded `item_id` key, with `SEED = 42`.

## 2. Before analysis: sampling and label-set finalization

Done after annotation and **before** looking at any model output.

1. **Merge rule.** Any emotion label with fewer than 15 adjudicated posts is merged into its nearest label. Proposed merges: `pikon` → `anger`; `tampo` → `sama_ng_loob` (as "relational hurt"); `fear` → `worry`. A merge keeps the target label's `emotion_type`. Report the counts before and after merging.
2. **Agreement gate.** A label whose one-vs-rest κ is below 0.40 on the double-annotated subset (§9 R1) is revised and re-adjudicated, or merged. Report κ before and after.
3. **Pre-screen selection bias.** The pre-screen (codebook §0) is used only for sampling. Since v1.3 it is the fixed keyword tagger (`config.toml [prescreen.keywords]`), not a person. Report:
   - the share of annotated posts with `cultural_concept ≠ none`, split by pre-screen `concept_likely` = `yes` / `no`;
   - the pre-screen's sensitivity and specificity against the gold concept label;
   - agreement between `domain_guess` and the gold domain.

## 3. RQ1: emotion recognition (H1a, H1b, H3a)

**Data:** ≈400 Task A outputs (≈200 posts × 2 models). Ratings are not needed. Because the keyword pre-screen favors posts that name a concept outright (v1.3), the by-`expression` breakdown below is also read as a check on that bias.

**Descriptives, per model:**
- accuracy, macro-F1, and per-class precision, recall and F1 for the emotion label;
- the same metrics for the cultural-concept label, as a secondary measure, using the schema's fixed `concept_aliases` (v1.1). Also report the share of concept outputs that needed an alias, per model, and the concept metrics with those outputs counted as invalid instead;
- the invalid-output rate;
- confusion matrices;
- a majority-class baseline (always predict the most frequent gold label).

**CIs:** percentile bootstrap that resamples **posts** (B = 2,000, `SEED = 42`), applied to accuracy and macro-F1, overall and by `emotion_type` and `expression`.

**Model (primary):** logistic regression fitted with **GEE** (binomial, logit link, exchangeable correlation, clustered by post), which accounts for the same post being read by both models (see §12 for why GEE and not a mixed logistic model)

```
correct ~ emotion_type * model + expression,   groups = post
```

| Hypothesis | Estimate | Direction |
| --- | --- | --- |
| H1a culture-bound < shared | OR of `emotion_type` (culture_bound vs shared), averaged over models | OR < 1 |
| H1b implicit < explicit | OR of `expression` (implicit vs explicit) | OR < 1 |
| H3a SEA-LION > Gemma on culture-bound | `sealion − gemma` contrast within `culture_bound` (from the interaction) | OR > 1 |

**Paired check:** McNemar test of Gemma vs. SEA-LION correctness on the same posts. It is run overall and on the culture-bound subset (H3a), with the effect reported as the discordant-pair OR.

**Calibration (descriptive):** a reliability diagram and expected calibration error (ECE) of `confidence / 100`, per model. In the pilot, confidence took only 2–3 values (85–95), so this is reported as descriptive only, with the confidence distribution, and is not interpreted as a calibration finding (§11, v1.1).

## 4. RQ2: advice quality vs. community (H2a, H2b)

### 4.1 Ratings (H2a)

**Data:** author ratings on the rated set, P0 replies plus community replies (3 × `N_eval` items). One linear mixed model per dimension *d* ∈ {empathy, cultural_fit, helpfulness}, with rating in points (1–5), a random intercept for post, and rater as a fixed effect (only 3–4 authors; see §12):

```
rating_d ~ source + rater + (1 | post)
```

All effects are **mean differences in rating points**, with 95% CIs, so they can be compared directly with the margins in §8.

**Contrasts:** `community − gemma` and `community − sealion`.

**Length sensitivity (v1.1):** community top replies are much shorter than LLM replies (pilot medians about 37 vs 150 words). Length and source are therefore nearly collinear, so length is not added to the primary model. Instead, report `reply_words` by source, and refit each model with `log(reply_words)` added, as a sensitivity analysis. If a community − LLM contrast changes sign or its CI crosses zero, report the H2a result as length-dependent.

| Hypothesis | Test |
| --- | --- |
| Community higher on cultural fit and empathy | Directional contrast, community − LLM mean difference > 0 |
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
- **Test:** McNemar on per-post autonomy (LLM vs. community), per model, using the primary rater's codes. H2b predicts drift > 0.
- **By domain and by cultural concept:** report drift with bootstrap CIs, **descriptive only**, with no per-subgroup tests (to avoid multiplicity). Cells with fewer than 10 posts are flagged.

## 5. RQ3: model comparison on advice (H3b)

The same rating models as §4.1, P0 replies, with the `sealion − gemma` contrast.

| Hypothesis | Test |
| --- | --- |
| SEA-LION higher on cultural fit | Directional contrast, sealion − gemma mean difference > 0 |
| SEA-LION higher on register and tone | Directional contrast, sealion − gemma mean difference > 0, from `rating_register_tone ~ source + rater + (1 | post)` |
| No advantage in safe escalation | **Equivalence (TOST)** on the rate of `missing` escalation among posts where raters marked escalation as needed, with margin ±Δ_esc (§8). If fewer than 20 such posts exist, report descriptively only and say so. |

Harm-flag rates per source are reported descriptively, with Wilson CIs.

## 6. Linking question (H4a, H4b)

### 6.1 Recognition → advice (H4a, observational)

P0 ratings on the rated set, with `correct` taken from the full Task A run, per dimension:

```
rating_d ~ correct * emotion_type + model + domain + expression + post_length_words + rater + (1 | post)
```

H4a predicts a positive `correct` effect.

**Within-post contrast:** take the rated posts where exactly one model is correct. Compare the correct model's mean rating with the incorrect model's on the same post using a **Wilcoxon signed-rank** test, with rank-biserial r as the effect size. This holds the post fixed, but the models differ in more than correctness, so it is reported as an **association**. If fewer than 15 such posts exist, report descriptively only.

### 6.2 Oracle test (H4b, interventional)

Per rated post and model, Δ = mean rating(P-Oracle) − mean rating(P0).

- **Paired test:** Wilcoxon signed-rank on Δ, per model and per `emotion_type`, with rank-biserial r.
- **Model:** `rating_d ~ condition * emotion_type * model + rater + (1 | post)`. H4b predicts the `condition × emotion_type` interaction to be positive for `culture_bound`.

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
- Fit `rating_d ~ source + (1 | post) + (1 | respondent)` (linear mixed model; respondents are many, so they stay random, fitted as crossed variance components in `statsmodels`) for empathy, cultural fit and helpfulness, with the same contrasts as §4.1. Report them next to the author results.
- **Author–reader agreement:** per reply, the Spearman ρ between the author mean and the reader mean per dimension (bootstrap CI). Also agreement on best-fit choices (% agreement and Cohen's κ).
- Survey reliability: Krippendorff's α (ordinal) per dimension.

If it does not run, report this as a construct limitation: the ratings come from the authors, not from lay Filipino readers.

## 8. Pre-registered margins (fixed 2026-11-18)

| Symbol | Used in | Value | Rationale |
| --- | --- | --- | --- |
| Δ | H2a non-inferiority (helpfulness) | 0.5 points on the 1–5 scale | A difference smaller than half a rating step is treated as practically negligible. With 60 posts (v1.3) the 95% CI half-width is about ±0.25 points (±0.2 at the original 80), so this margin is still testable. |
| Δ_esc | H3b safe-escalation equivalence | 10 percentage points | A gap under 10 points in missed escalations is treated as no practical difference. Likely unused: if fewer than 20 posts need escalation, H3b escalation is descriptive (§5). |
| Δ_oracle | H4b "substantial" improvement | 0.5 points on the 1–5 scale | Same reasoning as Δ. A smaller but real gain is read as "Oracle improves advice little" (response bottleneck), as intended by the interpretation table in §6.2. |

## 9. Reliability

| Id | What | Statistic |
| --- | --- | --- |
| R1 | Annotation (≥ 25% double-annotated) | Cohen's κ per field (`primary_emotion`, `cultural_concept`, `expression`), plus one-vs-rest κ per emotion label; the gate is in §2, step 2 |
| R2 | Strategy coding (double-coded items) | Cohen's κ between the two authors |
| R3 | Author ratings (≈ 25% double-rated) | Krippendorff's α (ordinal) per dimension; percent agreement on safe escalation and best fit |
| R4 | Generation variance (sampled runs on a subset) | Agreement of Task A labels across samples and with the greedy run; for advice, the spread of reply length across samples (the strategy-change share was dropped in v1.2); compare with the size of the effects above |

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
| v1 | 2026-11-18 | Frozen. `N_eval` = 80; margins fixed (§8); Python-only tooling (§12): linear mixed models for ratings with rater as a fixed effect, GEE for the logistic model in §3. |
| v1.1 | 2026-11-23 | After the 30-post pilot (health checks only, no accuracy viewed): calibration marked descriptive only (§3), because confidence was near-constant; length sensitivity analysis added for H2a (§4.1), because community replies were about 4× shorter. Concept metrics use the fixed alias map, and the strict version is reported alongside (§3), because the models put emotion ids (`pakikisama_strain`, `utang_na_loob_guilt`) in the concept field even under prompt v2. Run changes outside this file: Task A prompt v2, the `concept_aliases` parsing rule in the schema, and `max_new_tokens.advice` raised from 320 to 400. |
| v1.2 | 2026-11-23 | Implementation decisions made while writing the code (`03-evaluation/payo_eval/`), before annotation: (1) merges (§2) are applied at **scoring time** to gold and predictions alike, from `config.toml [merges]`; prompts, schema and the adjudicated labels (the P-Oracle input) keep the original label. (2) R4's advice strategy-change share is dropped (it needed about 360 extra hand-coded replies); R4 keeps Task A stability and adds reply-length spread. (3) With more than two annotators or raters, κ pools first vs. second coder across pairs. (4) The rated set is allocated proportionally across `emotion_type` × `domain` cells (largest remainder). (5) Whole posts go to raters, so a double-rated post is double-rated on all 5 items; raters avoid posts they annotated or adjudicated where the roster allows, and conflicts are recorded. (6) The reader survey (§7) does not run; the authors' ratings are reported with the construct limitation stated in §7. |
| v1.3 | 2026-11-24 | Changes to save team time, made before any sampling, annotation or model run on study posts: (1) **No shared pilot annotation.** The codebook (v1) is used as frozen; the 25% double annotation and the κ gate (§2) are the only agreement check, and they come after annotation. (2) **Keyword pre-screen.** The manual pre-screen is replaced by fixed word lists applied in code (`config.toml [prescreen.keywords]`; no LLM, no person): `concept_likely` from concept cue words, `domain_guess` by the codebook §7 rule, and crisis terms excluded. Annotators still apply the full exclusion screen. Its selection bias is reported as in §2 step 3. (3) **Smaller sizes:** target 200 posts (draw 236 at 0.85 retention, since exclusions now happen almost only at annotation) and `N_eval` = 60, the floor in §1.1 for inferential H4. (4) **Order:** Task A and P0 advice run on the sampled posts before annotation; the outputs stay unopened and uncommitted until `finalize-labels` has frozen the labels, so annotation stays blind. P-Oracle runs after, since it needs the gold labels. |
| v1.4 | 2026-11-24 | All GPU work before any human step: (1) **P-Oracle is pre-generated.** For each rated-set candidate, both models answer the P-Oracle prompt once for every emotion × concept pair (13 × 8 = 104; `run.py --task advice-oracle-grid`). After `finalize-labels`, the reply whose prompt carried the post's gold (unmerged) labels is selected (`payo_eval/oracle.py`). With greedy decoding this is the reply a post-annotation run would give; vLLM batching can cause tiny numeric differences, which a later run would have too. Nobody sees the grid before the labels are frozen. (2) **The rated set is drawn up front**: 71 candidates (`ceil(60 / 0.85)`), stratified by the pre-screen stratum (arm × `domain_guess`) instead of gold `emotion_type` × `domain` (§1.1). The rated set is the first 60 candidates, in a fixed random order, that are kept after annotation. The realized `emotion_type` × `domain` mix is reported. (3) No P-Oracle variance runs; R4 uses recognition and P0. |
| v1.5 | 2026-11-26 | Advice length calibrated to the upper quartile of community comments (~90 words, max 110; prompts `advice-p0_v2.md`, `advice-oracle_v2.md`; `max_new_tokens.advice` = 250), reducing the length asymmetry and protecting rater blinding while preserving the 3-beat empathy structure. |
| v1.6 | 2026-11-26 | Schema v2 update: `utang_na_loob_guilt` replaced by `konsensya` (shared); `pakikisama_strain` retired; `legacy_id_map` added for migration; native terms added; `sadness` narrowed; Task A prompt v3 (`emotion-recognition_v3.md`). Grid scale is now 12 emotions × 8 concepts = 96 per candidate. |
| v1.7 | 2026-11-27 | Pilot of schema v2 / prompts v3 (Task A) and v2 (advice), health checks only, no accuracy viewed. (1) Task A: valid 97% (Gemma; one `"none"` instead of `none_practical`) and 100% (SEA-LION); no emotion ids in the concept field now that `concept_aliases` is gone; `konsensya` is the top prediction (41% / 30%, under the 0.50 collapse threshold); `fear`, `jealousy`, `sama_ng_loob` never predicted; confidence still takes two values (90, 95). (2) P0 advice: median 99.5 / 101 words (was ~150); 20% / 27% over the 110-word cap; one SEA-LION reply truncated at 250 tokens (170 words). (3) First P-Oracle (v2) pilot, as a 2-post oracle-grid smoke test (2 × 96 pairs × 2 models, 0 errors, 0 truncated, max 225 output tokens): median 104.5 / 106 words, 47% / 48% over 110. Over-cap is set by the post, not the condition: on the short post 0% of grid replies exceed 110 (P0 78–88 words); on the long post 95–97% do (P0 121–128). Grid median ≈ P0 on the same post, so the oracle prompt does not lengthen replies. The pilot report now checks the 110-word cap (`payo_llm/pilot.py`). No prompt or `max_new_tokens` change yet; any change is a new prompt version and is logged here. |
| v1.8 | 2026-11-27 | Advice prompts and rater-facing text fixed after three pilot rounds (health checks only, no accuracy viewed). Prompt rewording did not hold on the 4B models: v3 (firm 110-word cap, opener ban) gave 33–43% of P0 replies over 110 words, 87% opening "Okay so", and replies drifted to English (Filipino function-word share 1–4% vs 10–16% in v2 and 22% in community replies); v4 (4–5 sentences, Taglish first sentence) gave 88% over 110 in the first 16 Gemma replies and was stopped. **Adopted:** (1) `advice-p0_v5` / `advice-oracle_v5` = v2 wording plus one line against emoji and persona claims; pilot: median 99 / 98.5 words, emoji 3% / 3% (v2: 17% / 37%), persona claims 0 / 0, Filipino share 15% / 20%, 0 truncated. (2) A fixed cleanup (`02-llm/scripts/payo_llm/cleanup.py`) applied to **every** rater-facing reply, community included, when rating sheets are built: remove emoji; remove a leading interjection followed by punctuation ("Uy,", "Naku!", "Okay, so"); drop an unfinished last sentence of a truncated reply; trim to the last whole sentence within 110 words (hard cut only if one sentence exceeds it). Raw output in the run JSONL is never changed. The rating key stores `reply_words` (cleaned, what raters see, used in §4.1 length sensitivity), `reply_words_raw` and `clean_ops`; the share of replies trimmed per source is reported. Pilot after cleanup: 0% over 110 for both models (trimmed 6/30 and 5/30). (3) `max_new_tokens.advice` 250 → 320. (4) Task A prompt v4 adds that the no-emotion id is `none_practical`, not `none`; pilot: 0 / 0 invalid emotions. **Not changed:** emotion-reading instructions in P0, kept out so the P0 vs P-Oracle contrast stays clean. **Residual limitation:** after cleanup, 30% / 43% of LLM replies still open with an empathy formula ("Naiintindihan ko…", "Gets ko…"); it carries content, so it is not removed, and it is reported as a possible source cue for raters. |

## Threats specific to author rating

- **Raters know the label set and set the gold emotions.** Mitigations: blinded `item_id`, shuffled order, P0 and Oracle mixed together, raters skip posts they annotated where possible, and R3 α is reported.
- **Authors are not lay readers.** The survey (§7) does not run (v1.2), so this is reported as a construct limitation.
- **Best-fit candidates reveal the condition split.** The best-fit sheet lists only the three P0-level replies, so a rater can tell that the other two replies of a post were P-Oracle LLM replies. The ratings are made first, on the rating sheet, where all five replies are mixed.
- **The rated set is smaller than the full run** (`N_eval` = 60 vs. ≈200), so H2–H4 have less power than RQ1. Report this alongside the effect sizes.

## Threats from the v1.3 shortcuts

- **The keyword pre-screen over-represents explicit posts.** Posts that name a cultural concept outright are more likely to enter the culture arm. Report recognition by `expression`, and the pre-screen's sensitivity and specificity (§2 step 3).
- **The rated set is stratified by pre-screen tags (v1.4)**, not by gold emotion type, so the culture-bound vs. shared split in the rated set is not fixed in advance. Report it, and read H4b's interaction with that n in mind.
- **No pilot calibration.** Unclear codebook rules surface only at the κ gate. A label that fails is revised and re-adjudicated or merged (§2), and this is reported.

## 12. Tooling (decided: Python only)

All analyses run in Python inside the project's `uv` environment. Dependencies to add when the notebooks are written: `statsmodels`, `scipy`, `scikit-learn`, `krippendorff`.

| Analysis | Tool |
| --- | --- |
| Ratings (§4.1, §5, §6) | `statsmodels` `MixedLM`: linear mixed model, random intercept for post, rater as a fixed effect |
| Recognition correctness (§3) | `statsmodels` `GEE`: binomial, exchangeable correlation, clustered by post |
| McNemar, Wilcoxon, TOST, Spearman | `statsmodels` / `scipy` |
| Bootstrap CIs | `numpy`, resampling posts, `SEED = 42` |
| κ, α | `scikit-learn` (`cohen_kappa_score`), `krippendorff` |
| Holm correction | `statsmodels` `multipletests(method="holm")` |

**Why a linear and not an ordinal mixed model.** Python has no frequentist ordinal model with random effects. Treating 1–5 ratings as numbers keeps the margins (§8) in rating points and keeps p-values and the Holm correction as planned. This is common practice for rating-scale outcomes, and it is reported as a limitation. If time allows, an ordinal sensitivity check can be added with `bambi` (Bayesian cumulative model); it would be labeled as a sensitivity analysis.

**Why rater is a fixed effect.** With only 3–4 authors rating, a rater variance component can't be estimated reliably. A fixed rater term still adjusts for raters who are harsher or more lenient overall.

**Why GEE for recognition.** `statsmodels` has no frequentist mixed logistic model. GEE clustered by post gives odds ratios with cluster-robust standard errors and handles the same pairing (both models on one post). The ORs are population-averaged rather than per-post, and this is noted in the methods.

**Methodology wording to update.** "Ordinal mixed models" becomes "linear mixed models (ratings treated as interval), with post as a random intercept and rater as a fixed effect". "Mixed-effects logistic regression (post as random effect)" becomes "logistic GEE clustered by post".
