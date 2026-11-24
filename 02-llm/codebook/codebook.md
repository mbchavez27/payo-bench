# Annotation codebook — Filipino emotions in r/AdvicePH posts (RQ1)

**Version:** v1 · **Status:** frozen (2026-11-24). Sources were checked by the team against the originals ([`source-verification.md`](source-verification.md)); full references are in §11. Any change after this bumps the version and is logged in §10.

The one-line definitions and all label ids live in [`../gold/emotion-recognition/label-schema.json`](../gold/emotion-recognition/label-schema.json). The model prompts read from that file, so **if you change a definition, change it there** and copy it here. The longer definitions below are for annotators only and are never shown to models.

---

## 0. Pre-screen (before sampling)

**Since v1.1 (analysis plan v1.3) this is done in code.** `workflow.py sample` tags every silver post with the fixed word lists in `03-evaluation/config.toml [prescreen.keywords]`, using no LLM and no person: `concept_likely` = `yes` if a concept cue word appears, `domain_guess` by the §7 rule (the relationship with the most cue words; `money` only if no relationship is mentioned), and `exclude_flag` = `yes` for crisis terms. The matched words are saved in `prescreen.csv`. The manual pass below is still supported (`[prescreen] method = "manual"`).

Manual alternative: before the posts are sampled, one author does a quick pass over a seeded random pool of silver posts (900 by default, `03-evaluation/config.toml [prescreen] n`; pilot posts excluded; about 30 seconds per post) in the sheet made by `workflow.py prescreen-sheet` (columns as in [`prescreen-template.csv`](prescreen-template.csv)):

| Field            | Values                                                             |
| ---------------- | ------------------------------------------------------------------ |
| `exclude_flag`   | `yes` / `no`, using the criteria in §2                             |
| `exclude_reason` | one id from `exclusion_reasons`, or empty                          |
| `concept_likely` | `yes` / `no`: does a Filipino cultural concept (§5) seem to shape the situation? |
| `domain_guess`   | one domain id (§7)                                                 |

No LLM is used. The stratified sample (by `domain_guess` and `concept_likely`, with a low-culture control group) is drawn in code (`03-evaluation/00-sampling.ipynb`). The pre-screen is a quick guess: the full annotation below still applies the exclusion screen again and assigns the final labels. Its selection bias is measured in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) §2.

## 1. What you label

For each post (Problem/Goal + Context + Previous Attempts), fill one row of [`annotation-template.csv`](annotation-template.csv):

| Field              | Values                                                         | Section |
| ------------------ | -------------------------------------------------------------- | ------- |
| `exclude_flag`     | `yes` / `no`                                                   | §2      |
| `exclude_reason`   | one id from `exclusion_reasons`, or empty                      | §2      |
| `primary_emotion`  | one emotion id                                                 | §3–§4   |
| `cultural_concept` | one concept id, or `none`                                      | §5      |
| `expression`       | `explicit` / `implicit`                                        | §6      |
| `domain`           | `family` `romantic` `friendship` `work_school` `money` `other` | §7      |
| `need_type`        | `emotional` / `practical` / `both`                             | §7      |
| `notes`            | free text: doubts, second-choice label, anything for adjudication | —    |

`emotion_type` (culture-bound vs. shared) is **not** annotated. It is fixed by the schema, so it can't be influenced by model output.

Label from the **poster's text only**. Do not read the replies first, and do not open the Reddit thread.

---

## 2. Exclusion screen (do this first)

If any of these applies, set `exclude_flag = yes`, pick the main reason, and **stop**: excluded posts get no further labels.

| Reason             | Exclude when                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `self_harm_crisis` | Suicidal thoughts, self-harm, or an active emergency (e.g. "iniisip ko nang tapusin lahat").              |
| `minor_or_abuse`   | The poster or a central person is a minor in a sensitive situation, or there is physical, sexual, or domestic abuse. |
| `sexual_content`   | Explicit sexual content, or the question is about sex acts.                                               |
| `doxxing_prone`    | The story could still identify real people after masking: a named company or school plus unique events, a public figure, a viral incident. |
| `bot_or_spam`      | Generated, promotional, or spam text.                                                                     |

When unsure, flag it and write why in `notes`. Adjudication decides. Report how many posts were excluded for each reason.

---

## 3. Primary emotion: decision rules

1. **The poster's felt state, not the situation.** Label what the poster expresses or clearly shows, not what someone in that situation *should* feel. If a loan is unpaid but the poster only asks how to file a small-claims case, the label is `none_practical`, not `anger`.
2. **One label: the most salient.** If several emotions appear, pick the one that drives the request. This is usually what the Problem/Goal is about. Put the runner-up in `notes`.
3. **Culture-bound vs. shared tie-break.** If a culture-bound label and a shared label both fit equally, choose the **culture-bound** label *only if* its defining feature (in §4) is present in the text. Otherwise choose the shared label. Example: hurt at a partner is `tampo` only if there is withdrawal plus a wish to be noticed or won back. Otherwise it is `sadness` or `anger`.
4. **Emotion ≠ cultural concept.** The emotion is what the poster feels. The concept (§5) is the cultural norm framing the situation. They are labeled separately and do not have to match. A poster can feel `worry` in a `family_obligation` situation.
5. **`none_practical`** is used when the post is a request for information, procedure, or a decision, with no emotion expressed beyond ordinary politeness or urgency. "Please help po, urgent" alone is not an emotion.

---

## 4. Emotion labels

### Culture-bound

#### `hiya`
**Definition.** Shame or social embarrassment at being seen by others as improper, a burden, or out of place. It often stops the person from asking, refusing, or speaking up.
**Defining feature.** An *audience*: the poster worries how they look to others, and this holds them back.
**Use when.** "Nahihiya akong humingi", "nakakahiya sa kanila", avoiding asking for what they are owed, feeling unable to face people after an incident.
**Don't confuse with.** `utang_na_loob_guilt` (the pull is a *debt*, not looking bad); `worry` (fear of an outcome, with no social audience). Note: *hiyang* (suited to, e.g. a medicine) is a different word.
**Example (paraphrased).** A poster has needed to ask a coworker to return a loan for months but can't bring it up because it would make things awkward and make them look petty.
**Source.** Bulatao (1964, p. 428): hiya as a painful emotion in a social or authority relationship that inhibits self-assertion. Enriquez (1992) and Pe-Pua & Protacio-Marcelino (2000) for hiya as *propriety*, not only shame.

#### `tampo`
**Definition.** Hurt, sulking withdrawal toward someone close who did not give expected care or attention, usually hoping to be noticed and won back (*nilalambing*).
**Defining feature.** Withdrawal plus an unspoken wish for the other person to notice and make up. It is **mild and repairable**.
**Use when.** Going quiet after a partner forgot to greet them at midnight; not replying so the other person will reach out first; "nagtatampo ako".
**Don't confuse with.** `sama_ng_loob` (deeper, accumulated, and no longer just waiting to be appeased); `sadness` (no withdrawal directed at a person); `anger` (open confrontation).
**Example (paraphrased).** A poster's partner spent New Year's Eve drinking with family and forgot to call. The poster stopped replying and is now asking if they are being unreasonable.
**Source.** Church, Katigbak, Reyes & Jensen (1998, pp. 66–67, 81): tampo as a culture-specific Filipino emotion concept.

#### `sama_ng_loob`
**Definition.** Deep, lingering hurt and resentment toward someone close, built up from repeated or serious wrongs and often left unspoken.
**Defining feature.** **Accumulation** (a history of hurts) plus a relationship that *continues* despite the resentment.
**Use when.** Years of being taken for granted by family; "first sama ng loob of the year" after yet another request for money; hurt at a parent's favoritism.
**Don't confuse with.** `tampo` (a single, recent, repairable episode); `anger` (an acute reaction to one event, expressed openly).
**Example (paraphrased).** A poster's in-law keeps borrowing money and never repays. The poster finally said no, but still carries the hurt from all the earlier times.
**Source.** Alejo (1990, pp. 82–95) on sama ng loob as a wounding of *loob* held inside to preserve the relationship; Church et al. (1998, p. 66) for its status as a Filipino emotion concept.

#### `pikon`
**Definition.** Irritability or loss of composure triggered by teasing, banter, or minor slights, often accompanied by self-consciousness or social stigma for failing to take a joke.
**Defining feature.** A **trigger that others consider minor** (teasing, a joke) plus a reaction the poster now questions ("pikon ba ako?").
**Use when.** Barkada keep joking about the poster's weight and they snapped; asking if they are too sensitive to a coworker's banter.
**Don't confuse with.** `anger` (the trigger is a real wrong, and the poster doesn't question their reaction); `hiya` (embarrassment with no provocation).
**Example (paraphrased).** A poster walked out of a family gathering after cousins kept teasing them about being single, and now wonders whether they overreacted.
**Source.** Pe-Pua & Protacio-Marcelino (2000, pp. 56–57) on *biro* (teasing) norms and being pikon. The self-questioning defining feature is the team's operationalization: the literature treats the affect itself as irritability or anger, not a separate emotion.
**Team operationalization.** The "pikon ba ako?" feature is ours, not from the literature (outcome b, `source-verification.md`).
**Note.** This is likely rare (only 5 posts in silver use the word). It may need to be merged (§9).

#### `utang_na_loob_guilt`
**Definition.** Guilt or pressure from a debt of gratitude: feeling unable to refuse, or obliged to repay, someone (often family) who helped or raised them.
**Defining feature.** An explicit or clearly implied **debt**: "they raised me / paid for my school / helped me before, so...".
**Use when.** Feeling unable to refuse a relative's loan because they once paid the poster's tuition; guilt about moving out after parents "sacrificed everything".
**Don't confuse with.** `hiya` (no debt, only how one looks); `burnout` (exhaustion from supporting family, with no guilt or debt framing). If both apply, use the one the poster emphasizes.
**Example (paraphrased).** A poster wants to stop giving most of their salary to their parents, but feels like an ungrateful child because their parents worked abroad to pay for their education.
**Source.** Kaut (1961, pp. 256–260) and Hollnsteiner (1961/1973) on the debt of gratitude and the pressure, *hiya* and inability to refuse it creates. Enriquez (1992) for utang na loob as gratitude and solidarity, which separates the value (concept `utang_na_loob`) from this distressing emotion.

#### `pakikisama_strain`
**Definition.** Psychological distress or exhaustion resulting from yielding to group or interpersonal expectations to maintain smooth interpersonal relations (*pakikisama*) at the expense of one's own wishes, values, or boundaries.
**Defining feature.** **Conformity for harmony** that the poster resents or is tired of.
**Use when.** Being forced to drink at office gatherings to "makisama"; always agreeing to barkada plans they can't afford; not wanting to be called *KJ*.
**Don't confuse with.** `hiya` (the focus is how one looks, not going along with the group); `burnout` (exhaustion with no group-conformity element).
**Example (paraphrased).** A new employee keeps joining after-work drinking sessions they can't afford because declining would make them seem *suplado*.
**Source.** Lynch (1962/1973) on pakikisama and smooth interpersonal relations; Enriquez (1992) on the cost of surface-level pakikisama without *pakikipagkapwa*. The sources describe the norm; the *strain* category is the team's affective operationalization of its personal cost.

### Broadly shared

| id         | Definition                                                                                   | Use when / boundary                                                                                       | Source                                     |
| ---------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `worry`    | Anxious, repeated thinking about what might go wrong in an uncertain situation.             | Future-oriented ("baka...", "paano kung..."). Not `fear`, which is about a present, specific threat.     | Borkovec et al. (1983, p. 10). Not Ekman: worry is not a basic emotion |
| `sadness`  | Feeling down, hurt, or grieving after a loss, rejection, or disappointment.                  | Breakups, loss, being let down. Use `tampo` or `sama_ng_loob` if their defining features apply.          | Ekman (1992, pp. 170, 177)                 |
| `anger`    | Open frustration or rage at being wronged, blocked, or treated unfairly.                     | "Galit na galit ako", "nakakainis". Not `pikon` (minor trigger questioned) or `sama_ng_loob` (accumulated, unspoken). | Ekman (1992, pp. 170, 177)       |
| `fear`     | Feeling unsafe or afraid of a specific, present threat to oneself or others.                 | Threats, stalking, unsafe homes (check §2 first: abuse is excluded).                                     | Ekman (1992, pp. 170, 177)                 |
| `burnout`  | Emotional and physical exhaustion from prolonged demands, with loss of motivation or feeling numb. | "Pagod na pagod na ako", "wala na akong pake", long-running school, work, or caregiving load.       | Maslach & Leiter (2016, p. 103)            |
| `jealousy` | Distress at a threat to a relationship from a rival (*selos*), or at someone having what one lacks (*inggit*). | Includes both selos and envy. Write `selos` or `inggit` in `notes` so they can be split later. | Parrott & Smith (1993, pp. 906–907). They separate jealousy and envy; this label merges selos and inggit on purpose |
| `none_practical` | No clear emotion is expressed; mainly a practical, procedural, or informational request. | See §3 rule 5.                                                                                      | —                                          |

---

## 5. Cultural concept

Pick the **one** cultural concept that most shapes the situation or the poster's dilemma, or `none`. A concept applies only if removing it would change what the right advice is.

| id                   | Definition                                                                                   | Typical cues                                                  | Source                         |
| -------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------ |
| `utang_na_loob`      | A lasting debt of gratitude to someone who helped you, expected to be repaid through loyalty or help. | "pinag-aral ako", "malaki utang na loob ko"             | Hollnsteiner (1961/1973); Enriquez (1992) |
| `hiya`               | Social norm of propriety and avoiding shame or loss of face in front of others. | "ano sasabihin ng kapitbahay", "nakakahiya sa pamilya"      | Bulatao (1964, p. 428); Enriquez (1992) |
| `pakikisama`         | Getting along with the group and avoiding open conflict to keep relationships smooth.        | "makisama ka na lang", "baka sabihing KJ"                     | Lynch (1962/1973); Enriquez (1992) |
| `amor_propio`        | Personal pride and self-esteem that is easily wounded by criticism or loss of face.          | refusing help out of pride, deeply hurt by public criticism   | Lynch (1962/1973)              |
| `family_obligation`  | Duty to support or prioritize the family, such as being the breadwinner or panganay, sending money, or caring for parents. | "breadwinner", "panganay", "padala", "ako lahat sumasalo" | Medina (2001); Hollnsteiner (1961/1973) |
| `respect_for_elders` | Deference to parents and older relatives, where disagreeing openly is seen as disrespectful. | "walang galang", "sumasagot", "magulang mo pa rin yan"        | Jocano (1997); Medina (2001)   |
| `bahala_na`          | Facing an uncertain situation by leaving the outcome to fate or God and pushing ahead.        | "bahala na", "ipagdasal na lang"                              | Bostrom (1968, p. 401); Enriquez (1992) |
| `none`               | No specific Filipino cultural concept shapes the situation.                                  | —                                                             | —                              |

`hiya` appears both as an emotion and as a concept: a poster may feel `hiya` (emotion), and a situation may be shaped by the hiya norm (concept). Label each on its own terms.

---

## 6. Explicit vs. implicit expression

- **`explicit`**: the poster names the chosen emotion, or a close synonym, in any language: *nahihiya, nagtatampo, masama ang loob, naiinis, takot, selos, pagod na pagod, guilty*.
- **`implicit`**: the emotion is inferable only from events, behavior, or tone.

This refers to the label you chose. If the poster writes "naiinis ako" but you labeled `tampo`, it is `implicit` unless tampo or a synonym is also named.

---

## 7. Domain and need type

**Domain:** the relationship or area where the problem sits. If it spans two (e.g. money owed by a sibling), choose the relationship (`family`). Use `money` only when no close relationship is central.

**Need type:** what the poster asks for. Read the Problem/Goal first.
- `emotional`: comfort, validation, "am I wrong?", "kailangan ko lang ng kausap".
- `practical`: steps, information, a decision, "paano...".
- `both`: clearly asks for both.

---

## 8. Annotation process

1. **Pilot (30 posts), skipped in v1.1.** The plan was for all annotators to label the same 30 posts independently, then discuss every disagreement. To save time it was skipped (analysis plan v1.3): the codebook is used as frozen, and the double-annotated subset (step 3) is the agreement check. Pilot posts are still excluded from the sample.
2. **Main pass.** Each post gets one annotator. **At least 25%** (≈59 of 236) are double-annotated, chosen at random (`SEED` logged) and not revealed to annotators.
3. **Agreement.** Computed on the double-annotated subset. The statistics and the κ gate for revising or merging a label are in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) (§2, §9 R1).
4. **Adjudication.** A third team member resolves the double-annotated disagreements, and the adjudicated label is final. One fixed adjudicator (`config.toml [roster] adjudicator`) does all of them; the sampling step never gives them a double-annotated post, so they are never one of the pair. These final labels are also the **gold emotion** used in the P-Oracle condition.
5. **Blindness.** The models may run on the sampled posts before or during annotation, but nobody opens their output until labeling is finished and frozen (`finalize-labels`).

## 9. Merging rare labels

Rare labels are merged after adjudication and before any model output is viewed. The threshold and the proposed merges are a statistical decision, so they are kept in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) §2. Merges are recorded in `03-evaluation/config.toml [merges]` and applied at scoring time (analysis plan v1.2): the schema, the prompts and the adjudicated labels keep the original label. `workflow.py finalize-labels` stops if a rare label has no merge target. Log any merge in §10.

## 10. Changelog

| Version  | Date       | Change        |
| -------- | ---------- | ------------- |
| v1-draft | 2026-11-18 | First draft.  |
| v1       | 2026-11-24 | Sources verified by the team ([`source-verification.md`](source-verification.md)); references added (§11). Definitions revised for `pikon` and `pakikisama_strain` (emotions) and `hiya` (concept), in `label-schema.json` too. The pilot model run used the v1-draft wording; pilot outputs are not part of the study. |
| v1.1     | 2026-11-24 | Process only, no definition changed: keyword pre-screen in code (§0), shared pilot skipped (§8.1), models may run before annotation but stay unopened (§8.5). See analysis plan v1.3. |

## 11. References

- Alejo, A. E. (1990). *Tao po! Tuloy!: Isang landas ng pag-unawa sa loob ng tao*. Office of Research and Publications, Ateneo de Manila University.
- Borkovec, T. D., Robinson, E., Pruzinsky, T., & DePree, J. A. (1983). Preliminary exploration of worry: Some characteristics and processes. *Behaviour Research and Therapy, 21*(1), 9–16.
- Bostrom, L. C. (1968). Filipino bahala na and American fatalism. *Silliman Journal, 15*(3), 399–413.
- Bulatao, J. C. (1964). Hiya. *Philippine Studies, 12*(3), 424–438.
- Church, A. T., Katigbak, M. S., Reyes, J. A. S., & Jensen, S. M. (1998). Language and organisation of Filipino emotion concepts: Comparing emotion concepts and dimensions across cultures. *Cognition & Emotion, 12*(1), 63–92.
- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion, 6*(3–4), 169–200.
- Enriquez, V. G. (1992). *From colonial to liberation psychology: The Philippine experience*. University of the Philippines Press.
- Hollnsteiner, M. R. (1961). Reciprocity in the lowland Philippines. *Philippine Studies, 9*(3), 387–413. Reprinted in F. Lynch & A. de Guzman II (Eds.), *Four readings on Philippine values* (4th ed., pp. 69–91). Ateneo de Manila University Press, 1973.
- Jocano, F. L. (1997). *Filipino value system: A cultural definition*. Punlad Research House.
- Kaut, C. (1961). Utang na loob: A system of contractual obligation among Tagalogs. *Southwestern Journal of Anthropology, 17*(3), 256–272.
- Lynch, F. (1973). Social acceptance reconsidered. In F. Lynch & A. de Guzman II (Eds.), *Four readings on Philippine values* (4th ed., pp. 1–68). Ateneo de Manila University Press.
- Maslach, C., & Leiter, M. P. (2016). Understanding the burnout experience: Recent research and its implications for psychiatry. *World Psychiatry, 15*(2), 103–111.
- Medina, B. T. G. (2001). *The Filipino family* (2nd ed.). University of the Philippines Press.
- Parrott, W. G., & Smith, R. H. (1993). Distinguishing the experiences of envy and jealousy. *Journal of Personality and Social Psychology, 64*(6), 906–920.
- Pe-Pua, R., & Protacio-Marcelino, E. A. (2000). Sikolohiyang Pilipino (Filipino psychology): A legacy of Virgilio G. Enriquez. *Asian Journal of Social Psychology, 3*(1), 49–71.

## Limitation (to report)

The taxonomy is justified through the RRL, not by expert validation. The P-Oracle condition depends on the team's own labels.
