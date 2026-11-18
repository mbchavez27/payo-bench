# Annotation codebook — Filipino emotions in r/AdvicePH posts (RQ1)

**Version:** v1-draft · **Status:** draft. Definitions and sources marked `TO-VERIFY` must be checked against the team's RRL before the pilot.

The one-line definitions and all label ids live in [`../gold/emotion-recognition/label-schema.json`](../gold/emotion-recognition/label-schema.json). The model prompts read from that file, so **if you change a definition, change it there** and copy it here. The longer definitions below are for annotators only and are never shown to models.

---

## 0. Pre-screen (before sampling)

Before the ~300 posts are sampled, one author does a quick pass over the whole pool (~800–1,000 posts, about 30 seconds per post) using [`prescreen-template.csv`](prescreen-template.csv):

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
**Source.** `TO-VERIFY`: Bulatao (1964) on hiya; Lynch (1962/1973) on social acceptance; Enriquez (1992) on hiya as *propriety*, not only shame.

#### `tampo`
**Definition.** Hurt, sulking withdrawal toward someone close who did not give expected care or attention, usually hoping to be noticed and won back (*nilalambing*).
**Defining feature.** Withdrawal plus an unspoken wish for the other person to notice and make up. It is **mild and repairable**.
**Use when.** Going quiet after a partner forgot to greet them at midnight; not replying so the other person will reach out first; "nagtatampo ako".
**Don't confuse with.** `sama_ng_loob` (deeper, accumulated, and no longer just waiting to be appeased); `sadness` (no withdrawal directed at a person); `anger` (open confrontation).
**Example (paraphrased).** A poster's partner spent New Year's Eve drinking with family and forgot to call. The poster stopped replying and is now asking if they are being unreasonable.
**Source.** `TO-VERIFY`: candidate Filipino emotion-lexicon studies (e.g. Church, Katigbak and colleagues); Enriquez (1992) for *loob*-related terms.

#### `sama_ng_loob`
**Definition.** Deep, lingering hurt and resentment toward someone close, built up from repeated or serious wrongs and often left unspoken.
**Defining feature.** **Accumulation** (a history of hurts) plus a relationship that *continues* despite the resentment.
**Use when.** Years of being taken for granted by family; "first sama ng loob of the year" after yet another request for money; hurt at a parent's favoritism.
**Don't confuse with.** `tampo` (a single, recent, repairable episode); `anger` (an acute reaction to one event, expressed openly).
**Example (paraphrased).** A poster's in-law keeps borrowing money and never repays. The poster finally said no, but still carries the hurt from all the earlier times.
**Source.** `TO-VERIFY`: Alejo (1990) on *loob*; Enriquez (1992).

#### `pikon`
**Definition.** Being provoked and losing composure over teasing, jokes, or small slights, and feeling bad or judged for having reacted.
**Defining feature.** A **trigger that others consider minor** (teasing, a joke) plus a reaction the poster now questions ("pikon ba ako?").
**Use when.** Barkada keep joking about the poster's weight and they snapped; asking if they are too sensitive to a coworker's banter.
**Don't confuse with.** `anger` (the trigger is a real wrong, and the poster doesn't question their reaction); `hiya` (embarrassment with no provocation).
**Example (paraphrased).** A poster walked out of a family gathering after cousins kept teasing them about being single, and now wonders whether they overreacted.
**Source.** `TO-VERIFY`: candidate Filipino trait or emotion-lexicon studies.
**Note.** This is likely rare (only 5 posts in silver use the word). It may need to be merged (§9).

#### `utang_na_loob_guilt`
**Definition.** Guilt or pressure from a debt of gratitude: feeling unable to refuse, or obliged to repay, someone (often family) who helped or raised them.
**Defining feature.** An explicit or clearly implied **debt**: "they raised me / paid for my school / helped me before, so...".
**Use when.** Feeling unable to refuse a relative's loan because they once paid the poster's tuition; guilt about moving out after parents "sacrificed everything".
**Don't confuse with.** `hiya` (no debt, only how one looks); `burnout` (exhaustion from supporting family, with no guilt or debt framing). If both apply, use the one the poster emphasizes.
**Example (paraphrased).** A poster wants to stop giving most of their salary to their parents, but feels like an ungrateful child because their parents worked abroad to pay for their education.
**Source.** `TO-VERIFY`: Hollnsteiner (1961/1973) on reciprocity; Enriquez (1992) on utang na loob as gratitude and solidarity; Kaut (1961).

#### `pakikisama_strain`
**Definition.** Strain from going along with a group or person to keep the peace, against one's own wishes, values, or limits.
**Defining feature.** **Conformity for harmony** that the poster resents or is tired of.
**Use when.** Being forced to drink at office gatherings to "makisama"; always agreeing to barkada plans they can't afford; not wanting to be called *KJ*.
**Don't confuse with.** `hiya` (the focus is how one looks, not going along with the group); `burnout` (exhaustion with no group-conformity element).
**Example (paraphrased).** A new employee keeps joining after-work drinking sessions they can't afford because declining would make them seem *suplado*.
**Source.** `TO-VERIFY`: Lynch (1962) on smooth interpersonal relations; Enriquez (1992) on pakikisama as one level of *pakikipagkapwa*.

### Broadly shared

| id         | Definition                                                                                   | Use when / boundary                                                                                       | Source `TO-VERIFY`                         |
| ---------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `worry`    | Anxious, repeated thinking about what might go wrong in an uncertain situation.             | Future-oriented ("baka...", "paano kung..."). Not `fear`, which is about a present, specific threat.     | Borkovec et al. on worry; Ekman for basic set |
| `sadness`  | Feeling down, hurt, or grieving after a loss, rejection, or disappointment.                  | Breakups, loss, being let down. Use `tampo` or `sama_ng_loob` if their defining features apply.          | Ekman (1992)                               |
| `anger`    | Open frustration or rage at being wronged, blocked, or treated unfairly.                     | "Galit na galit ako", "nakakainis". Not `pikon` (minor trigger questioned) or `sama_ng_loob` (accumulated, unspoken). | Ekman (1992)                     |
| `fear`     | Feeling unsafe or afraid of a specific, present threat to oneself or others.                 | Threats, stalking, unsafe homes (check §2 first: abuse is excluded).                                     | Ekman (1992)                               |
| `burnout`  | Emotional and physical exhaustion from prolonged demands, with loss of motivation or feeling numb. | "Pagod na pagod na ako", "wala na akong pake", long-running school, work, or caregiving load.       | Maslach & Leiter on burnout                |
| `jealousy` | Distress at a threat to a relationship from a rival (*selos*), or at someone having what one lacks (*inggit*). | Includes both selos and envy. Write `selos` or `inggit` in `notes` so they can be split later. | Parrott & Smith (1993) on envy vs. jealousy |
| `none_practical` | No clear emotion is expressed; mainly a practical, procedural, or informational request. | See §3 rule 5.                                                                                      | —                                          |

---

## 5. Cultural concept

Pick the **one** cultural concept that most shapes the situation or the poster's dilemma, or `none`. A concept applies only if removing it would change what the right advice is.

| id                   | Definition                                                                                   | Typical cues                                                  | Source `TO-VERIFY`             |
| -------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------ |
| `utang_na_loob`      | A lasting debt of gratitude to someone who helped you, expected to be repaid through loyalty or help. | "pinag-aral ako", "malaki utang na loob ko"             | Hollnsteiner (1973)            |
| `hiya`               | The norm of avoiding shame and keeping face, including worry about what other people will say. | "ano sasabihin ng kapitbahay", "nakakahiya sa pamilya"      | Bulatao (1964); Lynch (1962)   |
| `pakikisama`         | Getting along with the group and avoiding open conflict to keep relationships smooth.        | "makisama ka na lang", "baka sabihing KJ"                     | Lynch (1962); Enriquez (1992)  |
| `amor_propio`        | Personal pride and self-esteem that is easily wounded by criticism or loss of face.          | refusing help out of pride, deeply hurt by public criticism   | Lynch (1962)                   |
| `family_obligation`  | Duty to support or prioritize the family, such as being the breadwinner or panganay, sending money, or caring for parents. | "breadwinner", "panganay", "padala", "ako lahat sumasalo" | — |
| `respect_for_elders` | Deference to parents and older relatives, where disagreeing openly is seen as disrespectful. | "walang galang", "sumasagot", "magulang mo pa rin yan"        | —                              |
| `bahala_na`          | Facing an uncertain situation by leaving the outcome to fate or God and pushing ahead.        | "bahala na", "ipagdasal na lang"                              | Bostrom (1968); Enriquez (1992) |
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

1. **Pilot (20 posts).** All annotators label the same 20 posts independently, then discuss every disagreement. Revise this codebook and log the changes in §10. Pilot posts are not part of the final set.
2. **Main pass.** Each post gets one annotator. **At least 25%** (≈75 of 300) are double-annotated, chosen at random (`SEED` logged) and not revealed to annotators.
3. **Agreement.** Computed on the double-annotated subset. The statistics and the κ gate for revising or merging a label are in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) (§2, §9 R1).
4. **Adjudication.** A third team member resolves the double-annotated disagreements, and the adjudicated label is final. These final labels are also the **gold emotion** used in the P-Oracle condition.
5. **Blindness.** Labeling is finished and frozen before any model output is viewed.

## 9. Merging rare labels

Rare labels are merged after adjudication and before any model output is viewed. The threshold and the proposed merges are a statistical decision, so they are kept in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) §2. If a merge happens, update this codebook and the schema, and log it in §10.

## 10. Changelog

| Version  | Date       | Change        |
| -------- | ---------- | ------------- |
| v1-draft | 2026-11-18 | First draft.  |

## Limitation (to report)

The taxonomy is justified through the RRL, not by expert validation. The P-Oracle condition depends on the team's own labels.
