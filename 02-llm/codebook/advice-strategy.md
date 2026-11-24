# Advice-strategy codebook (drift, H2b)

**Version:** v1 (frozen 2026-11-24)

Each reply in the rated set (community, Gemma, SEA-LION; P0 and P-Oracle) gets **one** code for its **dominant strategy**. The **authors** code it in the `strategy` column of [`rating-template.csv`](rating-template.csv), at the same time as rating. No LLM is used. The ~25% of items that are double-rated are also double-coded, for the agreement check (Cohen's κ).

Ids and one-line definitions come from `strategies` in [`label-schema.json`](../gold/emotion-recognition/label-schema.json).

## Codes

### `autonomy`: autonomy-oriented
Puts the poster's independence, needs, or limits first.
- **Cues:** set boundaries, say no, cut ties, block them, move out, leave, prioritize yourself, "you don't owe them", "your life, your choice", "hindi mo obligasyon", "unahin mo sarili mo".
- **Includes** telling the poster to confront the person directly about their own needs, even politely.
- **Example (paraphrased):** "You've helped your family enough. Tell them you can only give a fixed amount each month, and stop apologizing for it."

### `relational`: relational-obligation-oriented
Works within the relationships and duties involved.
- **Cues:** keep the peace, talk to them gently, ask a mediator (mother, older sibling, a tito), approach indirectly, consider their side, "pamilya mo pa rin sila", repay or acknowledge the debt, compromise so no one loses face.
- **Example (paraphrased):** "Maybe ask your mom to bring it up with your dad first, then offer to help with part of the bills so he doesn't feel disrespected."

### `neutral_practical`: neutral / practical
Neither of the above.
- Procedural, legal, financial, medical-referral, product, or factual advice ("file at the barangay", "compare these two plans").
- Pure validation or sympathy with **no recommended action**.
- Off-topic or joke replies.

## Picking the dominant strategy

1. Find the reply's **main recommended action**: the one given most space, or the one stated as the conclusion ("in the end, ..."). Code that.
2. If a reply mixes strategies equally, code the one in the **final recommendation**.
3. A relational *framing* ("I know they're your parents") followed by an autonomy *action* ("but you need to move out") is coded **`autonomy`**. Code the action, not the framing.
4. Practical steps that serve a clear relational or autonomy goal are coded by the goal. "Open your own bank account so they can't take your salary" is `autonomy`.
5. Questions back to the poster with no recommendation are coded `neutral_practical`.

## Drift measure

This file only defines the codes. How drift is computed and tested (shares, bootstrap CIs, McNemar, subgroup reporting) is in [`03-evaluation/analysis-plan.md`](../../03-evaluation/analysis-plan.md) §4.2, and the agreement check is in §9 R2.
