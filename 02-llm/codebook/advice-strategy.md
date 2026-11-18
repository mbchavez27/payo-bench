# Advice-strategy codebook (drift, H2b)

**Version:** v1-draft

Each reply (Gemma, SEA-LION, or community) gets **one** code for its **dominant strategy**. The judge applies this codebook to every reply via [`../prompts/strategy-coding_v1.md`](../prompts/strategy-coding_v1.md). Two team members code a random subset of about 50 replies to check agreement (Cohen's κ).

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

Drift = (share of `autonomy` in LLM replies) − (share of `autonomy` in community replies). It is reported overall, by domain, and by cultural concept. Posts coded `neutral_practical` stay in the denominator. A version excluding them is reported as a sensitivity check.
