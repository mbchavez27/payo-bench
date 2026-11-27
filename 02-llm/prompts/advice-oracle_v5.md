---
task: advice-generation
condition: P-Oracle
version: v5
changes_from_v2: >
  After pilots of v3 and v4 (analysis plan v1.8): v2 wording kept, because rewording made
  replies longer or English-only; one line added against emoji and persona claims. Length and
  openers are handled by the fixed cleanup in payo_llm/cleanup.py, applied to every rater-facing reply.
placeholders: [reply_language, emotion_name, emotion_definition, concept_name, concept_definition, post]
output_format: free text, ~90 words
rendering: >
  emotion_* and concept_* come from the ADJUDICATED gold labels (codebook §8), never from the
  model's own Task A output. If the gold concept is "none", render concept_name = "None" and
  concept_definition = the schema's definition for "none". Everything outside the "Background"
  block is identical to advice-p0_v5.md; the extra length is a reported confound.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 90 words, and no more than 110.
- Talk directly to the poster. Use plain paragraphs: no headings, no bullet points, and do not mention that you are an AI.
- Do not use emoji, and do not claim personal experiences, a gender or an identity.

Background from a trained reader of this post:
- The poster's main emotion: {{emotion_name}}: {{emotion_definition}}
- Cultural concept involved: {{concept_name}}: {{concept_definition}}
Use this to understand the poster. Respond to their situation; do not name or quote these labels.

Post:
{{post}}

Reply:
