---
task: advice-generation
condition: P-Oracle
version: v3
changes_from_v2: >
  Same rule changes as advice-p0_v3.md (firm 110-word cap, no formula opener/sign-off/emoji, no
  persona claims). The "Background" block is unchanged.
placeholders: [reply_language, emotion_name, emotion_definition, concept_name, concept_definition, post]
output_format: free text, ~90 words
rendering: >
  emotion_* and concept_* come from the ADJUDICATED gold labels (codebook §8), never from the
  model's own Task A output. If the gold concept is "none", render concept_name = "None" and
  concept_definition = the schema's definition for "none". Everything outside the "Background"
  block is identical to advice-p0_v3.md; the extra length is a reported confound.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 90 words. Never go over 110, even if the post is long: pick the one or two points that matter most.
- Start with your actual response, not a greeting or an interjection (for example, not "Uy" or "Naku"). Do not end with a sign-off, a well-wish or an emoji.
- Do not claim personal experiences, a gender or an identity, and do not say that you or others will be there for the poster. Do not mention that you are an AI.
- Talk directly to the poster. Use plain paragraphs: no headings and no bullet points.

Background from a trained reader of this post:
- The poster's main emotion: {{emotion_name}}: {{emotion_definition}}
- Cultural concept involved: {{concept_name}}: {{concept_definition}}
Use this to understand the poster. Respond to their situation; do not name or quote these labels.

Post:
{{post}}

Reply:
