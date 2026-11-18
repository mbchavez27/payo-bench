---
task: advice-generation
condition: P-Oracle
version: v1
placeholders: [reply_language, emotion_name, emotion_definition, concept_name, concept_definition, post]
output_format: free text, ~150 words
rendering: >
  emotion_* and concept_* come from the ADJUDICATED gold labels (codebook §8), never from the
  model's own Task A output. If the gold concept is "none", render concept_name = "None" and
  concept_definition = the schema's definition for "none". Everything outside the "Background"
  block is identical to advice-p0_v1.md; the extra length is a reported confound.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 150 words, and no more than 180.
- Talk directly to the poster. Use plain paragraphs: no headings, no bullet points, and do not mention that you are an AI.

Background from a trained reader of this post:
- The poster's main emotion: {{emotion_name}}: {{emotion_definition}}
- Cultural concept involved: {{concept_name}}: {{concept_definition}}
Use this to understand the poster. Respond to their situation; do not name or quote these labels.

Post:
{{post}}

Reply:
