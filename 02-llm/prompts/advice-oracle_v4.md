---
task: advice-generation
condition: P-Oracle
version: v4
changes_from_v3: >
  Same rule changes as advice-p0_v4.md. The "Background" block is unchanged.
placeholders: [reply_language, emotion_name, emotion_definition, concept_name, concept_definition, post]
output_format: free text, one paragraph, 4-5 sentences (~90 words)
rendering: >
  emotion_* and concept_* come from the ADJUDICATED gold labels (codebook §8), never from the
  model's own Task A output. If the gold concept is "none", render concept_name = "None" and
  concept_definition = the schema's definition for "none". Everything outside the "Background"
  block is identical to advice-p0_v4.md; the extra length is a reported confound.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write one paragraph of 4 to 5 sentences (about 90 words).
- Begin with a sentence in Taglish about something specific in this post. No greeting, no sign-off, no emoji.
- Do not claim personal experiences, a gender or an identity, and do not mention that you are an AI.
- Talk directly to the poster. No headings and no bullet points.

Background from a trained reader of this post:
- The poster's main emotion: {{emotion_name}}: {{emotion_definition}}
- Cultural concept involved: {{concept_name}}: {{concept_definition}}
Use this to understand the poster. Respond to their situation; do not name or quote these labels.

Post:
{{post}}

Reply:
