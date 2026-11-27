---
task: advice-generation
condition: P0
version: v5
changes_from_v2: >
  After pilots of v3 and v4 (analysis plan v1.8): v2 wording kept, because rewording made
  replies longer or English-only; one line added against emoji and persona claims. Length and
  openers are handled by the fixed cleanup in payo_llm/cleanup.py, applied to every rater-facing reply.
placeholders: [reply_language, post]
output_format: free text, ~90 words
decoding_note: greedy for the main run; max_new_tokens 320 so a ~90-word Taglish reply is never cut off. Log truncation.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 90 words, and no more than 110.
- Talk directly to the poster. Use plain paragraphs: no headings, no bullet points, and do not mention that you are an AI.
- Do not use emoji, and do not claim personal experiences, a gender or an identity.

Post:
{{post}}

Reply:
