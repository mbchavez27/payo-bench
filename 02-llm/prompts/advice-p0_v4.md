---
task: advice-generation
condition: P0
version: v4
changes_from_v3: >
  After the v3 re-pilot (analysis plan v1.8): v3's word cap and opener ban did not hold (33-43% over
  110 words; 87% of replies opened "Okay so"; replies drifted to English). v4 sets length as
  sentences (one paragraph, 4-5 sentences), replaces the opener ban with a positive rule (first
  sentence in Taglish, about something specific in the post), and keeps the no-emoji and no-persona rules.
placeholders: [reply_language, post]
output_format: free text, one paragraph, 4-5 sentences (~90 words)
decoding_note: greedy for the main run; max_new_tokens 320 so a ~90-word Taglish reply is never cut off. Log truncation.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write one paragraph of 4 to 5 sentences (about 90 words).
- Begin with a sentence in Taglish about something specific in this post. No greeting, no sign-off, no emoji.
- Do not claim personal experiences, a gender or an identity, and do not mention that you are an AI.
- Talk directly to the poster. No headings and no bullet points.

Post:
{{post}}

Reply:
