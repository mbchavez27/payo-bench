---
task: advice-generation
condition: P0
version: v2
placeholders: [reply_language, post]
output_format: free text, ~90 words
decoding_note: greedy for the main run; max_new_tokens ~250 so a ~90-word Taglish reply is never cut off. Log truncation.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 90 words, and no more than 110.
- Talk directly to the poster. Use plain paragraphs: no headings, no bullet points, and do not mention that you are an AI.

Post:
{{post}}

Reply:
