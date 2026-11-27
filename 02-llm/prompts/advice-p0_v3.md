---
task: advice-generation
condition: P0
version: v3
changes_from_v2: >
  After the 2026-11-27 pilot (analysis plan v1.7, v1.8): the 110-word cap is stated as firm even for
  long posts; the formula opener ("Uy, naku") and sign-offs/emoji are ruled out, since they reveal the
  source to raters; invented persona claims ("As a girl", "We're here for you") are ruled out.
placeholders: [reply_language, post]
output_format: free text, ~90 words
decoding_note: greedy for the main run; max_new_tokens 320 so a ~90-word Taglish reply is never cut off. Log truncation.
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community.

Write a reply to the poster, the way a caring and sensible commenter would.
- Write in {{reply_language}}.
- Write about 90 words. Never go over 110, even if the post is long: pick the one or two points that matter most.
- Start with your actual response, not a greeting or an interjection (for example, not "Uy" or "Naku"). Do not end with a sign-off, a well-wish or an emoji.
- Do not claim personal experiences, a gender or an identity, and do not say that you or others will be there for the poster. Do not mention that you are an AI.
- Talk directly to the poster. Use plain paragraphs: no headings and no bullet points.

Post:
{{post}}

Reply:
