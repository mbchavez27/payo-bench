---
task: emotion-recognition
version: v4
changes_from_v3: >
  After the 2026-11-27 pilot (analysis plan v1.8): one rule naming "none_practical" as the
  emotion id for no clear emotion, because a model wrote "none" (a concept id) in the emotion field.
placeholders: [emotion_list, concept_list, post]
output_format: 'JSON {"emotion": <emotion id>, "cultural_concept": <concept id or "none">, "confidence": <int 0-100>}'
parsing: >
  Take the first {...} object in the output. Lowercase and trim each value, then require an
  exact match to a schema id. A non-matching or missing value is scored as incorrect and
  counted separately as "invalid" (not coerced to the nearest label).
---
You will read a help-seeking post from r/AdvicePH, a Filipino advice community. The post may be in Filipino, English, or Taglish.

Your task: identify the poster's PRIMARY emotion, and the main Filipino cultural concept involved, if any.

Emotion labels (choose exactly one id):
{{emotion_list}}

Cultural concepts (choose exactly one id):
{{concept_list}}

Rules:
- Choose the emotion the poster expresses or clearly shows most strongly, not what someone in their situation might be expected to feel.
- Use "none_practical" if the post is mainly a practical or informational request with no clear emotion.
- If the post shows no clear emotion, the emotion id is "none_practical" (not "none"). "none" is only valid for "cultural_concept".
- Choose a cultural concept only if it shapes the poster's situation or dilemma; otherwise use "none".
- "confidence" is an integer from 0 to 100 for how sure you are about the emotion label.
- The two lists use different ids. "emotion" must be an id from the emotion list, and "cultural_concept" must be an id from the cultural concept list. Never put an emotion id in "cultural_concept".
- Use the ids exactly as written above.

Post:
{{post}}

Answer with JSON only, no other text:
{"emotion": "<emotion id>", "cultural_concept": "<concept id>", "confidence": <0-100>}
