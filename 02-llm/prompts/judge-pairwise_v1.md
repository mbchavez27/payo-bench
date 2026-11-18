---
task: judge-pairwise
version: v1
placeholders: [post, reply_a, reply_b]
output_format: 'JSON {"reason": "...", "winner": "A" | "B" | "tie"}'
protocol: every pair is run in both orders; a win counts only if consistent across orders, otherwise tie
---
You are comparing two replies to a help-seeking post from r/AdvicePH, a Filipino advice community. Posts and replies may be in Filipino, English, or Taglish. You do not know who wrote either reply.

Which reply would this poster most want to receive? Weigh together:
- empathy: does it correctly recognize and respond to what the poster feels?
- cultural fit: does it understand the poster's Filipino family and social context?
- helpfulness: does it give useful, feasible next steps for this poster?
Do not prefer a reply because it is longer or comes first. Answer "tie" only if they are truly equal.

Post:
{{post}}

Reply A:
{{reply_a}}

Reply B:
{{reply_b}}

Answer with JSON only, no other text:
{"reason": "<one or two sentences>", "winner": "<A|B|tie>"}
