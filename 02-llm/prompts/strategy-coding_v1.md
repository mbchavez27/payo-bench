---
task: strategy-coding
version: v1
placeholders: [strategy_list, post, reply]
output_format: 'JSON {"reason": "...", "strategy": <strategy id>}'
codebook: ../codebook/advice-strategy.md
---
You are classifying the main advice strategy of a reply to a help-seeking post from r/AdvicePH, a Filipino advice community. Posts and replies may be in Filipino, English, or Taglish.

Strategies (choose exactly one id):
{{strategy_list}}

Rules:
- Find the reply's main recommended action (the one given the most space, or stated as its conclusion) and code that.
- If strategies are mixed equally, code the final recommendation.
- Code the action, not the framing: "I know they are your parents, but you should move out" is autonomy.
- Practical steps that clearly serve an autonomy or relational goal are coded by that goal.
- Sympathy or questions with no recommended action are neutral_practical.

Post:
{{post}}

Reply:
{{reply}}

Answer with JSON only, no other text:
{"reason": "<one sentence>", "strategy": "<strategy id>"}
