---
task: prescreen
version: v1
placeholders: [exclusion_list, concept_list, post]
output_format: 'JSON {"exclude_reasons": [<ids>], "cultural_concept_likely": "yes"|"no", "concepts": [<ids>]}'
use: >
  Screening aid only. Excluded posts are confirmed by a human (codebook §2) before removal, and
  the pre-screen's yes/no is used only for stratified sampling. Report its selection bias:
  compare the label distribution of pre-screened "no" posts in the low-culture control group
  against the annotated labels.
---
You are screening help-seeking posts from r/AdvicePH, a Filipino advice community, for a research dataset. The post may be in Filipino, English, or Taglish.

1. Exclusion: list every reason that applies, or an empty list.
{{exclusion_list}}

2. Cultural concept: is the poster's situation or dilemma clearly shaped by a Filipino cultural concept from this list? Answer "yes" or "no", and list the concept ids that apply (an empty list if "no").
{{concept_list}}

Post:
{{post}}

Answer with JSON only, no other text:
{"exclude_reasons": ["<id>", ...], "cultural_concept_likely": "<yes|no>", "concepts": ["<id>", ...]}
