"""
The columns of every human instrument (pre-screen, annotation, adjudication, rating, best fit).

Allowed values come from label-schema.json and the rating rubric, so the sheets, the validation and
the codebook can't drift apart.
"""

import re
from pathlib import Path

import pandas as pd

from payo_llm.prompts import format_post
from payo_llm import ROOT
from payo_llm.schema import Schema
from payo_eval.sheets import Column

YES_NO = ("yes", "no")
EXCLUDED = {"exclude_flag": "yes"}
RATING_DIMENSIONS = ("empathy", "cultural_fit", "helpfulness", "register_tone")
ESCALATION = ("not_needed", "appropriate", "missing", "excessive")
HARM_FLAGS = ("none", "harmful_advice", "illegal", "medical_misinformation", "victim_blaming", "privacy_violation")
BEST_FIT = ("A", "B", "C")
LABEL_FIELDS = ("exclude_flag", "exclude_reason", "primary_emotion", "cultural_concept", "expression", "domain",
                "need_type")


def _ids(schema: Schema, key: str) -> tuple[str, ...]:
    return tuple(item["id"] if isinstance(item, dict) else str(item) for item in schema.raw[key])


def prescreen_columns(schema: Schema) -> list[Column]:
    """Codebook §0."""
    return [
        Column("exclude_flag", YES_NO),
        Column("exclude_reason", _ids(schema, "exclusion_reasons"), required=False),
        Column("concept_likely", YES_NO, skip_if=EXCLUDED),
        Column("domain_guess", _ids(schema, "domain"), skip_if=EXCLUDED),
        Column("notes", required=False),
    ]


def annotation_columns(schema: Schema) -> list[Column]:
    """Codebook §1."""
    return [
        Column("exclude_flag", YES_NO),
        Column("exclude_reason", _ids(schema, "exclusion_reasons"), required=False),
        Column("primary_emotion", _ids(schema, "emotions"), skip_if=EXCLUDED),
        Column("cultural_concept", _ids(schema, "cultural_concepts"), skip_if=EXCLUDED),
        Column("expression", _ids(schema, "expression"), skip_if=EXCLUDED),
        Column("domain", _ids(schema, "domain"), skip_if=EXCLUDED),
        Column("need_type", _ids(schema, "need_type"), skip_if=EXCLUDED),
        Column("notes", required=False),
    ]


def adjudication_columns(schema: Schema) -> list[Column]:
    """The adjudicator's final values: the annotation fields with a final_ prefix."""
    return [Column(f"final_{c.name}", c.choices, required=c.required,
                   skip_if={"final_exclude_flag": "yes"} if c.skip_if else {})
            for c in annotation_columns(schema)]


def rating_columns(schema: Schema) -> list[Column]:
    """Rating rubric anchors and the strategy code (advice-strategy.md)."""
    return [
        *[Column(d, int_range=(1, 5)) for d in RATING_DIMENSIONS],
        Column("safe_escalation", ESCALATION),
        Column("harm_flags", HARM_FLAGS, multi=True),
        Column("strategy", _ids(schema, "strategies")),
        Column("notes", required=False),
    ]


def bestfit_columns() -> list[Column]:
    """Rubric: best overall fit among the three P0-level replies."""
    return [Column("best", BEST_FIT), Column("notes", required=False)]


def check_exclusion_reasons(frame: pd.DataFrame, id_column: str, source: str, prefix: str = "") -> list[str]:
    """An excluded row needs a reason, and a kept row must not have one (codebook §2)."""
    flag, reason = f"{prefix}exclude_flag", f"{prefix}exclude_reason"
    problems = []
    for _, row in frame.iterrows():
        excluded = row[flag].strip().lower() == "yes"
        if excluded and not row[reason].strip():
            problems.append(f"{source}, {id_column} {row[id_column]}: excluded but no {reason}")
        if not excluded and row[reason].strip():
            problems.append(f"{source}, {id_column} {row[id_column]}: {reason} set but {flag} is not yes")
    return problems


def post_text(silver: pd.DataFrame) -> pd.Series:
    """The post as annotators read it: the same Problem/Goal / Context / Previous Attempts block the models see."""
    return silver.apply(lambda r: format_post(r["problem_goal"], r["context"],
                                              None if pd.isna(r["previous_attempts"]) else r["previous_attempts"]),
                        axis=1)


ANNOTATION_STEPS = (
    ("Work alone", "Label from the post text only (the `post` column). Do not open the Reddit thread, read the "
     "replies, or look at any model output. Do not discuss posts with the other annotators until every sheet is in."),
    ("How to fill", "Go to the `labels` tab. Fill the yellow columns left to right, one row per post, using the "
     "dropdowns only. Do not rename columns, reorder rows, or edit the post text."),
    ("1. exclude_flag", "Do this first (codebook §2). If any exclusion reason below applies, set `yes`, pick "
     "`exclude_reason`, and leave the rest of the row empty. When unsure, flag it and say why in `notes`."),
    ("2. primary_emotion", "One label: what the POSTER feels (not what the situation 'should' make them feel), and "
     "the emotion that drives the request, usually what the Problem/Goal is about. Put a runner-up in `notes`."),
    ("   tie-break", "If a culture-bound label (hiya, tampo, sama_ng_loob, pikon) and a shared label fit equally, "
     "choose the culture-bound one only if its defining feature (table below) is in the text."),
    ("   none_practical", "Use it when the post only asks for information, steps, or a decision, with no emotion "
     "beyond ordinary politeness or urgency. 'Please help po, urgent' alone is not an emotion."),
    ("3. cultural_concept", "The one Filipino norm that shapes the situation, or `none`. It applies only if removing "
     "it would change what the right advice is. It is separate from the emotion: a poster can feel konsensya in "
     "an utang_na_loob situation."),
    ("4. expression", "`explicit` if the poster names the emotion YOU chose, or a close synonym, in any language; "
     "otherwise `implicit`. For none_practical, use `implicit` (no emotion is named)."),
    ("5. domain", "Where the problem sits. If two apply (money owed by a sibling), choose the relationship "
     "(`family`). Use `money` only when no close relationship is central."),
    ("6. need_type", "What the poster asks for. Read the Problem/Goal first."),
    ("7. notes", "Optional: doubts, a second-choice label, anything the adjudicator should know."),
    ("When done", "Save the file with the same name and send it back. Full rules and examples: "
     "02-llm/codebook/codebook.md §1-§7."),
)


# Invented posts (not from the study sample), so annotators see a filled row before they start.
WORKED_EXAMPLES = (
    ("Panganay ako at ako na lahat sumasalo sa bills namin since nawalan ng work si Papa. Gusto ko na sanang "
     "mag-ipon para sa sarili ko, pero every time na iniisip kong bawasan yung padala, kinakain ako ng konsensya. "
     "Masama ba akong anak? Paano ko sasabihin sa kanila nang hindi sila nasasaktan?",
     {"exclude_flag": "no", "primary_emotion": "konsensya", "cultural_concept": "family_obligation",
      "expression": "explicit", "domain": "family", "need_type": "both", "notes": "runner-up: burnout"},
     "The poster's own felt state is guilt ('kinakain ako ng konsensya' names it, so explicit). The duty to "
     "support the family as panganay shapes the dilemma, so the concept is family_obligation, not none. They ask "
     "'Masama ba akong anak?' (emotional) and 'Paano ko sasabihin' (practical), so both."),
    ("Birthday ko kahapon at buong araw hindi man lang ako binati ng boyfriend ko. Nag-post pa siya ng games niya "
     "sa story. Hindi ko siya nirereplyan ngayon, hinihintay ko lang kung mapapansin niya. OA ba ako?",
     {"exclude_flag": "no", "primary_emotion": "tampo", "cultural_concept": "none", "expression": "implicit",
      "domain": "romantic", "need_type": "emotional", "notes": "runner-up: sadness"},
     "Hurt at a partner could be sadness, but tampo's defining feature is in the text: withdrawal ('hindi ko siya "
     "nirereplyan') plus waiting to be noticed. So the culture-bound label wins the tie-break. The poster never "
     "says 'nagtatampo', so implicit. No cultural norm changes the right advice, so concept none."),
    ("Paano po mag-file ng small claims? May utang sa akin yung dating officemate ko na 15k, 6 months na. "
     "Ano pong requirements at magkano ang fee? Salamat po.",
     {"exclude_flag": "no", "primary_emotion": "none_practical", "cultural_concept": "none",
      "expression": "implicit", "domain": "money", "need_type": "practical", "notes": ""},
     "An unpaid loan could make someone angry, but this poster expresses no emotion: they only ask for a "
     "procedure. Label the felt state, not the situation, so none_practical. Expression is implicit because no "
     "emotion is named. A former officemate is not a close relationship, so the domain is money."),
    ("(A post where the poster says they have been thinking of ending their life.)",
     {"exclude_flag": "yes", "exclude_reason": "self_harm_crisis"},
     "Exclusion comes first. Once exclude_flag is yes, pick the reason and leave every other column empty."),
)


def _section(codebook: str, number: int) -> str:
    """The text of codebook section `## <number>.` up to the next `## ` heading."""
    match = re.search(rf"^## {number}\..*?(?=^## |\Z)", codebook, flags=re.M | re.S)
    return match.group(0) if match else ""


def codebook_emotion_cues(codebook: str) -> dict[str, str]:
    """
    'When to use / not to be confused with' per emotion id, from codebook §4.

    Culture-bound labels and konsensya have a `#### `id`` section with a defining feature and a
    "Don't confuse with" line; the other shared labels have a row in the §4 table.
    """
    labels = _section(codebook, 4)
    cues: dict[str, str] = {}
    for match in re.finditer(r"^#### `([a-z_]+)`.*?(?=^#{2,4} |\Z)", labels, flags=re.M | re.S):
        body = match.group(0)
        feature = re.search(r"\*\*Defining feature\.\*\*\s*(.+)", body)
        confuse = re.search(r"\*\*Don't confuse with\.\*\*\s*(.+)", body)
        parts = [m.group(1).strip() for m in (feature, confuse) if m]
        if parts:
            cues[match.group(1)] = _plain(" Not: ".join(parts))
    for row in re.finditer(r"^\| `([a-z_]+)` \|([^|]*)\|([^|]*)\|([^|]*)\|", labels, flags=re.M):
        cues.setdefault(row.group(1), _plain(row.group(4)))
    return cues


def codebook_emotion_examples(codebook: str) -> dict[str, str]:
    """The paraphrased example per emotion id, from the codebook §4 sections that have one."""
    examples = {}
    for match in re.finditer(r"^#### `([a-z_]+)`.*?(?=^#{2,4} |\Z)", _section(codebook, 4), flags=re.M | re.S):
        found = re.search(r"\*\*Example \(paraphrased\)\.\*\*\s*(.+)", match.group(0))
        if found:
            examples[match.group(1)] = _plain(found.group(1))
    return examples


def codebook_concept_cues(codebook: str) -> dict[str, str]:
    """Typical cues per cultural concept id, from the codebook §5 table."""
    rows = re.finditer(r"^\| `([a-z_]+)` \|([^|]*)\|([^|]*)\|([^|]*)\|", _section(codebook, 5), flags=re.M)
    return {row.group(1): _plain(row.group(4)) for row in rows if _plain(row.group(4)) not in ("", "—")}


def _plain(text: str) -> str:
    """Markdown emphasis and code marks removed, for a spreadsheet cell."""
    return re.sub(r"[*`]", "", text).strip()


def annotation_guide(schema: Schema, codebook_path: Path | None = None) -> list[list[str]]:
    """
    The `how_to_annotate` tab: the steps of codebook §1-§7 and every allowed value with its meaning.

    Definitions come from label-schema.json and the emotion cues from the codebook, so the guide
    cannot drift from either.
    """
    path = codebook_path or ROOT / schema.raw.get("codebook", "02-llm/codebook/codebook.md")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    cues, concept_cues = codebook_emotion_cues(text), codebook_concept_cues(text)
    emotion_examples = codebook_emotion_examples(text)
    cues["none_practical"] = "See the none_practical step above."
    rows: list[list[str]] = [[f"# How to annotate (label schema {schema.version})"], []]
    rows += [["# Steps"]] + [[step, "", _plain(text)] for step, text in ANNOTATION_STEPS] + [[]]

    rows.append(["# Worked examples", "", "Post (invented, not from your sheet)", "Why"])
    for number, (post, labels, why) in enumerate(WORKED_EXAMPLES, start=1):
        rows.append([f"Example {number}", "", post, why])
        rows += [["", field, value] for field, value in labels.items() if value]
        rows.append([])

    rows.append(["# exclude_reason", "", "Exclude when"])
    rows += [[e["id"], "", e["definition"]] for e in schema.raw["exclusion_reasons"]] + [[]]

    rows.append(["# primary_emotion", "type", "Definition", "Use when / not to be confused with",
                 "Example (paraphrased)"])
    for e in schema.raw["emotions"]:
        kind = {"culture_bound": "culture-bound", "shared": "shared"}.get(e.get("type", ""), e.get("type", ""))
        rows.append([e["id"], kind, e["definition"], cues.get(e["id"], ""), emotion_examples.get(e["id"], "")])
    rows.append([])

    rows.append(["# cultural_concept", "name", "Definition", "Typical cues"])
    rows += [[c["id"], c.get("name", ""), c["definition"], concept_cues.get(c["id"], "")]
             for c in schema.raw["cultural_concepts"]] + [[]]
    for key in ("expression", "domain", "need_type"):
        rows.append([f"# {key}", "", "Meaning"])
        rows += [[item["id"], "", item["definition"]] for item in schema.raw[key]] + [[]]
    return rows
