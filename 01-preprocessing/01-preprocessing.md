---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    name: python3
---

# 01 — Preprocessing: Bronze → Silver

Builds `silver/advice_pairs.parquet`: one row per r/adviceph post that follows the
subreddit template, paired with its top-voted human reply.

**Stored:** post ID, flair, template fields, timestamp, top-voted reply and its score.
**Included:** template present (Problem/Goal + Context) and a substantive top reply (≥ 10 words).
Language is **not** filtered — Filipino, English, and Taglish posts are all kept.
**Cleaning:** deduplication, and PII removal (regex + NER + manual review) with placeholders.

**Stack:** `pandas` + `pyarrow` for tables (~67k rows, fits in memory). `spaCy` with the
calamanCy Tagalog model (`tl_calamancy_md`) for person-name detection, because it is trained
on Tagalog and handles code-switched text better than English-only NER models.

```python
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import spacy
from huggingface_hub import snapshot_download
```

## Configuration

All thresholds live here so the inclusion criteria in the paper match the code exactly.
The pipeline is deterministic (no sampling), but `SEED` is used for the spot-check sample.
The NER model is pinned to an exact Hugging Face commit so name detection is reproducible.

```python
# Paths resolve from the repo root so the notebook runs from any working directory via nbconvert
ROOT = Path.cwd().resolve()
ROOT = ROOT if (ROOT / "pyproject.toml").exists() else ROOT.parent
BRONZE_DIR = ROOT / "00-data" / "bronze"
DUMP_DIR = BRONZE_DIR / "reddit-adviceph_2026-11-06"
PREP_DIR = ROOT / "01-preprocessing"
SILVER_DIR = PREP_DIR / "silver"
PII_DIR = PREP_DIR / "pii"
OUT_FILE = SILVER_DIR / "advice_pairs.parquet"
ATTRITION_FILE = SILVER_DIR / "attrition.csv"
MANIFEST_FILE = SILVER_DIR / "MANIFEST.json"
NER_CACHE_FILE = PII_DIR / "ner_cache.parquet"
NAME_REVIEW_FILE = PII_DIR / "name_review.csv"

SEED = 42
MIN_REPLY_WORDS = 10
REMOVED_BODIES = {"[removed]", "[deleted]", ""}
BOT_AUTHORS = {"AutoModerator"}
TEXT_COLS = ["problem_goal", "context", "previous_attempts", "reply_body"]

# AutoModerator removal notices: "We've removed your submission", "had to remove your submission",
# "your submission was removed", "Your post has been removed". Checked against all 3,795 AutoMod
# comments: matches all 473 removal notices and none of the 3,322 standard rules notices.
AUTOMOD_REMOVAL_RE = (
    r"(?i)\bremov(?:e|ed|ing)\b[^.\n]{0,40}\b(?:submission|post)\b"
    r"|\b(?:submission|post)\b[^.\n]{0,40}\bremoved\b"
)

NER_REPO = "ljvmiranda921/tl_calamancy_md"
NER_REVISION = "4960655b2d63fd9f306d60d554a95d7bfa5dd25d"  # tl_calamancy_md 0.2.0
NAME_PLACEHOLDER = "[NAME]"
```

## Load bronze

```python
# Read the raw dump exactly as collected; bronze files are never modified
posts = pd.read_json(DUMP_DIR / "posts.jsonl", lines=True, dtype={"id": str})
comments = pd.read_json(DUMP_DIR / "comments.jsonl", lines=True, dtype={"comment_id": str, "post_id": str})
print(f"posts: {len(posts):,}  comments: {len(comments):,}")
```

### Flair

Flair was missing from the original dump, so it is collected separately by
`00-data/collect_flair.py` into its own bronze folder. If that folder doesn't exist yet,
`flair` is left null and the rest of the pipeline is unaffected.

```python
# Use the most recent flair snapshot, if one has been collected
flair_files = sorted(BRONZE_DIR.glob("reddit-adviceph-flair_*/flair.jsonl"))
if flair_files:
    flair = pd.read_json(flair_files[-1], lines=True, dtype={"id": str})[["id", "link_flair_text"]]
    flair = flair.rename(columns={"id": "post_id", "link_flair_text": "flair"})
    print(f"flair from {flair_files[-1].relative_to(ROOT)}: {len(flair):,} rows")
else:
    flair = pd.DataFrame({"post_id": pd.Series(dtype=str), "flair": pd.Series(dtype=str)})
    print("No flair snapshot found — flair will be null")
```

```python
# Attrition log: every filter appends one row, which becomes the paper's data table
attrition: list[dict] = []


def log_step(step: str, df: pd.DataFrame, reason: str) -> None:
    """Record how many posts survive a pipeline step."""
    attrition.append({"step": step, "posts": len(df), "reason": reason})
    print(f"{step:<28} {len(df):>6,}  {reason}")


log_step("bronze", posts, "raw dump")
```

## Step 1 — Drop removed and deleted posts

```python
# Removed/deleted posts have no text to parse, so they can't contribute template fields
live = posts[~posts["selftext"].fillna("").str.strip().isin(REMOVED_BODIES)].copy()
log_step("live posts", live, "selftext not [removed]/[deleted]/empty")
```

## Step 2 — Drop posts removed by AutoModerator

AutoModerator leaves a comment on nearly every post. Most of these are a standard rules
notice, which says nothing about the post and is ignored. A removal notice means the post
was taken down: wrong post format, links in the post, body under 400 characters, or a
low-karma / low-quality account. Those posts are not valid benchmark items even when their
text is still in the dump.

```python
# Find posts that AutoModerator removed, then exclude them
automod = comments[comments["author"].isin(BOT_AUTHORS)]
auto_removed = set(automod.loc[automod["body"].str.contains(AUTOMOD_REMOVAL_RE, regex=True), "post_id"])
print(f"posts with an AutoModerator removal notice: {len(auto_removed):,}")
live = live[~live["id"].isin(auto_removed)].copy()
log_step("not auto-removed", live, "no AutoModerator removal notice")
```

## Step 3 — Parse the post template

r/adviceph asks posters to use three headers: **Problem/Goal**, **Context**, and
**Previous Attempts**. Posters spell the last one many ways ("attempts", "attemps",
"attempt/s", "prev attempts", "what I've tried"), so the regex covers the observed variants.

**Design decisions**
- Headers must start a line (after optional markdown like `**`, `#`, `>`) and end with a colon
  or the end of the line. Matching mid-line would also catch prose such as "for context: ...",
  splitting fields in the wrong place.
- A field that contains only other header names (an unfilled template) counts as empty.
- A field is the text between its header and the next header. If a header appears twice,
  the first occurrence wins.
- Markdown emphasis (`*`, `_`) is stripped; other text is kept verbatim, including emoji
  and code-switching.
- **Template present = Problem/Goal and Context both non-empty.** Previous Attempts is
  optional (stored as null when absent), since many posters skip it.

```python
# One named group per canonical field; the matched group name tells us which header we hit.
# Previous Attempts may carry a short suffix before the colon ("/Question", " to Resolve").
HEADER_PATTERNS = {
    "problem_goal": r"problem\s*/\s*goals?",
    "context": r"context",
    "previous_attempts": (
        r"(?:previous|prev\.?)\s+attemp(?:t|ts|s|t/s|t\(s\))"
        r"|attempts?"
        r"|what\s+i\s*(?:'|’)?\s*(?:ve|have)\s+(?:tried|done)(?:\s+so\s+far)?"
    ),
}
HEADER_RE = re.compile(
    r"(?im)^[ \t>#*_\-]*(?:"
    + "|".join(
        f"(?P<{k}>(?:{v})" + (r"[^:\n]{0,20}?" if k == "previous_attempts" else "") + ")"
        for k, v in HEADER_PATTERNS.items()
    )
    + r")[ \t*_'’]*(?::[ \t*_]*|$)"
)
# A field holding only other header names (e.g. "context: previous attempts:") is an unfilled template
EMPTY_FIELD_RE = re.compile(r"(?i)^[\W_]*(?:(?:problem\s*/\s*goals?|context|previous\s+attempts?)[\W_]*)*$")


def parse_template(text: str) -> dict[str, str | None]:
    """
    Split a post body into its template fields.

    Args:
        text: Post selftext

    Returns:
        Dict with problem_goal, context, previous_attempts (None when missing or empty)
    """
    matches = list(HEADER_RE.finditer(text))
    fields: dict[str, str | None] = dict.fromkeys(HEADER_PATTERNS)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if fields[m.lastgroup] is None:
            value = re.sub(r"[*_]{1,3}", "", text[m.end() : end]).strip()
            fields[m.lastgroup] = None if EMPTY_FIELD_RE.match(value) else value
    return fields
```

```python
# Parse every live post, then keep those with both core template fields
parsed = pd.DataFrame(live["selftext"].map(parse_template).tolist(), index=live.index)
live = live.join(parsed)
templated = live[live["problem_goal"].notna() & live["context"].notna()].copy()
log_step("template present", templated, "Problem/Goal and Context both non-empty")
```

## Step 4 — Deduplicate posts

Some posters resubmit the same story (often after a removal). Two posts are duplicates
when their Problem/Goal and Context match after lowercasing and collapsing punctuation and
whitespace. The **earliest** post is kept, since later copies are resubmissions.

```python
# Normalize text so trivial edits (case, punctuation, spacing) don't hide a resubmission
def normalize(text: str | None) -> str:
    """Lowercase and collapse non-word characters, for duplicate detection only."""
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


templated["dedup_key"] = templated["problem_goal"].map(normalize) + "|" + templated["context"].map(normalize)
deduped = templated.sort_values("created_utc").drop_duplicates("dedup_key", keep="first").copy()
print(f"duplicate posts removed: {len(templated) - len(deduped):,}")
log_step("deduplicated", deduped, "earliest copy of identical Problem/Goal + Context kept")
```

## Step 5 — Select the top-voted reply

**Eligible replies** are top-level comments (`depth == 0`, i.e. direct answers to the post)
that are not from AutoModerator, not `[removed]`/`[deleted]`, and not written by the
original poster (OP replies are clarifications, not advice).

**Top reply** = highest `score`. Ties (common at low scores) go to the **earliest**
reply, which held the vote lead first. Scores are a snapshot from scrape time.

**Substantive** = the top reply itself has ≥ `MIN_REPLY_WORDS` whitespace tokens. We do
**not** fall back to a lower-voted reply, because the benchmark reference should be what
the community actually ranked highest.

```python
# Filter to replies that count as human advice to the original poster
op_author = posts.set_index("id")["author"]
eligible = comments[
    (comments["depth"] == 0)
    & ~comments["author"].isin(BOT_AUTHORS | {"[deleted]"})
    & ~comments["body"].fillna("").str.strip().isin(REMOVED_BODIES)
    & (comments["author"] != comments["post_id"].map(op_author))
]
print(f"eligible top-level replies: {len(eligible):,}")
```

```python
# Sort by score desc, then time asc, so the first row per post is the top reply with the tie-break applied
top = (
    eligible.sort_values(["post_id", "score", "created_utc"], ascending=[True, False, True])
    .drop_duplicates("post_id", keep="first")
    .set_index("post_id")
)
with_reply = deduped[deduped["id"].isin(top.index)].copy()
log_step("has eligible reply", with_reply, "≥ 1 human top-level reply not by OP")
```

```python
# Keep only posts whose single top reply is substantive
with_reply = with_reply.join(
    top[["comment_id", "body", "score"]].rename(
        columns={"comment_id": "reply_comment_id", "body": "reply_body", "score": "reply_score"}
    ),
    on="id",
)
substantive = with_reply[with_reply["reply_body"].str.split().str.len() >= MIN_REPLY_WORDS].copy()
log_step("substantive top reply", substantive, f"top reply ≥ {MIN_REPLY_WORDS} words")
```

## Step 6 — PII removal: regex

Structured identifiers are replaced with typed placeholders, so the text stays readable
and models can still see that, for example, a phone number was mentioned.

| Pattern | Placeholder |
|---|---|
| Email address | `[EMAIL]` |
| URL (`http(s)://`, `www.`, bare `facebook.com/...` etc.) | `[URL]` |
| Philippine mobile (`09xx…`, `+63 9xx…`) and Metro Manila landline (`(02) 8xxx xxxx`) | `[PHONE]` |
| `@handle` | `[HANDLE]` |
| Reddit `u/username` | `u/[USER]` |

All URLs are masked, not only social-media ones: a link can identify someone (a shared
Facebook post) and the benchmark doesn't need working links. Emails run before handles so
the `@` inside an address isn't treated as a handle.

```python
# Ordered so broader patterns run first and later ones never see half-replaced text
PII_PATTERNS: list[tuple[str, str]] = [
    ("[EMAIL]", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("[URL]", r"(?i)\bhttps?://\S+|\bwww\.\S+|\b(?:facebook|fb|instagram|tiktok|twitter|x)\.com/\S+"),
    ("[PHONE]", r"(?<!\d)(?:\+?63[\s-]?|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)|\(?02\)?[\s-]?8\d{3}[\s-]?\d{4}(?!\d)"),
    ("[HANDLE]", r"(?<![\w@])@[A-Za-z0-9_.]{3,30}"),
    ("u/[USER]", r"(?i)(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}"),
]


def mask_regex_pii(text: str | None) -> str | None:
    """Replace structured PII with typed placeholders."""
    if text is None or pd.isna(text):
        return None
    for placeholder, pattern in PII_PATTERNS:
        text = re.sub(pattern, placeholder, text)
    return text
```

```python
# Count hits per placeholder before masking, for the paper's PII table
pii_counts = {
    ph: int(sum(substantive[c].dropna().str.count(pat).sum() for c in TEXT_COLS)) for ph, pat in PII_PATTERNS
}
print(pii_counts)
cleaned = substantive.copy()
for c in TEXT_COLS:
    cleaned[c] = cleaned[c].map(mask_regex_pii)
```

## Step 7 — PII removal: person names (NER + cues + manual review)

Names don't follow a fixed pattern, so detection has two sources:
- **calamanCy NER**: `PER` entities from `tl_calamancy_md`.
- **Cue regex**: a capitalized word after a Filipino or English cue that usually comes
  before a name: *si, ni, kay, sina, nina, kina, Ate, Kuya, Tita, Tito, Lola, Lolo, Ma'am,
  Sir, Mr., Ms., Mrs., Dr., Atty., named, "name is", pangalan*.

**Manual pass.** Every candidate goes into `pii/name_review.csv` with its source, count,
and an example snippet. A reviewer sets `decision` to `replace` or `keep` and
`reviewed_by` to their initials. The notebook keeps existing decisions on rerun and only
adds new candidates.

**Auto-prefill for unreviewed candidates** (`reviewed_by = auto`). It is only a starting
point; the manual pass is what makes the decision.
- `keep` if it is a kinship term, honorific, religious term, or platform/brand.
- `keep` if it isn't name-shaped: emoji, digits, ALL CAPS, single initials like "A." (these
  are mostly list labels), or other NER artifacts.
- `keep` if it is a common word: its lowercase form appears ≥ 2 times in the corpus
  (*May*, *Fortunately*, *Zinc*). Single-token slang like *Anw* or *Yaan* is caught here too.
- `replace` otherwise, noting whether it came after a name cue or from NER only.

The reviewer should especially check `replace` rows for **public figures, brands, drugs,
places, and titles** (e.g. *Taylor Swift*, *Cetaphil*, *Victoria Peak*), which NER also tags
as persons but which aren't private PII.

Approved names are replaced with `[NAME]` everywhere they appear as a whole word
(case-sensitive), longest first so "Juan dela Cruz" is masked before "Juan".

NER is slow (~10+ min on CPU), so results are cached per text hash in
`pii/ner_cache.parquet`. The cache and review file hold real names, so both are
gitignored; their hashes go in the manifest.

```python
# Load the pinned calamanCy model; only the NER pipe (and its tok2vec) is needed
model_path = snapshot_download(NER_REPO, revision=NER_REVISION, ignore_patterns=["*.whl"])
nlp = spacy.load(model_path, exclude=["trainable_lemmatizer", "tagger", "morphologizer", "parser"])


def text_sha(text: str) -> str:
    """Stable cache key for a text."""
    return hashlib.sha256(text.encode()).hexdigest()
```

```python
# Run NER only on texts not already in the cache, then save the updated cache
PII_DIR.mkdir(parents=True, exist_ok=True)
texts = sorted({t for c in TEXT_COLS for t in cleaned[c].dropna()})
cache = (
    pd.read_parquet(NER_CACHE_FILE)
    if NER_CACHE_FILE.exists()
    else pd.DataFrame({"sha": pd.Series(dtype=str), "persons": pd.Series(dtype=object)})
)
known = set(cache["sha"])
todo = [t for t in texts if text_sha(t) not in known]
print(f"texts: {len(texts):,}  cached: {len(texts) - len(todo):,}  to run: {len(todo):,}")
new_rows = [
    {"sha": text_sha(t), "persons": [e.text for e in doc.ents if e.label_ == "PER"]}
    for t, doc in zip(todo, nlp.pipe(todo, batch_size=64))
]
if new_rows:
    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    cache.to_parquet(NER_CACHE_FILE, index=False)
ner_by_sha = dict(zip(cache["sha"], cache["persons"]))
```

```python
# Collect candidates from both sources with counts and one example snippet each
NAME_CUE_RE = re.compile(
    r"\b(?:[Ss]i|[Nn]i|[Kk]ay|[Ss]ina|[Nn]ina|[Kk]ina|[Aa]te|[Kk]uya|[Tt]ita|[Tt]ito|[Ll]ola|[Ll]olo"
    r"|[Mm]a'?am|[Ss]ir|M[rs]s?\.?|Dr\.?|Atty\.?|named|name is|pangalan(?: niya| nya| ko)?)"
    r"\s+([A-Z][a-z]{2,}(?:\s+(?:dela|de|del|delos|de los|[A-Z][a-z]+))*)"
)
sources: dict[str, set[str]] = {}
examples: dict[str, str] = {}
for t in texts:
    found = [(n, "ner") for n in ner_by_sha.get(text_sha(t), [])] + [(m, "cue") for m in NAME_CUE_RE.findall(t)]
    for name, src in found:
        name = name.strip()
        if len(name) < 2 or name.startswith("["):
            continue
        sources.setdefault(name, set()).add(src)
        if name not in examples:
            i = t.find(name)
            examples[name] = t[max(0, i - 40) : i + len(name) + 40].replace("\n", " ")

corpus = "\n".join(texts)
lower_tokens = Counter(re.findall(r"\b[a-z][a-z'’]+\b", corpus))
counts = {n: len(re.findall(rf"(?<!\w){re.escape(n)}(?!\w)", corpus)) for n in sources}
print(f"name candidates: {len(sources):,}")
```

```python
# Prefill a decision for each candidate; manual decisions in an existing review file always win
KEEP_WORDS = {
    # kinship and honorifics (used as forms of address, not names)
    "papa", "mama", "mom", "mommy", "dad", "daddy", "ate", "kuya", "tita", "tito", "lola", "lolo",
    "nanay", "tatay", "inay", "itay", "ma'am", "sir", "miss", "hubby", "wifey", "bunso", "atty",
    "boss", "bro", "sis", "beh", "bes", "mare", "pare", "tol",
    # religious
    "lord", "god", "jesus", "christ", "allah", "mary", "lord jesus",
    # platforms, brands, institutions
    "gcash", "maya", "reddit", "google", "facebook", "messenger", "shopee", "lazada", "tiktok",
    "chatgpt", "netflix", "spotify", "youtube", "instagram", "twitter", "jollibee", "sss",
    "pag-ibig", "philhealth", "dfa", "nbi", "dswd", "bpi", "bdo",
}


# A name is Title-case words, optionally joined by Filipino surname particles; anything else
# (emoji, digits, ALL CAPS, pipes) is an NER artifact
NAME_SHAPE_RE = re.compile(
    r"^[A-Z][a-z][A-Za-z'’\-]*(?:\s+(?:dela|de|del|delos|de los|[A-Z][a-z][A-Za-z'’\-]*))*$"
)


def prefill(name: str, source: str) -> tuple[str, str]:
    """
    Suggest a decision for a candidate that no human has reviewed yet.

    Args:
        name: Candidate string
        source: "cue", "ner", or "cue+ner"

    Returns:
        (decision, note) where decision is "replace" or "keep"
    """
    low = name.lower()
    if low in KEEP_WORDS:
        return "keep", "keep-list"
    if not NAME_SHAPE_RE.match(name):
        return "keep", "not name-shaped (NER artifact, initial, or list label)"
    if " " not in name and lower_tokens[low] >= 2:
        return "keep", f"common word ({lower_tokens[low]}x lowercase)"
    if "cue" in source:
        return "replace", "after a name cue"
    return "replace", "NER person"


review = pd.DataFrame(
    [
        {"candidate": n, "source": "+".join(sorted(sources[n])), "count": counts[n], "example": examples[n]}
        for n in sorted(sources)
    ]
)
if NAME_REVIEW_FILE.exists():
    previous = pd.read_csv(NAME_REVIEW_FILE, dtype=str, keep_default_na=False)
    previous = previous[previous["reviewed_by"] != "auto"][["candidate", "decision", "reviewed_by", "note"]]
else:
    previous = pd.DataFrame(columns=["candidate", "decision", "reviewed_by", "note"])
review = review.merge(previous, on="candidate", how="left")
auto = review["decision"].isna()
for i in review.index[auto]:
    review.loc[i, ["decision", "note"]] = prefill(review.at[i, "candidate"], review.at[i, "source"])
review.loc[auto, "reviewed_by"] = "auto"
review = review.sort_values(["decision", "count"], ascending=[False, False])
review.to_csv(NAME_REVIEW_FILE, index=False)
print(review["decision"].value_counts().to_dict(), "| human-reviewed:", int((review["reviewed_by"] != "auto").sum()))
```

```python
# Replace every approved name, longest first so multi-word names are masked before their parts
to_replace = sorted(review.loc[review["decision"] == "replace", "candidate"], key=len, reverse=True)
NAME_RE = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, to_replace)) + r")(?!\w)") if to_replace else None
name_hits = 0
for c in TEXT_COLS:
    if NAME_RE is None:
        break
    name_hits += int(cleaned[c].dropna().str.count(NAME_RE).sum())
    cleaned[c] = cleaned[c].map(lambda t: NAME_RE.sub(NAME_PLACEHOLDER, t) if isinstance(t, str) else t)
pii_counts[NAME_PLACEHOLDER] = name_hits
print(f"{NAME_PLACEHOLDER} replacements: {name_hits:,}")
```

## Step 8 — Assemble the silver table

No author information is kept: usernames stay in bronze only.

```python
# Final schema; text columns are the masked versions
silver = (
    cleaned.rename(columns={"id": "post_id"})
    .merge(flair, on="post_id", how="left")
    .assign(created_at=lambda d: pd.to_datetime(d["created_utc"], unit="s", utc=True))
    [["post_id", "flair", "problem_goal", "context", "previous_attempts",
      "created_at", "reply_comment_id", "reply_body", "reply_score"]]
    .sort_values("created_at")
    .reset_index(drop=True)
    .astype({"post_id": "string", "flair": "string", "problem_goal": "string", "context": "string",
             "previous_attempts": "string", "reply_comment_id": "string", "reply_body": "string",
             "reply_score": "int64"})
)
silver.info()
```

## Validation

```python
# Structural invariants: keys, inclusion criteria, and traceability back to bronze
by_id = comments.set_index("comment_id")
assert silver["post_id"].is_unique
assert silver["problem_goal"].str.len().gt(0).all() and silver["context"].str.len().gt(0).all()
assert (silver["reply_body"].str.split().str.len() >= MIN_REPLY_WORDS).all()
assert silver["reply_comment_id"].isin(by_id.index).all()
assert (by_id.loc[silver["reply_comment_id"], "post_id"].to_numpy() == silver["post_id"].to_numpy()).all()
assert (by_id.loc[silver["reply_comment_id"], "depth"] == 0).all()
reply_authors = by_id.loc[silver["reply_comment_id"], "author"].to_numpy()
assert not set(reply_authors) & BOT_AUTHORS
assert (reply_authors != op_author.loc[silver["post_id"]].to_numpy()).all()
assert not silver["post_id"].isin(auto_removed).any()
assert not silver["reply_body"].str.contains(
    r"(?i)i am a bot|i'm a bot|this action was performed automatically", regex=True
).any()
```

```python
# Cleaning invariants: no duplicates, and no PII pattern or approved name left in any text column
norm_keys = silver["problem_goal"].map(normalize) + "|" + silver["context"].map(normalize)
assert norm_keys.is_unique and silver["reply_body"].is_unique
for placeholder, pattern in PII_PATTERNS[:4]:
    assert not silver[TEXT_COLS].apply(lambda c: c.str.contains(pattern, regex=True)).any().any(), placeholder
assert not silver[TEXT_COLS].apply(lambda c: c.str.contains(r"(?i)(?<![\w/])/?u/(?!\[USER\])\w", regex=True)).any().any()
if NAME_RE is not None:
    assert not silver[TEXT_COLS].apply(lambda c: c.str.contains(NAME_RE)).any().any()

# Usernames from bronze must not appear (handles with _, - or digits; plain words collide with prose)
usernames = {u for u in set(posts["author"]) | set(comments["author"])
             if isinstance(u, str) and u != "[deleted]" and len(u) >= 4 and re.search(r"[_\d-]", u)}
username_re = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, sorted(usernames, key=len, reverse=True))) + r")(?![\w])")
leaks = [m.group(0) for c in TEXT_COLS for t in silver[c].dropna() for m in username_re.finditer(t)]
assert not leaks, f"username leaked into silver: {leaks[:5]}"
print("all checks passed")
```

```python
# Spot-check sample with permalinks for manual comparison against Reddit
sample = silver.sample(10, random_state=SEED).merge(posts[["id", "permalink"]], left_on="post_id", right_on="id")
for row in sample.itertuples():
    print(row.permalink)
    print(f"  PG: {row.problem_goal[:100]!r}")
    print(f"  reply ({row.reply_score}): {row.reply_body[:100]!r}\n")
```

## Save outputs

```python
# Write silver, the attrition and PII tables, and a manifest tying the outputs to their inputs and commit
SILVER_DIR.mkdir(parents=True, exist_ok=True)
silver.to_parquet(OUT_FILE, index=False)
attrition_df = pd.DataFrame(attrition)
attrition_df.to_csv(ATTRITION_FILE, index=False)
pd.DataFrame([{"placeholder": k, "replacements": v} for k, v in pii_counts.items()]).to_csv(
    SILVER_DIR / "pii_counts.csv", index=False
)


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
manifest = {
    "layer": "silver",
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "generated_by": "01-preprocessing/01-preprocessing.ipynb",
    "git_commit": commit,
    "ner_model": {"repo": NER_REPO, "revision": NER_REVISION},
    "inputs": {
        str(p.relative_to(ROOT)): sha256_file(p)
        for p in [DUMP_DIR / "posts.jsonl", DUMP_DIR / "comments.jsonl", *flair_files[-1:], NAME_REVIEW_FILE]
    },
    "files": {OUT_FILE.name: {"sha256": sha256_file(OUT_FILE), "rows": len(silver), "bytes": OUT_FILE.stat().st_size}},
}
MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
attrition_df
```

## Summary

- Final row count and per-step losses are in `silver/attrition.csv`; placeholder counts are
  in `silver/pii_counts.csv`. Both are copied into `01-preprocessing/README.md`.
- No author information reaches silver.
- **Limitations:** reply scores and flair are snapshots at collection time. The header regex
  only recognises headers at the start of a line, so posts that put every header on one line
  are excluded. Name masking depends on NER recall and the manual review; names that neither
  NER nor a cue word catches can remain, and a name that is also a common word (e.g. *May*)
  is kept unless a reviewer marks it.
