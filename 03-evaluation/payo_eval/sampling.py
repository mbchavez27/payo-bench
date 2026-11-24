"""
Seeded draws: the pre-screen pool, the stratified annotation sample, the rated set, and who gets what.

Every function takes an explicit seed, so each draw is reproducible from config.toml alone.
"""

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def allocate(available: Mapping[str, int], n: int, floor: int = 0) -> dict[str, int]:
    """
    Split n draws across strata: a floor per stratum first, then proportional (largest remainder).

    No stratum gets more than it has; any shortfall is passed to strata with room left.

    Args:
        available: Posts available per stratum.
        n: Total to draw (capped at the total available).
        floor: Minimum per stratum (capped at what the stratum has).

    Returns:
        Draws per stratum, summing to min(n, total available).
    """
    n = min(n, sum(available.values()))
    keys = sorted(available)
    take = {k: min(floor, available[k]) for k in keys}
    if sum(take.values()) > n:  # floors alone exceed n: fall back to pure proportional
        take = dict.fromkeys(keys, 0)
    while sum(take.values()) < n:
        room = {k: available[k] - take[k] for k in keys if available[k] > take[k]}
        remaining = n - sum(take.values())
        weight = sum(available[k] for k in room)
        quotas = {k: remaining * available[k] / weight for k in room}
        whole = {k: min(math.floor(q), room[k]) for k, q in quotas.items()}
        if sum(whole.values()) == 0:  # hand out single draws by largest remainder
            for k in sorted(room, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), k))[:remaining]:
                whole[k] = 1
        for k, v in whole.items():
            take[k] += v
    return take


def draw_prescreen_pool(silver_ids: Sequence[str], exclude: set[str], n: int, seed: int) -> list[str]:
    """
    Simple random draw of the posts to pre-screen.

    Args:
        silver_ids: Every silver post id.
        exclude: Ids that can never be sampled (the pilot posts).
        n: Pool size (capped at what is available).
        seed: Random seed.

    Returns:
        Drawn ids in a random order (the order the screener works in).
    """
    pool = sorted(set(silver_ids) - exclude)
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.permutation(len(pool))[: min(n, len(pool))]]


RELATIONSHIP_DOMAINS = ("family", "romantic", "friendship", "work_school")  # tie order = codebook §7 priority


def _pattern(words: Sequence[str]) -> re.Pattern:
    """Whole-word, case-insensitive match for any of the words or phrases (spaces match any whitespace)."""
    alternatives = sorted({re.escape(w.lower()).replace(r"\ ", r"\s+") for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])", re.IGNORECASE)


def keyword_prescreen(silver: pd.DataFrame, exclude_ids: set[str], keywords: Mapping) -> pd.DataFrame:
    """
    Automatic pre-screen of every silver post with fixed keyword lists (code only, no LLM).

    It returns the same columns as the screener's sheet, so the stratified draw and the pre-screen bias
    analysis work unchanged.
    - `exclude_flag` = yes when a crisis term matches (the annotators apply the full §2 screen again).
    - `concept_likely` = yes when any concept term matches.
    - `domain_guess` follows codebook §7: the relationship domain with the most matches (ties in
      RELATIONSHIP_DOMAINS order), else `money` if money terms match, else `other`.

    Args:
        silver: Silver posts (post_id, problem_goal, context, previous_attempts).
        exclude_ids: Ids never sampled (the pilot posts).
        keywords: `[prescreen.keywords]` from config.toml: concept, exclude, and domain.<id> lists.

    Returns:
        post_id, exclude_flag, exclude_reason, concept_likely, domain_guess, notes (the matched terms).
    """
    concept, crisis = _pattern(keywords["concept"]), _pattern(keywords["exclude"])
    domains = {d: _pattern(words) for d, words in keywords["domain"].items()}
    rows = []
    for _, post in silver.sort_values("post_id").iterrows():
        if post["post_id"] in exclude_ids:
            continue
        text = " ".join(str(post[c]) for c in ("problem_goal", "context", "previous_attempts")
                        if isinstance(post[c], str))
        hits = {d: len(p.findall(text)) for d, p in domains.items()}
        relation = [d for d in RELATIONSHIP_DOMAINS if hits.get(d, 0) > 0]
        if relation:
            domain = max(relation, key=lambda d: (hits[d], -RELATIONSHIP_DOMAINS.index(d)))
        else:
            domain = "money" if hits.get("money", 0) else "other"
        concept_terms = sorted({m.lower() for m in concept.findall(text)})
        crisis_terms = sorted({m.lower() for m in crisis.findall(text)})
        rows.append({"post_id": post["post_id"], "exclude_flag": "yes" if crisis_terms else "no",
                     "exclude_reason": "self_harm_crisis" if crisis_terms else "",
                     "concept_likely": "yes" if concept_terms else "no", "domain_guess": domain,
                     "notes": "; ".join(concept_terms + [f"crisis:{t}" for t in crisis_terms])})
    return pd.DataFrame(rows, columns=["post_id", "exclude_flag", "exclude_reason", "concept_likely",
                                       "domain_guess", "notes"])


def stratified_sample(prescreen: pd.DataFrame, n_draw: int, concept_share: float, domain_floor: int,
                      seed: int) -> pd.DataFrame:
    """
    Draw the annotation sample from the pre-screened pool (codebook §0, analysis plan §2).

    Two arms by `concept_likely` (yes = culture-likely, no = low-culture control), then proportional by
    `domain_guess` within each arm with a per-domain floor. If an arm is short, the other arm fills in.

    Args:
        prescreen: Pre-screen rows (post_id, exclude_flag, concept_likely, domain_guess).
        n_draw: Posts to draw.
        concept_share: Target share of the concept_likely = yes arm.
        domain_floor: Minimum posts per domain within an arm.
        seed: Random seed.

    Returns:
        post_id, arm, domain_guess, stratum, in random order.
    """
    pool = prescreen[prescreen["exclude_flag"].str.lower() != "yes"].copy()
    pool["arm"] = np.where(pool["concept_likely"].str.lower() == "yes", "culture", "control")
    sizes = pool["arm"].value_counts().to_dict()
    n_draw = min(n_draw, len(pool))
    want_culture = round(n_draw * concept_share)
    culture = min(want_culture, sizes.get("culture", 0))
    control = min(n_draw - culture, sizes.get("control", 0))
    culture = min(n_draw - control, sizes.get("culture", 0))
    rng = np.random.default_rng(seed)

    parts = []
    for arm, n_arm in (("culture", culture), ("control", control)):
        rows = pool[pool["arm"] == arm]
        counts = rows["domain_guess"].value_counts().to_dict()
        for domain, k in allocate(counts, n_arm, domain_floor).items():
            cell = rows[rows["domain_guess"] == domain].sort_values("post_id")
            parts.append(cell.iloc[rng.permutation(len(cell))[:k]])
    sample = pd.concat(parts) if parts else pool.iloc[0:0]
    sample = sample.assign(stratum=sample["arm"] + "/" + sample["domain_guess"])
    order = rng.permutation(len(sample))
    return sample.iloc[order][["post_id", "arm", "domain_guess", "stratum"]].reset_index(drop=True)


def assign(post_ids: Sequence[str], people: Sequence[str], double_share: float, seed: int,
           avoid: Mapping[str, set[str]] | None = None, not_on_doubles: set[str] | None = None) -> pd.DataFrame:
    """
    Give every post one person, and a random `double_share` of posts a second, different person.

    Loads are balanced greedily. `avoid` lists people who should not get a post (e.g. a rater who
    annotated it); it is respected when anyone else can take the post, and otherwise recorded.

    Args:
        post_ids: Posts to assign.
        people: Annotator or rater ids.
        double_share: Share of posts that get a second person (0 for none).
        seed: Random seed.
        avoid: post_id -> people to avoid for that post.
        not_on_doubles: People never given a double-coded post (the fixed adjudicator, so they are
            never one of a disagreeing pair). They still get single-coded posts.

    Returns:
        post_id, person, slot (1 = primary, 2 = second), conflict (avoid rule could not be met).
    """
    avoid = avoid or {}
    not_on_doubles = not_on_doubles or set()
    if double_share > 0 and len(set(people) - not_on_doubles) < 2:
        raise SystemExit("Double coding needs at least two people besides the adjudicator.")
    rng = np.random.default_rng(seed)
    posts = [post_ids[i] for i in rng.permutation(len(post_ids))]
    n_double = round(len(posts) * double_share)
    doubles = set(posts[:n_double])
    tie_order = {p: i for i, p in enumerate(people[i] for i in rng.permutation(len(people)))}
    load = dict.fromkeys(people, 0)
    rows = []
    for post in posts:
        taken: list[str] = []
        for slot in (1, 2) if post in doubles else (1,):
            free = [p for p in people if p not in taken and not (post in doubles and p in not_on_doubles)]
            allowed = [p for p in free if p not in avoid.get(post, set())] or free
            person = min(allowed, key=lambda p: (load[p], tie_order[p]))
            load[person] += 1
            taken.append(person)
            rows.append({"post_id": post, "person": person, "slot": slot,
                         "conflict": person in avoid.get(post, set())})
    return pd.DataFrame(rows)


def rated_set(labels: pd.DataFrame, n_eval: int, seed: int) -> pd.DataFrame:
    """
    Draw the rated set, stratified by emotion_type x domain (analysis plan §1.1).

    Args:
        labels: Kept adjudicated posts with post_id, emotion_type, domain.
        n_eval: Posts to draw.
        seed: Random seed.

    Returns:
        The drawn rows, with a `stratum` column.
    """
    frame = labels.assign(stratum=labels["emotion_type"] + "/" + labels["domain"])
    counts = frame["stratum"].value_counts().to_dict()
    rng = np.random.default_rng(seed)
    parts = []
    for stratum, k in allocate(counts, n_eval).items():
        cell = frame[frame["stratum"] == stratum].sort_values("post_id")
        parts.append(cell.iloc[rng.permutation(len(cell))[:k]])
    return pd.concat(parts).sort_values("post_id").reset_index(drop=True)


def rating_candidates(sample: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Draw the rated-set candidates before annotation (analysis plan v1.4), so P-Oracle can be pre-generated.

    Stratified by the pre-screen stratum (arm x domain_guess), proportional with largest remainder. The
    returned order is random and fixed: the rated set is later the first `n_eval` candidates that are kept
    after annotation.

    Args:
        sample: sample_posts.csv (post_id, stratum).
        n: Candidates to draw (n_eval / expected retention, rounded up).
        seed: Random seed.

    Returns:
        post_id, stratum, candidate_rank (1 = first in line).
    """
    rng = np.random.default_rng(seed)
    parts = []
    for stratum, k in allocate(sample["stratum"].value_counts().to_dict(), n).items():
        cell = sample[sample["stratum"] == stratum].sort_values("post_id")
        parts.append(cell.iloc[rng.permutation(len(cell))[:k]])
    drawn = pd.concat(parts).sort_values("post_id")
    drawn = drawn.iloc[rng.permutation(len(drawn))][["post_id", "stratum"]].reset_index(drop=True)
    return drawn.assign(candidate_rank=range(1, len(drawn) + 1))


def n_to_draw(n_target: int, expected_retention: float) -> int:
    """Posts to draw so that about n_target survive the annotation exclusions."""
    return math.ceil(n_target / expected_retention)
