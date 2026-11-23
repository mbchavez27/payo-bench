"""Inter-annotator and inter-rater agreement (analysis plan §9)."""

import math
from collections.abc import Sequence

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Cohen's κ between two parallel label lists.

    Returns 1.0 when both lists use one identical label throughout (perfect agreement, no variance),
    and NaN for empty input.
    """
    a, b = list(a), list(b)
    if not a:
        return math.nan
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(cohen_kappa_score(a, b))


def one_vs_rest_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    """
    κ for each label as a binary "this label vs. anything else" decision (the §2 gate).

    Args:
        a, b: Parallel labels from the first and second annotator.
        labels: Labels to report (labels used by neither annotator get NaN).

    Returns:
        label, n_a, n_b, kappa.
    """
    rows = []
    for label in labels:
        x = [v == label for v in a]
        y = [v == label for v in b]
        used = any(x) or any(y)
        rows.append({"label": label, "n_a": sum(x), "n_b": sum(y),
                     "kappa": cohen_kappa(x, y) if used else math.nan})
    return pd.DataFrame(rows)


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Share of items where the two labels are identical."""
    a, b = list(a), list(b)
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)])) if a else math.nan


def ordinal_alpha(frame: pd.DataFrame, item: str, rater: str, value: str) -> float:
    """
    Krippendorff's α (ordinal) from long data; items rated by one rater only add nothing.

    Args:
        frame: One row per (item, rater) rating.
        item, rater, value: Column names.

    Returns:
        α, or NaN when fewer than two items have two ratings.
    """
    wide = frame.pivot_table(index=rater, columns=item, values=value, aggfunc="first")
    wide = wide.loc[:, wide.notna().sum() >= 2]
    if wide.shape[1] < 2:
        return math.nan
    data = wide.to_numpy(dtype=float)
    if len(np.unique(data[~np.isnan(data)])) < 2:
        return 1.0
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"))


def pairs(frame: pd.DataFrame, key: str, slot: str, value: str) -> tuple[list[str], list[str]]:
    """
    Turn long double-coded rows into two parallel lists (first slot vs. second slot).

    With more than two people in the team, pairs differ across items; pooling "first vs. second"
    is the usual approximation and is noted in the methods.

    Args:
        frame: Rows with an item key, a slot (1 or 2) and a value.
        key, slot, value: Column names.

    Returns:
        (first values, second values) for items that have both slots.
    """
    wide = frame.pivot_table(index=key, columns=slot, values=value, aggfunc="first").dropna()
    if wide.shape[1] < 2:
        return [], []
    return wide.iloc[:, 0].astype(str).tolist(), wide.iloc[:, 1].astype(str).tolist()
