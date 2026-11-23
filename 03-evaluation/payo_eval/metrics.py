"""Classification metrics, post-level bootstrap, calibration and proportion CIs (analysis plan §3)."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from payo_llm.pilot import wilson_ci

INVALID = "(invalid)"

__all__ = ["accuracy", "macro_f1", "per_class", "confusion", "majority_baseline", "bootstrap_ci", "ece",
           "wilson_ci", "INVALID"]


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Share of exact matches; invalid predictions count as wrong."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else float("nan")


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 over the gold labels present; an invalid prediction is a miss for its gold class."""
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if labels else float("nan")


def per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support per gold label."""
    labels = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return pd.DataFrame({k: report[k] for k in labels}).T.rename_axis("label").reset_index()


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Counts with gold labels as rows and predictions (including invalid) as columns."""
    table = pd.crosstab(pd.Series(y_true, name="gold"), pd.Series(y_pred, name="pred"))
    return table


def majority_baseline(y_true: Sequence[str]) -> dict[str, float]:
    """Always predict the most frequent gold label."""
    top = pd.Series(y_true).value_counts().index[0]
    guess = [top] * len(y_true)
    return {"label": top, "accuracy": accuracy(y_true, guess), "macro_f1": macro_f1(y_true, guess)}


def bootstrap_ci(frame: pd.DataFrame, cluster: str, stat: Callable[[pd.DataFrame], float], b: int, seed: int,
                 level: float = 0.95) -> tuple[float, float, float]:
    """
    Percentile bootstrap resampling whole clusters (posts), so both models' rows for a post move together.

    Args:
        frame: The data.
        cluster: Column to resample (post_id).
        stat: Statistic computed on a resampled frame.
        b: Replicates.
        seed: Random seed.
        level: CI level.

    Returns:
        (estimate on the full data, lower, upper).
    """
    estimate = stat(frame)
    groups = {k: np.asarray(v) for k, v in frame.groupby(cluster).indices.items()}
    keys = list(groups)
    if not keys:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(b):
        drawn = rng.integers(0, len(keys), len(keys))
        index = np.concatenate([groups[keys[i]] for i in drawn])
        values.append(stat(frame.iloc[index]))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return estimate, float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    return estimate, float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


def ece(confidence: Sequence[float], correct: Sequence[int], n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """
    Expected calibration error with equal-width bins on [0, 1].

    Args:
        confidence: Stated confidence as a probability (confidence / 100).
        correct: 1 if the prediction was right.
        n_bins: Number of bins.

    Returns:
        (ECE, per-bin table: bin, n, mean_confidence, accuracy).
    """
    conf = np.clip(np.asarray(confidence, dtype=float), 0, 1)
    hit = np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows, total = [], 0.0
    for k in range(n_bins):
        mask = bins == k
        if mask.any():
            rows.append({"bin": f"{k / n_bins:.1f}-{(k + 1) / n_bins:.1f}", "n": int(mask.sum()),
                         "mean_confidence": conf[mask].mean(), "accuracy": hit[mask].mean()})
            total += mask.sum() / len(conf) * abs(conf[mask].mean() - hit[mask].mean())
    return float(total), pd.DataFrame(rows)
