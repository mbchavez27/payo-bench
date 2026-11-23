"""
Statistical models and tests (analysis plan §3-§6, §10, §12).

Predictors are built as explicit 0/1 columns rather than formula factors, so every contrast in the plan
is a named linear combination of coefficients that can be checked by eye.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

Z95 = stats.norm.ppf(0.975)


@dataclass(frozen=True)
class Contrast:
    """A linear combination of coefficients with its Wald CI and p-values."""

    name: str
    estimate: float
    se: float
    lower: float
    upper: float
    p_two_sided: float
    p_greater: float  # H: estimate > 0
    p_less: float     # H: estimate < 0

    def as_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


def contrast(params: pd.Series, cov: pd.DataFrame, weights: Mapping[str, float], name: str) -> Contrast:
    """
    Estimate sum(w_k * beta_k) with a Wald (normal) CI.

    Args:
        params: Coefficients.
        cov: Their covariance matrix.
        weights: Coefficient name -> weight (names must exist in params).
        name: Label for the result.

    Returns:
        The contrast.
    """
    unknown = set(weights) - set(params.index)
    if unknown:
        raise KeyError(f"Unknown coefficients in contrast {name!r}: {sorted(unknown)}")
    w = pd.Series(0.0, index=params.index)
    for k, v in weights.items():
        w[k] = v
    est = float(w @ params)
    se = float(np.sqrt(w @ cov.loc[params.index, params.index] @ w))
    z = est / se if se > 0 else np.nan
    return Contrast(name, est, se, est - Z95 * se, est + Z95 * se, float(2 * stats.norm.sf(abs(z))),
                    float(stats.norm.sf(z)), float(stats.norm.cdf(z)))


def dummies(frame: pd.DataFrame, column: str, reference: str, prefix: str | None = None,
            levels: Sequence[str] | None = None) -> pd.DataFrame:
    """0/1 columns `<prefix>_<level>` for every level except the reference."""
    prefix = prefix or column
    levels = levels or sorted(set(frame[column]) - {reference})
    return pd.DataFrame({f"{prefix}_{lvl}": (frame[column] == lvl).astype(float) for lvl in levels if lvl != reference},
                        index=frame.index)


def gee_logit(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    """Binomial GEE, logit link, exchangeable correlation, clustered by `groups` (§3)."""
    model = sm.GEE(y.astype(float), sm.add_constant(x, has_constant="add"), groups=groups,
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    return model.fit()


def mixed(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    """
    Linear mixed model with a random intercept per group (post), fixed effects in `x` (§4.1).

    Returns:
        (fixed-effect params, their covariance, the fit) — convergence warnings are kept on the fit.
    """
    exog = sm.add_constant(x, has_constant="add")
    exog = exog.loc[:, (exog != 0).any()]  # a dummy with no rows would make the design singular
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.MixedLM(y.astype(float), exog, groups=groups).fit(reml=True, method=["lbfgs", "powell"])
    fit.payo_warnings = [str(w.message) for w in caught]
    fe = fit.fe_params
    return fe, fit.cov_params().loc[fe.index, fe.index], fit


def mcnemar(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """
    Exact McNemar on paired 0/1 outcomes, with the discordant-pair odds ratio.

    The OR is b_only / a_only, so OR > 1 favours b (e.g. a = Gemma, b = SEA-LION).

    Returns:
        n, a_only (a=1, b=0), b_only (a=0, b=1), odds_ratio = b_only / a_only, p.
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    both, neither = int(((a == 1) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
    result = sm_mcnemar([[both, a_only], [b_only, neither]], exact=True)
    odds = b_only / a_only if a_only else np.inf if b_only else np.nan
    return {"n": len(a), "a_only": a_only, "b_only": b_only, "odds_ratio": odds, "p": float(result.pvalue)}


def wilcoxon(diff: Sequence[float], alternative: str = "two-sided") -> dict[str, float]:
    """
    Wilcoxon signed-rank on paired differences, with the matched-pairs rank-biserial r.

    Zero differences are dropped (Wilcox method); r = (sum of positive ranks - sum of negative) / total.
    """
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    nonzero = d[d != 0]
    out = {"n": len(d), "n_nonzero": len(nonzero), "median_diff": float(np.median(d)) if len(d) else np.nan,
           "mean_diff": float(d.mean()) if len(d) else np.nan, "W": np.nan, "p": np.nan, "r_rank_biserial": np.nan}
    if len(nonzero) == 0:
        return out
    ranks = stats.rankdata(np.abs(nonzero))
    pos, neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    out["r_rank_biserial"] = float((pos - neg) / (pos + neg))
    res = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
    out["W"], out["p"] = float(res.statistic), float(res.pvalue)
    return out


def tost_paired(x: Sequence[float], y: Sequence[float], margin: float) -> dict[str, float]:
    """
    Paired equivalence test (TOST) of mean(x - y) within ±margin, with the matching 90% CI.

    Returns:
        n, diff, ci90_low, ci90_high, p (the larger of the two one-sided p-values).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    n = len(d)
    if n < 2 or np.all(d == d[0]):
        diff = float(d.mean()) if n else np.nan
        inside = n >= 2 and -margin < diff < margin
        return {"n": n, "diff": diff, "ci90_low": diff, "ci90_high": diff, "p": 0.0 if inside else 1.0}
    p, _, _ = ttost_paired(x, y, -margin, margin)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.95, n - 1)
    return {"n": n, "diff": float(d.mean()), "ci90_low": float(d.mean() - t * se),
            "ci90_high": float(d.mean() + t * se), "p": float(p)}


def holm(frame: pd.DataFrame, p: str = "p", family: str = "family") -> pd.DataFrame:
    """Holm-adjusted p-values within each family (§10); rows without a p-value are left as NaN."""
    out = frame.reset_index(drop=True)
    out["p_holm"] = np.nan
    for _, rows in out.groupby(family):
        valid = rows[p].notna()
        if valid.any():
            out.loc[rows.index[valid], "p_holm"] = multipletests(rows.loc[valid, p], method="holm")[1]
    return out
