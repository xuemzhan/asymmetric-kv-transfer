"""Shared statistics utilities for V3 experiments.

Provides non-parametric CI95 and significance tests suitable for small samples (n~14).
All functions work on paired per-sample vectors from rows[] dicts.

Usage:
    from stats_utils import bootstrap_ci95, paired_wilcoxon_test, paired_bootstrap_test
    from stats_utils import rankdata_average, spearman, bootstrap_spearman_ci
"""

import numpy as np
from scipy import stats


def bootstrap_ci95(values, n_boot=10000, seed=0):
    """Paired bootstrap CI95 for mean.

    Args:
        values: 1-D array of per-sample metric values.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high) — point estimate and 2.5/97.5 percentiles.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.array([rng.choice(values, n, replace=True).mean() for _ in range(n_boot)])
    return (
        float(values.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def paired_wilcoxon_test(a, b):
    """Wilcoxon signed-rank test for paired samples.

    Args:
        a: 1-D array of values from condition A (e.g. teacher).
        b: 1-D array of values from condition B (e.g. student baseline).

    Returns:
        (statistic, p_value, cohens_d) — two-sided test.
        cohens_d uses mean(diff)/std(diff, ddof=1).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    std_d = float(diff.std(ddof=1))
    cohens_d = float(diff.mean() / std_d) if std_d > 0 else 0.0
    # If all differences are zero, Wilcoxon is undefined — return no-signal result
    if std_d == 0 or np.all(diff == 0):
        return 0.0, 1.0, 0.0
    stat, p = stats.wilcoxon(diff)
    return float(stat), float(p), cohens_d


def paired_bootstrap_test(a, b, n_boot=10000, seed=0):
    """Bootstrap test for paired difference.

    Args:
        a: 1-D array of values from condition A.
        b: 1-D array of values from condition B.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        (mean_delta, ci_low, ci_high, p_value).
        p_value is the two-sided proportion of the bootstrap distribution
        that crosses zero (2 * min(p_positive, p_negative)).
    """
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    boots = np.array([diff[rng.choice(n, n, replace=True)].mean() for _ in range(n_boot)])
    mean_delta = float(diff.mean())
    ci_low = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))
    p_value = float(2 * min((boots >= 0).mean(), (boots <= 0).mean()))
    return mean_delta, ci_low, ci_high, p_value

def rankdata_average(values):
    """1-based average ranks (ties share the mean of the ranks they span).

    This is the tie policy that ``scipy.stats.rankdata(method="average")`` and
    ``scipy.stats.spearmanr`` use. Ordinal ranks (``np.argsort(np.argsort(x))``)
    are *not* equivalent: they break ties by position in the input, so the same
    data in a different order yields a different correlation. The V3 reports
    used ordinal ranks until W31; see `paper/audit/METRIC_CORRECTION.md` section
    14 for the affected numbers.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    ranks = np.empty(n, dtype=float)
    if n == 0:
        return ranks
    order = np.argsort(x, kind="stable")
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j + 2)
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation of ``x`` and ``y`` using average ranks.

    Returns ``None`` when there are fewer than three paired points or when
    either vector is constant, because the correlation is then undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = rankdata_average(x), rankdata_average(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman_ci(err, em, n_boot=10000, seed=0):
    """Percentile CI95 for ``spearman(err, em)`` under a point-level bootstrap.

    Resamples the (error, EM) pairs with replacement — the "configuration-level"
    bootstrap used for the mapper-objective sweep — and skips degenerate
    resamples. Returns ``[None, None]`` if no resample is usable.
    """
    err = np.asarray(err, dtype=float)
    em = np.asarray(em, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(err)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(err[idx])) < 2 or len(np.unique(em[idx])) < 2:
            continue
        r = spearman(err[idx], em[idx])
        if r is not None:
            vals.append(r)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
