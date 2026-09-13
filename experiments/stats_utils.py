"""Shared statistics utilities for V3 experiments.

Provides non-parametric CI95 and significance tests suitable for small samples (n~14).
All functions work on paired per-sample vectors from rows[] dicts.

Usage:
    from stats_utils import bootstrap_ci95, paired_wilcoxon_test, paired_bootstrap_test
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
