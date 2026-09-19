"""Unit tests for stats_utils.py — W3 shared statistics utilities."""

import json
import os
import sys
import numpy as np
from pathlib import Path

# Ensure stats_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from scipy import stats
from stats_utils import (
    bootstrap_ci95,
    paired_wilcoxon_test,
    paired_bootstrap_test,
    rankdata_average,
    spearman,
    bootstrap_spearman_ci,
)


def test_bootstrap_ci95_normal():
    """Test bootstrap_ci95 on known normal(5, 1) — CI should contain 5."""
    rng = np.random.RandomState(42)
    values = rng.normal(loc=5.0, scale=1.0, size=1000)
    mean, ci_low, ci_high = bootstrap_ci95(values, n_boot=10000, seed=0)

    # Mean should be close to 5
    assert abs(mean - 5.0) < 0.2, f"Mean {mean} too far from 5.0"
    # CI should contain 5
    assert ci_low < 5.0 < ci_high, f"CI [{ci_low}, {ci_high}] does not contain 5.0"
    # CI width should be reasonable (~±0.12 for n=1000)
    assert (ci_high - ci_low) < 1.0, f"CI width {ci_high - ci_low} too large"

    return {
        "test": "bootstrap_ci95_normal",
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "status": "PASS",
    }


def test_bootstrap_ci95_small_sample():
    """Test bootstrap_ci95 on small sample (n=14) — mimics real V3 eval size."""
    rng = np.random.RandomState(7)
    values = rng.normal(loc=0.8, scale=0.05, size=14)
    mean, ci_low, ci_high = bootstrap_ci95(values, n_boot=10000, seed=0)

    # Should contain population mean 0.8
    assert ci_low < 0.8 < ci_high, f"CI [{ci_low}, {ci_high}] does not contain 0.8"
    # CI should be wider than for n=1000 (realistic: 0.03–0.10 width)
    width = ci_high - ci_low
    assert 0.01 < width < 0.20, f"CI width {width} outside expected range"

    return {
        "test": "bootstrap_ci95_small_sample",
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "width": width,
        "status": "PASS",
    }


def test_paired_wilcoxon_test_detects_difference():
    """Wilcoxon test on known paired data where a > b → p < 0.05."""
    rng = np.random.RandomState(42)
    n = 20
    a = rng.normal(loc=1.0, scale=0.3, size=n)
    b = rng.normal(loc=0.7, scale=0.3, size=n)

    stat, p, cohens_d = paired_wilcoxon_test(a, b)

    # a > b → positive Cohen's d
    assert cohens_d > 0, f"Cohen's d {cohens_d} should be positive (a > b)"
    # With n=20 and clear separation, p should be < 0.05
    assert p < 0.05, f"p-value {p} should be < 0.05"

    return {
        "test": "paired_wilcoxon_detects_difference",
        "statistic": stat,
        "p_value": p,
        "cohens_d": cohens_d,
        "status": "PASS",
    }


def test_paired_wilcoxon_test_no_difference():
    """Wilcoxon test on identical data → p should be ≈1.0 (or high)."""
    rng = np.random.RandomState(42)
    n = 14
    vals = rng.normal(loc=0.5, scale=0.1, size=n)

    stat, p, cohens_d = paired_wilcoxon_test(vals, vals)

    # Identical → Cohen's d = 0
    assert abs(cohens_d) < 1e-10, f"Cohen's d {cohens_d} should be 0 for identical data"
    # p should be high (no significant difference)
    assert p > 0.1, f"p-value {p} should be > 0.1 for identical data"

    return {
        "test": "paired_wilcoxon_no_difference",
        "statistic": stat,
        "p_value": p,
        "cohens_d": cohens_d,
        "status": "PASS",
    }


def test_paired_bootstrap_test_detects_difference():
    """Paired bootstrap test on data where a > b → CI should exclude 0."""
    rng = np.random.RandomState(42)
    n = 14
    a = rng.normal(loc=0.85, scale=0.05, size=n)
    b = rng.normal(loc=0.75, scale=0.05, size=n)

    mean_delta, ci_low, ci_high, p_value = paired_bootstrap_test(a, b, n_boot=10000, seed=0)

    # mean_delta should be positive (a > b)
    assert mean_delta > 0, f"mean_delta {mean_delta} should be positive"
    # CI should exclude 0 (both bounds positive)
    assert ci_low > 0, f"CI low {ci_low} should be > 0 (a > b clearly)"
    # p should be small
    assert p_value < 0.05, f"p_value {p_value} should be < 0.05"

    return {
        "test": "paired_bootstrap_detects_difference",
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "status": "PASS",
    }


def test_paired_bootstrap_test_no_difference():
    """Paired bootstrap test on identical data → CI should include 0."""
    rng = np.random.RandomState(42)
    n = 14
    vals = rng.normal(loc=0.5, scale=0.1, size=n)

    mean_delta, ci_low, ci_high, p_value = paired_bootstrap_test(vals, vals, n_boot=10000, seed=0)

    # mean_delta should be ~0
    assert abs(mean_delta) < 1e-10, f"mean_delta {mean_delta} should be ~0"
    # CI should include 0
    assert ci_low <= 0 <= ci_high, f"CI [{ci_low}, {ci_high}] should include 0"
    # p should be high
    assert p_value > 0.1, f"p_value {p_value} should be > 0.1 for identical data"

    return {
        "test": "paired_bootstrap_no_difference",
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "status": "PASS",
    }


def test_rankdata_average_ties():
    """Test rankdata_average: 1-based average ranks (scipy "average" tie policy)."""
    # Hand-checked tie case: the two 20s span ranks 2 and 3 → 2.5 each.
    hand = rankdata_average([10, 20, 20, 30])
    assert np.allclose(hand, [1.0, 2.5, 2.5, 4.0]), f"rankdata_average([10,20,20,30]) = {hand}"

    # Random vector with deliberate ties → must match scipy exactly.
    rng = np.random.RandomState(3)
    x = rng.randint(0, 5, size=50).astype(float)
    ours = rankdata_average(x)
    ref = stats.rankdata(x, method="average")
    assert np.allclose(ours, ref), "rankdata_average disagrees with scipy.stats.rankdata(method='average')"

    return {
        "test": "rankdata_average_ties",
        "hand_case": [float(v) for v in hand],
        "n_ties_in_random": int(len(x) - len(np.unique(x))),
        "max_abs_diff_vs_scipy": float(np.max(np.abs(ours - ref))),
        "status": "PASS",
    }


def test_spearman_matches_scipy_with_ties():
    """Test spearman: average-rank rho matches scipy.spearmanr even with heavy ties.

    Documents the W31 correction: until W31 the reports used ordinal ranks
    (``np.argsort(np.argsort(x))``), which break ties by input position, so the
    same data reordered gave a different correlation. The two values must differ
    here by more than round-off — that is the bug the average-rank helper fixes.
    """
    rng = np.random.RandomState(5)
    x = rng.randint(0, 5, size=40).astype(float)          # many ties
    y = rng.randint(0, 5, size=40).astype(float) / 4.0    # proportions on a small grid

    rho_average = spearman(x, y)
    rho_scipy = float(stats.spearmanr(x, y).statistic)
    rho_ordinal = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])

    assert rho_average is not None, "spearman returned None for usable data"
    assert abs(rho_average - rho_scipy) < 1e-12, (
        f"average-rank spearman {rho_average} != scipy {rho_scipy}"
    )
    assert abs(rho_average - rho_ordinal) > 1e-9, (
        f"ordinal ranks {rho_ordinal} unexpectedly match average ranks {rho_average}; "
        "the tie-handling difference is not exercised"
    )

    return {
        "test": "spearman_matches_scipy_with_ties",
        "rho_average": rho_average,
        "rho_ordinal": rho_ordinal,
        "rho_scipy": rho_scipy,
        "abs_diff_average_vs_scipy": abs(rho_average - rho_scipy),
        "abs_diff_average_vs_ordinal": abs(rho_average - rho_ordinal),
        "status": "PASS",
    }


def test_spearman_degenerate_returns_none():
    """Test spearman: None for constant input, len<3, and length mismatch."""
    constant = spearman([1, 2, 3], [5, 5, 5])
    assert constant is None, f"constant vector should give None, got {constant}"

    too_short = spearman([1, 2], [3, 4])
    assert too_short is None, f"len<3 should give None, got {too_short}"

    mismatched = spearman([1, 2, 3], [1, 2])
    assert mismatched is None, f"length mismatch should give None, got {mismatched}"

    return {
        "test": "spearman_degenerate_returns_none",
        "constant_vector": constant,
        "too_short": too_short,
        "length_mismatch": mismatched,
        "status": "PASS",
    }


def test_bootstrap_spearman_ci_deterministic_and_contains_estimate():
    """Test bootstrap_spearman_ci: percentile CI brackets the estimate and is seed-stable."""
    rng = np.random.RandomState(11)
    x = rng.normal(size=30)
    y = x + rng.normal(scale=0.5, size=30)

    lo, hi = bootstrap_spearman_ci(x, y, n_boot=2000, seed=0)
    rho = spearman(x, y)

    assert lo is not None and hi is not None, f"CI degenerate: [{lo}, {hi}]"
    assert lo < rho < hi, f"CI [{lo}, {hi}] does not bracket spearman {rho}"
    assert -1.0 <= lo <= hi <= 1.0, f"CI [{lo}, {hi}] outside [-1, 1]"

    # Determinism: same seed → bitwise identical bounds.
    lo2, hi2 = bootstrap_spearman_ci(x, y, n_boot=2000, seed=0)
    assert (lo, hi) == (lo2, hi2), f"seed=0 not reproducible: [{lo}, {hi}] vs [{lo2}, {hi2}]"

    lo3, hi3 = bootstrap_spearman_ci(x, y, n_boot=2000, seed=1)

    return {
        "test": "bootstrap_spearman_ci_deterministic_and_contains_estimate",
        "rho": rho,
        "ci_low": lo,
        "ci_high": hi,
        "ci_low_seed1": lo3,
        "ci_high_seed1": hi3,
        "status": "PASS",
    }


def main():
    """Run all tests and collect results."""
    results = []
    all_pass = True

    for test_fn in [
        test_bootstrap_ci95_normal,
        test_bootstrap_ci95_small_sample,
        test_paired_wilcoxon_test_detects_difference,
        test_paired_wilcoxon_test_no_difference,
        test_paired_bootstrap_test_detects_difference,
        test_paired_bootstrap_test_no_difference,
        test_rankdata_average_ties,
        test_spearman_matches_scipy_with_ties,
        test_spearman_degenerate_returns_none,
        test_bootstrap_spearman_ci_deterministic_and_contains_estimate,
    ]:
        try:
            result = test_fn()
            results.append(result)
            print(f"  ✓ {result['test']}")
        except AssertionError as e:
            results.append({"test": test_fn.__name__, "status": "FAIL", "error": str(e)})
            print(f"  ✗ {test_fn.__name__}: {e}")
            all_pass = False
        except Exception as e:
            results.append({"test": test_fn.__name__, "status": "ERROR", "error": str(e)})
            print(f"  ✗ {test_fn.__name__}: {e}")
            all_pass = False

    summary = {
        "module": "stats_utils.py",
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] != "PASS"),
        "all_pass": all_pass,
        "results": results,
    }

    # Save to reports
    report_path = Path(__file__).resolve().parent.parent / "reports" / "w3_stats_utils_test.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {report_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
