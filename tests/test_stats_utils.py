"""Unit tests for stats_utils.py — W3 shared statistics utilities."""

import json
import os
import sys
import numpy as np
from pathlib import Path

# Ensure stats_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from stats_utils import bootstrap_ci95, paired_wilcoxon_test, paired_bootstrap_test


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
