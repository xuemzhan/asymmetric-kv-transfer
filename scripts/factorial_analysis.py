#!/usr/bin/env python3
"""
factorial_analysis.py — 2x2 factorial (K x V) analysis for the four-arm
KV transfer protocol, plus the direct K-vs-V paired test.

Sources:
  - reports/phase4_scaling_law_v2_seed{0,1,2}.json  (6 pairs, 3 seeds, n=56)
  - reports/g0_v2_summary.json                      (flagship EM, all mappers)
  - reports/g0_v2_seed{0,1,2}.json                  (flagship per-sample rows)
  - reports/phase7_second_domain_seed{0,1,2}.json   (SQuAD cross-domain)

Definitions (all on mean answer log-likelihood, per sample):
  K main effect   = K-only  - Self
  V main effect   = V-only  - Self
  K x V interaction = Joint - K-only - V-only + Self

When per-sample four-arm rows are available (flagship and SQuAD), the script
also reports bootstrap CI95 and paired significance for every effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from stats_utils import bootstrap_ci95, paired_wilcoxon_test

REPORTS = Path(__file__).resolve().parent.parent / "reports"

PAIRS = [
    "8B_0.6B",
    "4B_0.6B",
    "1.7B_0.6B",
    "8B_4B",
    "4B_1.7B",
    "8B_1.7B",
]


def pair_latex(pair: str) -> str:
    a, b = pair.split("_")
    return f"{a}$\\to${b}"


def load_phase4() -> dict:
    """Return {pair: {seed: {'Self': mean, 'K-only': mean, ...}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for seed in (0, 1, 2):
        path = REPORTS / f"phase4_scaling_law_v2_seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for pr in data["pair_results"]:
            pair = pr["pair"]
            out.setdefault(pair, {})[str(seed)] = {
                k: v["mean"] for k, v in pr["summary_ll"].items()
            }
    return out


def effects_from_means(means: dict[str, float]) -> dict[str, float]:
    self_ll = means["Self"]
    k = means["K-only"] - self_ll
    v = means["V-only"] - self_ll
    j = means["Joint"] - self_ll
    return {
        "K_main": k,
        "V_main": v,
        "Joint": j,
        "interaction": j - k - v,
    }


def load_squad_rows() -> dict[str, list[dict]]:
    """Per-sample four-arm rows for SQuAD (3 seeds), if saved.

    Current phase7 reports only summary stats; per-sample rows are added in
    the Phase-2 rerun. Until then this returns an empty dict and the
    per-sample SQuAD block is omitted from the report.
    """
    out = {}
    for seed in (0, 1, 2):
        path = REPORTS / f"phase7_second_domain_seed{seed}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("rows") or data.get("samples")
            if rows:
                out[str(seed)] = rows
    return out


def load_flagship_rows() -> dict[str, list[dict]]:
    out = {}
    for seed in (0, 1, 2):
        path = REPORTS / f"g0_v2_seed{seed}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out[str(seed)] = data["rows"]
    return out


def load_phase4_rows() -> dict[str, dict[str, list[dict]]]:
    """Per-sample four-arm rows for all pairs, when phase4 saved them."""
    out: dict[str, dict[str, list[dict]]] = {}
    for seed in (0, 1, 2):
        candidates = [
            REPORTS / f"phase4_scaling_law_v2_withrows_seed{seed}.json",
            REPORTS / f"phase4_scaling_law_v2_seed{seed}.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            found = False
            for pr in data["pair_results"]:
                rows = pr.get("rows")
                if rows:
                    out.setdefault(pr["pair"], {})[str(seed)] = rows
                    found = True
            if found:
                break
    return out


def cluster_groups(rows: list[dict]) -> np.ndarray:
    """Document-cluster ids. phase4 rows carry no doc string, so recover groups
    from repeated answers is unsafe; fall back to sample index if absent."""
    if "doc_id" in rows[0]:
        ids = [r["doc_id"] for r in rows]
    elif "doc" in rows[0]:
        ids = [r["doc"] for r in rows]
    else:
        return np.arange(len(rows))
    uniq = {d: i for i, d in enumerate(dict.fromkeys(ids))}
    return np.array([uniq[d] for d in ids])


def clustered_ci95(values: np.ndarray, groups: np.ndarray, n_boot: int = 10000, seed: int = 1234):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    buckets = [values[groups == g] for g in uniq]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(buckets), len(buckets))
        boots[b] = np.concatenate([buckets[j] for j in pick]).mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def stat_block_clustered(values: np.ndarray, groups: np.ndarray) -> dict:
    mean, lo, hi = clustered_ci95(values, groups)
    return {"mean": round(float(mean), 3), "ci95_clustered": [round(lo, 3), round(hi, 3)]}


def rows_to_ll_matrix(rows: list[dict]) -> dict[str, np.ndarray]:
    """Map row dicts to four-arm LL vectors.

    G0 v2 rows store mapper arms with an 'Affine_' prefix; SQuAD rows (when
    added) use plain arm names. Detect which convention is present.
    """
    first = rows[0]
    if "Affine_K-only" in first:
        names = {
            "Self": "Self",
            "K-only": "Affine_K-only",
            "V-only": "Affine_V-only",
            "Joint": "Affine_Joint",
        }
    else:
        names = {
            "Self": "Self",
            "K-only": "K-only",
            "V-only": "V-only",
            "Joint": "Joint",
        }
    return {
        arm: np.asarray([r[names[arm]] for r in rows], dtype=float)
        for arm in ("Self", "K-only", "V-only", "Joint")
    }


def per_sample_effects(m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "K_main": m["K-only"] - m["Self"],
        "V_main": m["V-only"] - m["Self"],
        "Joint": m["Joint"] - m["Self"],
        "interaction": m["Joint"] - m["K-only"] - m["V-only"] + m["Self"],
        "K_minus_V": m["K-only"] - m["V-only"],
    }


def stat_block(values: np.ndarray) -> dict:
    mean, lo, hi = bootstrap_ci95(values, n_boot=10000, seed=1234)
    _, p, d = paired_wilcoxon_test(values, np.zeros_like(values))
    return {
        "mean": round(float(mean), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "wilcoxon_p": float(p),
        "cohens_d": round(float(d), 3),
    }


def analyze_rows(rows_by_seed: dict[str, list[dict]]) -> dict:
    out = {}
    for seed, rows in sorted(rows_by_seed.items()):
        m = rows_to_ll_matrix(rows)
        fx = per_sample_effects(m)
        out[seed] = {
            name: stat_block(vals) for name, vals in fx.items()
        }
        # exact-match retention for K-only / V-only / Joint
        em_self = np.mean([r["Self_em"] for r in rows])
        em = {}
        for arm in ("K-only", "V-only", "Joint"):
            key = f"Affine_{arm}_em"
            if key in rows[0]:
                em[arm] = np.mean([r[key] for r in rows])
        out[seed]["em"] = {
            "Self": round(float(em_self), 3),
            **{k: round(float(v), 3) for k, v in em.items()},
        }
    return out


def retention_table(phase4: dict) -> dict:
    """Apply the pre-registered retention criterion to EM (seed 0 EM strings)."""
    # seed-0 EM per arm from phase4 exact_match (already in the paper table)
    em = {}
    for seed in (0, 1, 2):
        path = REPORTS / f"phase4_scaling_law_v2_seed{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for pr in data["pair_results"]:
            pair = pr["pair"]
            em.setdefault(pair, {})[str(seed)] = {
                k: v for k, v in pr["exact_match"].items()
            }
    out = {}
    for pair in PAIRS:
        rows = []
        for seed in ("0", "1", "2"):
            e = em[pair][seed]
            for arm in ("K-only", "V-only", "Joint"):
                if e["Self"] > 0:
                    retention = e[arm] / e["Self"]
                else:
                    retention = float("nan")
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "em_self": e["Self"],
                        "em_arm": e[arm],
                        "retention": round(float(retention), 3),
                        "success_85": bool(retention >= 0.85),
                        "noninferior_3pp": bool(e[arm] - e["Self"] >= -0.03),
                    }
                )
        out[pair] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(REPORTS / "factorial_analysis.json"),
        help="Output JSON path",
    )
    args = ap.parse_args()

    phase4 = load_phase4()

    pair_effects = {}
    for pair in PAIRS:
        pair_effects[pair] = {
            seed: effects_from_means(phase4[pair][seed])
            for seed in ("0", "1", "2")
        }

    # Aggregate: mean across seeds for the paper table
    aggregate = {}
    for pair in PAIRS:
        rows = {
            effect: [pair_effects[pair][s][effect] for s in ("0", "1", "2")]
            for effect in ("K_main", "V_main", "Joint", "interaction")
        }
        aggregate[pair] = {
            effect: {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals, ddof=1)), 2),
                "per_seed": [round(float(v), 2) for v in vals],
            }
            for effect, vals in rows.items()
        }

    # Per-sample analyses where rows exist
    flagship_rows = load_flagship_rows()
    squad_rows = load_squad_rows()
    phase4_rows = load_phase4_rows()

    six_pair_per_sample = {}
    for pair, by_seed in phase4_rows.items():
        six_pair_per_sample[pair] = {}
        for seed, rows in by_seed.items():
            m = rows_to_ll_matrix(rows)
            fx = per_sample_effects(m)
            groups = cluster_groups(rows)
            six_pair_per_sample[pair][seed] = {
                name: {**stat_block(vals),
                       **stat_block_clustered(vals, groups)}
                for name, vals in fx.items()
            }

    result = {
        "task": "factorial_analysis",
        "note": (
            "K_main = K-only - Self; V_main = V-only - Self; "
            "interaction = Joint - K-only - V-only + Self (mean LL). "
            "Summary-based effects use per-seed mean LL from phase4; "
            "per-sample significance available where four-arm rows are saved "
            "(flagship 8B->0.6B, all six pairs once phase4 rows are persisted). "
            "ci95_clustered resamples whole documents (data repeats ~7 Q/doc)."
        ),
        "pair_effects_seed_means": pair_effects,
        "aggregate": aggregate,
        "flagship_per_sample": (
            analyze_rows(flagship_rows) if flagship_rows else None
        ),
        "squad_per_sample": analyze_rows(squad_rows) if squad_rows else None,
        "six_pair_per_sample": six_pair_per_sample or None,
        "retention": retention_table(phase4),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # Console table
    print("Pair          Effect   mean    std     seed0  seed1  seed2")
    for pair in PAIRS:
        for effect in ("K_main", "V_main", "Joint", "interaction"):
            a = aggregate[pair][effect]
            print(
                f"{pair:12s} {effect:11s} {a['mean']:+6.2f} {a['std']:5.2f}  "
                f"{a['per_seed'][0]:+5.2f} {a['per_seed'][1]:+5.2f} "
                f"{a['per_seed'][2]:+5.2f}"
            )

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
