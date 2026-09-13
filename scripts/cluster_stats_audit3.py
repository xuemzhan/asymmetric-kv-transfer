"""A6: document-clustered inference for every seed (audit3 sections 9-10).

The paper reports std over seeds for the four-arm table and says that
document-clustered EM intervals exist only in the seed-0 reports. That is
stale: the four-arm reports carry `doc_id` on every row for all three seeds,
and the control and mapper reports can be joined to
`data/test_v2_seed{seed}.json` on `id`. Adapter and causality reports carry no
per-sample rows at all, so they are computed only if they were regenerated with
`--dump-rows` (A2).

Cluster = document. The synthetic test set has 56 questions over 8 documents,
so per-sample resampling understates uncertainty. This script recomputes, for
every archived arm:
  - EM mean and per-sample bootstrap CI95 (for comparison with the paper),
  - document-clustered bootstrap CI95,
  - a document-level paired test against the report's Self arm (Wilcoxon on
    per-document mean EM, falling back to a sign test on document means when
    Wilcoxon is degenerate),
  - the number of clusters actually used.

Writes `reports/cluster_stats_audit3.json` and prints a compact table.

Usage:
  python scripts/cluster_stats_audit3.py [--reports reports] [--data data]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

N_BOOT = 10000


def bootstrap_ci95(values: np.ndarray, groups: np.ndarray | None = None,
                    n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Mean and bootstrap CI95; resamples whole clusters when groups are given.

    Self-contained on purpose: this script must run on a CPU-only machine with
    no torch/transformers installed.
    """
    values = np.asarray(values, dtype=float)
    if groups is None:
        buckets = [values]
    else:
        uniq = np.unique(groups)
        buckets = [values[np.asarray(groups) == g] for g in uniq]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(buckets), len(buckets))
        boots[b] = np.concatenate([buckets[j] for j in pick]).mean()
    return (float(values.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def doc_level_paired_test(values: np.ndarray, base: np.ndarray,
                          groups: np.ndarray) -> tuple:
    """Paired test on document means; returns (stat, p, test_name)."""
    uniq = np.unique(groups)
    v = np.array([values[groups == g].mean() for g in uniq])
    b = np.array([base[groups == g].mean() for g in uniq])
    d = v - b
    if np.allclose(d, 0.0):
        return float("nan"), 1.0, "all-zero differences"
    if np.count_nonzero(d) >= 6 and not np.allclose(d, 0.0):
        try:
            stat, p = stats.wilcoxon(v, b)
            if np.isfinite(p):
                return float(stat), float(p), "wilcoxon_on_document_means"
        except ValueError:
            pass
    n_pos = int(np.count_nonzero(d > 0))
    n_nonzero = int(np.count_nonzero(d != 0))
    if n_nonzero == 0:
        return float("nan"), 1.0, "no_nonzero_differences"
    p = float(stats.binomtest(n_pos, n_nonzero, 0.5).pvalue)
    return float(n_pos), p, "sign_test_on_document_means"


def summarise_arm(rows: list, arm: str, base_arm: str, groups: np.ndarray) -> dict:
    ems = np.array([float(r[arm + "_em"]) for r in rows])
    out = {
        "EM": float(ems.mean()),
        "EM_ci95": list(bootstrap_ci95(ems, None, n_boot=2000)[1:]),
        "n": int(len(ems)),
        "n_docs": int(len(np.unique(groups))),
    }
    _, lo, hi = bootstrap_ci95(ems, groups, n_boot=N_BOOT, seed=0)
    out["EM_ci95_clustered"] = [lo, hi]
    if base_arm is not None and (base_arm + "_em") in rows[0]:
        base = np.array([float(r[base_arm + "_em"]) for r in rows])
        stat, p, name = doc_level_paired_test(ems, base, groups)
        out["delta_vs_" + base_arm] = float(ems.mean() - base.mean())
        out["doc_level_p"] = p
        out["doc_level_test"] = name
        out["doc_level_stat"] = stat
    return out


def groups_for_rows(rows: list, seed: int, data_dir: str) -> np.ndarray | None:
    """Document id per row: `doc_id` if present, else join on `id`."""
    if rows and "doc_id" in rows[0]:
        return np.array([r["doc_id"] for r in rows])
    path = os.path.join(data_dir, f"test_v2_seed{seed}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        test = json.load(f)
    doc_of = {s["id"]: s["doc"] for s in test}
    if not all(r["id"] in doc_of for r in rows):
        return None
    uniq = {d: i for i, d in enumerate(dict.fromkeys(doc_of.values()))}
    return np.array([uniq[doc_of[r["id"]]] for r in rows])


def arm_names(rows: list) -> list:
    return [k for k in rows[0] if not k.endswith("_em")
            and k not in ("id", "hop", "answer", "doc_id")]


def collect_fourarm(reports_dir: str, data_dir: str) -> dict:
    """All six pairs over three seeds, with document-clustered EM intervals."""
    out: dict = {}
    files = sorted(glob.glob(os.path.join(reports_dir, "phaseB_fourarm_seed*.json")))
    files += sorted(glob.glob(os.path.join(reports_dir, "phaseB_fourarm_8B_4B_seed0.json")))
    for path in files:
        with open(path) as f:
            report = json.load(f)
        for entry in report["pair_results"]:
            rows = entry["rows"]
            groups = groups_for_rows(rows, entry["seed"], data_dir)
            arm = f"{entry['pair']}_seed{entry['seed']}"
            if groups is None:
                out[arm] = {"error": "no document identifiers"}
                continue
            entry_out = {}
            for name in arm_names(rows):
                base = "Self" if (name != "Self" and "Self_em" in rows[0]) else None
                entry_out[name] = summarise_arm(rows, name, base, groups)
            out[arm] = {"pair": entry["pair"], "seed": entry["seed"],
                        "n_docs": int(len(np.unique(groups))), "arms": entry_out}
    return out


def collect_controls(reports_dir: str, data_dir: str) -> dict:
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "phaseB_controls_*.json"))):
        name = os.path.basename(path)[:-5]
        if "LEGACY" in name:
            continue
        with open(path) as f:
            report = json.load(f)
        rows = report.get("rows")
        if not rows:
            continue
        seed = report.get("seed", 0)
        groups = groups_for_rows(rows, seed, data_dir)
        if groups is None:
            continue
        entry_out = {}
        for arm in arm_names(rows):
            base = "Self" if (arm != "Self" and "Self_em" in rows[0]) else None
            entry_out[arm] = summarise_arm(rows, arm, base, groups)
        out[name] = {"pair": report.get("pair"), "seed": seed,
                     "n_docs": int(len(np.unique(groups))), "arms": entry_out}
    return out


def collect_outaware(reports_dir: str, data_dir: str) -> dict:
    out: dict = {}
    pattern = os.path.join(reports_dir, "phaseB_outaware_*_seed*.json")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)[:-5]
        with open(path) as f:
            report = json.load(f)
        rows = report.get("rows") or []
        if not rows:
            continue
        seed = report.get("seed", 0)
        groups = groups_for_rows(rows, seed, data_dir)
        if groups is None:
            continue
        entry_out = {}
        for arm in arm_names(rows):
            base = "Self" if (arm != "Self" and "Self_em" in rows[0]) else None
            entry_out[arm] = summarise_arm(rows, arm, base, groups)
        out[name] = {"pair": report.get("pair"), "seed": seed,
                     "n_docs": int(len(np.unique(groups))), "arms": entry_out}
    return out


def collect_adapter(reports_dir: str, data_dir: str) -> dict:
    """Adapter / causality arms, only where --dump-rows was used."""
    out: dict = {}
    patterns = ["phaseB_adapter_causal20_*.json", "phaseB_adapter_causal_*.json"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(reports_dir, pattern))):
            name = os.path.basename(path)[:-5]
            if name in out:
                continue
            with open(path) as f:
                report = json.load(f)
            rows_by_condition = report.get("rows_by_condition")
            if not rows_by_condition:
                out[name] = {"missing": "regenerated without --dump-rows"}
                continue
            seed = report.get("seed", 0)
            entry_out = {}
            for condition, rows in rows_by_condition.items():
                groups = groups_for_rows(rows, seed, data_dir)
                if groups is None:
                    entry_out[condition] = {"error": "no document identifiers"}
                    continue
                arms = {}
                for arm in arm_names(rows):
                    base = "Self" if (arm != "Self" and "Self_em" in rows[0]) else None
                    arms[arm] = summarise_arm(rows, arm, base, groups)
                entry_out[condition] = {"n_docs": int(len(np.unique(groups))),
                                        "arms": arms}
            out[name] = {"pair": report.get("pair"), "seed": seed,
                         "epochs": report.get("epochs"),
                         "conditions": entry_out}
    return out


def collect_alignment_passthrough(reports_dir: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(
            reports_dir, "phaseB_alignment_repaired_*.json"))):
        name = os.path.basename(path)[:-5]
        with open(path) as f:
            report = json.load(f)
        maps = {}
        for map_name, entry in report.get("results", {}).items():
            summary = entry.get("summary", {})
            maps[map_name] = {
                arm: {"EM": v.get("EM"),
                      "EM_ci95_clustered": v.get("EM_ci95_clustered")}
                for arm, v in summary.items()
            }
        out[name] = {"pair": report.get("pair"), "seed": report.get("seed"),
                     "note": "clustered intervals already stored in the report",
                     "maps": maps}
    return out


def three_seed_summary(fourarm: dict) -> dict:
    """Mean/std of EM and of the clustered interval bounds per pair and arm."""
    by_pair: dict = {}
    for key, entry in fourarm.items():
        if "arms" not in entry:
            continue
        by_pair.setdefault(entry["pair"], []).append(entry)
    out = {}
    for pair, entries in sorted(by_pair.items()):
        arms = sorted(set().union(*[set(e["arms"]) for e in entries]))
        out[pair] = {}
        for arm in arms:
            ems = [e["arms"][arm]["EM"] for e in entries if arm in e["arms"]]
            los = [e["arms"][arm]["EM_ci95_clustered"][0] for e in entries
                   if arm in e["arms"]]
            his = [e["arms"][arm]["EM_ci95_clustered"][1] for e in entries
                   if arm in e["arms"]]
            out[pair][arm] = {
                "n_seeds": len(ems),
                "EM_mean": float(np.mean(ems)),
                "EM_std": float(np.std(ems, ddof=1)) if len(ems) > 1 else 0.0,
                "EM_clustered_lo_mean": float(np.mean(los)),
                "EM_clustered_hi_mean": float(np.mean(his)),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--data", default="data")
    ap.add_argument("--output", default="reports/cluster_stats_audit3.json")
    a = ap.parse_args()

    fourarm = collect_fourarm(a.reports, a.data)
    controls = collect_controls(a.reports, a.data)
    outaware = collect_outaware(a.reports, a.data)
    adapter = collect_adapter(a.reports, a.data)
    alignment = collect_alignment_passthrough(a.reports)

    report = {
        "task": "cluster_stats_audit3",
        "n_boot": N_BOOT,
        "cluster_unit": "document",
        "fourarm": fourarm,
        "controls": controls,
        "outaware": outaware,
        "adapter_causality": adapter,
        "alignment_repaired": alignment,
        "fourarm_three_seed": three_seed_summary(fourarm),
    }
    with open(a.output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[cluster-stats] saved: {a.output}")

    print("\n== four-arm EM, per seed (clustered CI95) ==")
    for key in sorted(fourarm):
        entry = fourarm[key]
        if "arms" not in entry:
            print(f"  {key}: {entry}")
            continue
        cells = " ".join(
            f"{arm}={entry['arms'][arm]['EM']:.3f}"
            f"[{entry['arms'][arm]['EM_ci95_clustered'][0]:.3f},"
            f"{entry['arms'][arm]['EM_ci95_clustered'][1]:.3f}]"
            for arm in ("Self", "K-only", "V-only", "Joint")
            if arm in entry["arms"])
        print(f"  {key} (docs={entry['n_docs']}): {cells}")

    print("\n== three-seed summary ==")
    for pair, arms in report["fourarm_three_seed"].items():
        cells = " ".join(
            f"{arm}={arms[arm]['EM_mean']:.3f}+-{arms[arm]['EM_std']:.3f}"
            for arm in ("Self", "K-only", "V-only", "Joint") if arm in arms)
        print(f"  {pair:9s} {cells}")

    print("\n== controls (clustered) ==")
    for name, entry in sorted(controls.items()):
        if "arms" not in entry:
            continue
        bad = [a for a in entry["arms"] if a.endswith("Wrong_K-only")]
        keys = [k for k in ("K-only", "V-only", "Joint") if k in entry["arms"]]
        cells = " ".join(
            f"{k}={entry['arms'][k]['EM']:.3f}"
            f"[{entry['arms'][k]['EM_ci95_clustered'][0]:.3f},"
            f"{entry['arms'][k]['EM_ci95_clustered'][1]:.3f}]" for k in keys)
        print(f"  {name}: Self={entry['arms']['Self']['EM']:.3f} {cells}")
        if bad:
            print(f"      wrong-document arms present: {bad}")

    print("\n== adapter / causality (clustered) ==")
    if not adapter:
        print("  none archived")
    for name, entry in sorted(adapter.items()):
        if "missing" in entry:
            print(f"  {name}: {entry['missing']}")
            continue
        for condition, block in entry.get("conditions", {}).items():
            if "arms" not in block:
                print(f"  {name}/{condition}: {block}")
                continue
            keys = [k for k in ("Self", "K-only", "V-only", "Joint",
                                "WrongJoint", "RandKV", "ZeroKV")
                    if k in block["arms"]]
            cells = " ".join(f"{k}={block['arms'][k]['EM']:.3f}" for k in keys)
            print(f"  {name}/{condition}: {cells}")


if __name__ == "__main__":
    main()
