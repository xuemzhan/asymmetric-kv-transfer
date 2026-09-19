#!/usr/bin/env python3
"""verify_corrected_paper.py — verify the rewritten paper against reports.

Checks (1) key corrected numbers recomputed from reports, (2) that main.tex
contains the corresponding strings, and (3) that invalidated overclaim phrases
are absent.

A missing or corrupt report/tex artifact is reported as a FAILURE and the run
continues, so "the guard failed" is never confusable with "the guard crashed".
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
REP = PROJ / "reports"

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def close(got, want, tol=0.01):
    """Tolerance check that fails (instead of raising) on missing/NaN input."""
    try:
        return bool(np.isfinite(float(got)) and abs(float(got) - want) <= tol)
    except (TypeError, ValueError):
        return False


def mean_of(seq):
    seq = [float(x) for x in (seq or []) if x is not None]
    return float(np.mean(seq)) if seq else float("nan")


def val(obj, *keys):
    """Nested lookup returning NaN instead of raising on a missing path."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def load_json(path: Path) -> dict:
    """Read one report; a missing/corrupt file is a failure, not a crash."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print("FAIL could not read %s (%s)" % (path, exc))
        fails.append("report %s readable" % path.name)
        return {}
    if not isinstance(data, dict):
        print("FAIL %s is not a JSON object" % path)
        fails.append("report %s is an object" % path.name)
        return {}
    return data


try:
    TEX = (PROJ / "paper" / "main.tex").read_text(encoding="utf-8")
except OSError as exc:
    print("FAIL could not read paper/main.tex (%s)" % exc)
    fails.append("paper/main.tex readable")
    TEX = ""

# --- corrected four-arm (3 seeds) ------------------------------------------
# Rows are keyed by (pair, seed) over *every* phaseB_fourarm_*.json: the 8B_4B
# seed-0 row lives in its own file (phaseB_fourarm_8B_4B_seed0.json), so reading
# only phaseB_fourarm_seed{0,1,2}.json silently gave 8B_4B a two-seed mean.
rows = {}
for path in sorted(REP.glob("phaseB_fourarm_*.json")):
    for r in load_json(path).get("pair_results", []):
        rows[(r["pair"], r["seed"])] = r

fa = {}
for (pair, seed), r in rows.items():
    e = fa.setdefault(pair, {"seeds": [], "Self": [], "K": [], "V": [], "J": [],
                             "Kll": [], "Vll": [], "Jll": []})
    e["seeds"].append(seed)
    e["Self"].append(r["self_EM"])
    for a, k in (("K-only", "K"), ("V-only", "V"), ("Joint", "J")):
        e[k].append(r["summary"][a]["EM"])
        e[k + "ll"].append(r["summary"][a]["delta_vs_self"])

PAIRS = ("8B_0.6B", "8B_4B", "8B_1.7B", "4B_1.7B", "4B_0.6B", "1.7B_0.6B")
check("four-arm: all six pairs present", set(PAIRS) <= set(fa))
for p in PAIRS:
    n = len(fa.get(p, {}).get("seeds", []))
    check("four-arm %s has 3 seeds (got %d)" % (p, n), n == 3)

f = fa.get("8B_0.6B") or {}
check("8B_0.6B Self EM ~0.899", close(mean_of(f.get("Self")), 0.899, 0.005))
check("8B_0.6B teacher arms EM ~0",
      max([mean_of(f.get(k)) for k in "KVJ"] or [float("nan")]) < 0.005)
check("8B_0.6B K dLL ~+2.62", close(mean_of(f.get("Kll")), 2.619, 0.02))
check("8B_0.6B V dLL ~-0.23", close(mean_of(f.get("Vll")), -0.227, 0.02))

# 8B_4B three-seed means: 0.803571/0.839286/0.803571 Self; 0.142857/0.160714/
# 0.142857 K; 0.428571/0.446429/0.446429 V; 0.017857/0.053571/0.125 Joint.
b4 = fa.get("8B_4B") or {}
check("8B_4B Self EM ~0.8157 (3 seeds)",
      close(mean_of(b4.get("Self")), 0.8157, 0.002))
check("8B_4B K EM ~0.1490 (3 seeds)", close(mean_of(b4.get("K")), 0.1490, 0.002))
check("8B_4B V EM ~0.4405 (3 seeds)", close(mean_of(b4.get("V")), 0.4405, 0.002))
check("8B_4B Joint EM ~0.0655 (3 seeds)",
      close(mean_of(b4.get("J")), 0.0655, 0.002))

# --- controls --------------------------------------------------------------
ctrl = load_json(REP / "phaseB_controls_8B_0.6B_seed0_fixed.json").get("summary") or {}
if not ctrl:
    check("controls report present with a summary", False)
else:
    check("control wrong-doc K LL ~ real",
          close(val(ctrl, "Wrong_K-only", "delta_vs_self")
                - val(ctrl, "K-only", "delta_vs_self"), 0.0, 0.1))
    check("control zero-KV LL > K-only",
          val(ctrl, "Zero_KV", "delta_vs_self") > val(ctrl, "K-only", "delta_vs_self"))
    check("control all teacher EM ~0",
          max(val(ctrl, k, "EM") for k in ["K-only", "V-only", "Joint",
                                           "Wrong_K-only", "Zero_KV",
                                           "Rand_K"]) < 0.03)
    check("control shuffled student EM high", val(ctrl, "Shuf_KV", "EM") > 0.9)

# --- mapper + adapter ------------------------------------------------------
mo = {p: load_json(REP / ("phaseB_outaware_%s_seed0.json" % p))
      for p in ("1.7B_0.6B", "8B_0.6B")}


def em(rows, a):
    return mean_of([r.get(a + "_em") for r in (rows or [])])


check("WO 1.7B ~0.929", close(em(mo["1.7B_0.6B"].get("rows"), "V-WOAware"), 0.929, 0.01))
check("WO 8B ~0.268", close(em(mo["8B_0.6B"].get("rows"), "V-WOAware"), 0.268, 0.01))

ad = {f.name: load_json(f) for f in REP.glob("phaseB_adapter_*.json")}


def _o_proj_name(s):
    a = "phaseB_adapter_1.7B_0.6B_o_proj_seed%d.json" % s
    return a if a in ad else "phaseB_adapter_1.7B_0.6B_seed%d.json" % s


j17 = [val(ad.get(_o_proj_name(s), {}), "results", "adapter(joint)", "Joint", "EM")
       for s in (0, 1, 2)]
check("adapter 1.7B joint Joint ~0.827", close(mean_of(j17), 0.827, 0.02))
a8 = ad.get("phaseB_adapter_8B_0.6B_o_proj_seed0.json", {}).get("results", {})
check("adapter 8B K ~0.857", close(val(a8, "adapter(joint)", "K-only", "EM"), 0.857, 0.01))
check("adapter 8B self-control Joint ~0.107",
      close(val(a8, "adapter(self)", "Joint", "EM"), 0.107, 0.01))

# --- tex content -----------------------------------------------------------
required = ["corrected greedy exact match", "consumption adapter", "0.94/0.94/0.83",
            "0.79/0.36/0.47", "$W_O$-aware", "top-1 attention agreement",
            "wrong document", "+2.62", "0.94"]
for r in required:
    check(f"main.tex contains {r!r}", r in TEX)
forbidden = ["near-universally", "capability-gated", "beats both", "EM $\\ge 0.71$",
             "Addressing Transfers, Content Does Not"]
for r in forbidden:
    check(f"main.tex free of {r!r}", r not in TEX)

print()
print(f"{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
raise SystemExit(1 if fails else 0)
