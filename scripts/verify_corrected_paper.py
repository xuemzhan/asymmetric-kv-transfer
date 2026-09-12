#!/usr/bin/env python3
"""verify_corrected_paper.py — verify the rewritten paper against reports.

Checks (1) key corrected numbers recomputed from reports, (2) that main.tex
contains the corresponding strings, and (3) that invalidated overclaim phrases
are absent.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
REP = PROJ / "reports"
TEX = (PROJ / "paper" / "main.tex").read_text(encoding="utf-8")

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# --- corrected four-arm (3 seeds) ------------------------------------------
fa = {}
for s in (0, 1, 2):
    d = json.loads((REP / f"phaseB_fourarm_seed{s}.json").read_text())
    for r in d["pair_results"]:
        fa.setdefault(r["pair"], {"Self": [], "K": [], "V": [], "J": [], "Kll": [], "Vll": [], "Jll": []})
        fa[r["pair"]]["Self"].append(r["self_EM"])
        for a, k in (("K-only", "K"), ("V-only", "V"), ("Joint", "J")):
            fa[r["pair"]][k].append(r["summary"][a]["EM"])
            fa[r["pair"]][k + "ll"].append(r["summary"][a]["delta_vs_self"])

f = fa["8B_0.6B"]
check("8B_0.6B Self EM ~0.899", abs(np.mean(f["Self"]) - 0.899) < 0.005)
check("8B_0.6B teacher arms EM ~0", max(np.mean(f[k]) for k in "KVJ") < 0.005)
check("8B_0.6B K dLL ~+2.62", abs(np.mean(f["Kll"]) - 2.619) < 0.02)
check("8B_0.6B V dLL ~-0.23", abs(np.mean(f["Vll"]) + 0.227) < 0.02)
check("8B_4B V EM ~0.44", abs(np.mean(fa["8B_4B"]["V"]) - 0.440) < 0.01)

# --- controls --------------------------------------------------------------
ctrl = json.loads((REP / "phaseB_controls_8B_0.6B_seed0_fixed.json").read_text())["summary"]
check("control wrong-doc K LL ~ real", abs(ctrl["Wrong_K-only"]["delta_vs_self"] - ctrl["K-only"]["delta_vs_self"]) < 0.1)
check("control zero-KV LL > K-only", ctrl["Zero_KV"]["delta_vs_self"] > ctrl["K-only"]["delta_vs_self"])
check("control all teacher EM ~0", max(ctrl[k]["EM"] for k in ["K-only", "V-only", "Joint", "Wrong_K-only", "Zero_KV", "Rand_K"]) < 0.03)
check("control shuffled student EM high", ctrl["Shuf_KV"]["EM"] > 0.9)

# --- mapper + adapter ------------------------------------------------------
mo = {p: json.loads((REP / f"phaseB_outaware_{p}_seed0.json").read_text()) for p in ["1.7B_0.6B", "8B_0.6B"]}
def em(rows, a):
    return float(np.mean([r[a + "_em"] for r in rows]))
check("WO 1.7B ~0.929", abs(em(mo["1.7B_0.6B"]["rows"], "V-WOAware") - 0.929) < 0.01)
check("WO 8B ~0.268", abs(em(mo["8B_0.6B"]["rows"], "V-WOAware") - 0.268) < 0.01)

ad = {f.name: json.loads(f.read_text()) for f in REP.glob("phaseB_adapter_*.json")}
def _o_proj_name(s):
    a = f"phaseB_adapter_1.7B_0.6B_o_proj_seed{s}.json"
    return a if a in ad else f"phaseB_adapter_1.7B_0.6B_seed{s}.json"
j17 = [ad[_o_proj_name(s)]["results"]["adapter(joint)"]["Joint"]["EM"] for s in (0, 1, 2)]
check("adapter 1.7B joint Joint ~0.827", abs(np.mean(j17) - 0.827) < 0.02)
a8 = ad["phaseB_adapter_8B_0.6B_o_proj_seed0.json"]["results"]
check("adapter 8B K ~0.857", abs(a8["adapter(joint)"]["K-only"]["EM"] - 0.857) < 0.01)
check("adapter 8B self-control Joint ~0.107", abs(a8["adapter(self)"]["Joint"]["EM"] - 0.107) < 0.01)

# --- tex content -----------------------------------------------------------
required = ["corrected greedy exact match", "consumption adapter", "0.94/0.94/0.83",
            "0.79/0.36/0.47", "$W_O$-aware", "top-1 key agreement",
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
