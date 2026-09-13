# -*- coding: utf-8 -*-
"""One-to-one checks for the audit3 paper-side edits.

Every assertion maps to an audit3 finding through the tables in
`paper/audit/REVISION_PLAN3.md`. Run from anywhere:

    python paper/audit/verify_audit3_edits.py

Exit code 0 = all assertions hold. The guard exists because audit3's main
technical objection is that one mechanism claim (mapped-key routing, a key-side
statement) was used to explain a value-side result that never sees a mapped key;
that claim appeared in the Abstract, the Introduction, Analysis 6.3, Discussion
7.1 and the Conclusion, so editing one site and missing another would leave the
paper self-contradictory again.

Tags: T1 (key/value separation), T1b (four-layer Discussion), T2 (evaluator
attribution), T3 (equivalence wording), A6 (document-clustered statistics, with
the numbers cross-checked against reports/cluster_stats_audit3.json).

Assertions for the items that still need GPU results (A1-A5) are deliberately
absent: they are added when those reports land.
"""
from __future__ import annotations

import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "paper", "main.tex")
CLUSTER = os.path.join(ROOT, "reports", "cluster_stats_audit3.json")

FAILURES: list[str] = []


def absent(tag: str, text: str, needle: str) -> None:
    n = text.count(needle)
    if n:
        FAILURES.append("%s: %r must be absent, found %d" % (tag, needle, n))


def present(tag: str, text: str, needle: str) -> None:
    if needle not in text:
        FAILURES.append("%s: %r must be present" % (tag, needle))


def close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    tex = io.open(TEX, encoding="utf-8").read()

    # ------------------------------------------------------- T1: audit3 par.4/5
    # The value arm uses the student's own keys, so mapped-key routing cannot
    # explain its residual gap. All five sites must agree.
    absent("T1", tex, "addressing perturbation")
    absent("T1", tex, "where routing diverges more")
    present("T1", tex, "this arm never sees a mapped key")
    present("T1", tex, "a key-side diagnostic")
    present("T1", tex, "The value arm uses the\n  student's own keys")
    present("T1", tex,
            "\\subsection{Two separate failures: mapped keys perturb routing, "
            "mapped values must be consumable}")
    present("T1", tex, "divergence as a diagnostic for the key side")
    present("T1", tex, "key-side mismatch is diagnosable (routing divergence)")

    # ------------------------------------------------------ T1b: audit3 par.16
    absent("T1b", tex, "The evidence supports three statements")
    absent("T1b", tex, "co-varies with how much the mapper and adapter recover")
    present("T1b", tex, "The evidence supports four statements")
    present("T1b", tex, "\\emph{(ii) Value transfer.}")
    present("T1b", tex, "\\emph{(iii) Key transfer.}")
    present("T1b", tex, "\\emph{(iv) Joint handoff.}")
    present("T1b", tex, "depends on which side is in play")
    present("T1b", tex, "routing divergence is a diagnostic for the\nkey side")

    # -------------------------------------------------------- T2: audit3 par.15
    # The two evaluator defects are in our own implementation; the paper must
    # not attribute them to a community protocol or to "the released code".
    absent("T2", tex, "published evaluation code")
    absent("T2", tex, "released code")
    absent("T2", tex, "The standard protocol overstates transfer")
    absent("T2", tex, "The standard implementation advances a single cache")
    absent("T2", tex, "the standard protocol's task\nmetric")
    present("T2", tex,
            "The evaluation protocol we used in earlier versions of this work "
            "overstates")
    present("T2", tex,
            "The implementation used in earlier versions of this work advances a "
            "single cache")
    present("T2", tex, "we did not audit third-party implementations")
    present("T2", tex, "both defects are in that\nimplementation")

    # --------------------------------------------------------- T3: audit3 par.3
    absent("T3", tex, "end-to-end equivalent to full prefill")
    present("T3", tex, "closely agrees with full prefill on the quantity we")
    present("T3", tex, "remaining distinct from it at the\ntoken level")

    # ---------------------------------------------- A6: audit3 par.9/10/11 (CPU)
    absent("A6", tex, "document-clustered EM intervals exist only in the seed-0 reports")
    absent("A6", tex, "archived for seed 0 only")
    absent("A6", tex, "the only seed for\nwhich the source archives document-clustered")
    present("A6", tex, "computed for all three seeds from the")
    present("A6", tex, "clusters $=$ documents, eight per seed")
    present("A6", tex, "$[0.375, 0.482]$, $[0.429, 0.482]$, and $[0.429, 0.482]$")
    present("A6", tex, "clustered CI95 is $[0.839, 0.982]$ against $[0.018, 0.107]$")

    # Cross-check those literals against the archived statistics.
    if not os.path.isfile(CLUSTER):
        FAILURES.append("A6: %s must exist" % CLUSTER)
    else:
        stats = json.load(io.open(CLUSTER, encoding="utf-8"))
        key = "8B_4B_seed0"
        entry = stats["fourarm"].get(key)
        if entry is None:
            FAILURES.append("A6: %s missing from cluster stats" % key)
        else:
            v = entry["arms"]["V-only"]
            if not close(v["EM"], 0.429):
                FAILURES.append("A6: 8B_4B seed0 V-only EM %.4f, paper says 0.43"
                                % v["EM"])
            if not (close(v["EM_ci95_clustered"][0], 0.375)
                    and close(v["EM_ci95_clustered"][1], 0.482)):
                FAILURES.append("A6: 8B_4B seed0 V-only clustered CI changed (%s)"
                                % v["EM_ci95_clustered"])
        ems = []
        for seed in (0, 1, 2):
            entry = stats["fourarm"].get("8B_4B_seed%d" % seed)
            if entry:
                ems.append(entry["arms"]["V-only"]["EM"])
        if ems and not all(0.42 <= e <= 0.45 for e in ems):
            FAILURES.append("A6: 8B_4B V-only per-seed EM outside 0.43-0.45: %s" % ems)
        oa = stats["outaware"].get("phaseB_outaware_1.7B_0.6B_seed0", {})
        oa_arms = oa.get("arms", {})
        if "V-OutAware" in oa_arms:
            lo, hi = oa_arms["V-OutAware"]["EM_ci95_clustered"]
            if not (close(lo, 0.839, 0.01) and close(hi, 0.982, 0.01)):
                FAILURES.append("A6: 1.7B V-OutAware clustered CI changed (%s, %s)"
                                % (lo, hi))

    # ------------------------------------- A7: audit3 par.13 (optional, CPU)
    # Rule-based taxonomy of the archived Self-baseline generations.
    present("A7", tex, "Classifying the archived\n  generations, every $1.7$B failure")
    present("A7", tex, "no generation truncated at the decode cap")
    tax_path = os.path.join(ROOT, "reports", "error_taxonomy_selfdiag.json")
    if not os.path.isfile(tax_path):
        FAILURES.append("A7: %s must exist" % tax_path)
    else:
        tax = json.load(io.open(tax_path, encoding="utf-8"))["students"]
        counts = tax.get("1.7B", {}).get("counts", {})
        if counts.get("wrong_entity") != 32 or counts.get("correct") != 24:
            FAILURES.append("A7: 1.7B taxonomy changed: %s" % counts)
        for bad in ("refusal_or_hedge", "repetition_loop", "no_content",
                    "hit_length_limit"):
            if counts.get(bad):
                FAILURES.append("A7: 1.7B failures include %s=%d"
                                % (bad, counts[bad]))

    if FAILURES:
        print("FAIL: audit3 paper-side assertions")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("OK: all audit3 paper-side assertions hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
