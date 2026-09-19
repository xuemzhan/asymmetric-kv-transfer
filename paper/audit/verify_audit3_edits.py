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

def load_report(name: str) -> dict:
    """Load reports/<name>; a missing or corrupt file is a FAILURE, not a crash.

    A guard that raises cannot be told apart from a guard that failed, so any
    unreadable artifact is recorded in FAILURES and an empty dict is returned.
    """
    path = os.path.join(ROOT, "reports", name)
    try:
        with io.open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        FAILURES.append("REPORT: %s could not be read (%s)" % (path, exc))
        return {}
    if not isinstance(data, dict):
        FAILURES.append("REPORT: %s is not a JSON object" % path)
        return {}
    return data


def main() -> int:
    tex = io.open(TEX, encoding="utf-8").read()

    # ------------------------------------------------------- T1: audit3 par.4/5
    # The value arm uses the student's own keys, so mapped-key routing cannot
    # explain its residual gap. All five sites must agree.
    absent("T1", tex, "addressing perturbation")
    absent("T1", tex, "where routing diverges more")
    present("T1", tex, "this arm never sees a mapped key")
    present("T1", tex, "a key-side diagnostic")
    # REVISION_PLAN4 T2 rewrote contribution 2, so the old wording is gone; the
    # claim is still pinned here and in the "this arm never sees a mapped key"
    # assertion above.
    # W32: contribution 2 was re-wrapped (wording unchanged), so the pinned phrase
    # now straddles a line break; pin its two halves instead.
    present("T1", tex, "never sees a mapped key")
    present("T1", tex, "its failure is not addressing")
    present("T1", tex,
            "\\subsection{Two separate failures: mapped keys perturb routing, "
            "mapped values must be consumable}")
    present("T1", tex, "divergence as a diagnostic for the key side")
    # W32: the key side is now intervened on, so "diagnosable (routing divergence)"
    # became "diagnosable and, on our evidence, not repairable".
    present("T1", tex, "key-side mismatch is diagnosable and, on our evidence, not repairable")
    absent("T1", tex, "diagnosable (routing divergence) and the value-side one is")

    # ------------------------------------------------------ T1b: audit3 par.16
    absent("T1b", tex, "The evidence supports three statements")
    absent("T1b", tex, "co-varies with how much the mapper and adapter recover")
    present("T1b", tex, "The evidence supports four statements")
    present("T1b", tex, "\\emph{(ii) Value transfer.}")
    present("T1b", tex, "\\emph{(iii) Key transfer.}")
    present("T1b", tex, "\\emph{(iv) Joint handoff.}")
    present("T1b", tex, "depends on which side is in play")
    # W32: the "we measure it rather than intervening on it" clause was withdrawn
    # when A1 intervened on the key side; the replacement sentence is pinned here.
    absent("T1b", tex, "we measure it rather than intervening on it")
    present("T1b", tex, "On the key side we can move the diagnostic")

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
    # The wording fix stands; the validator paragraph itself was rewritten again
    # when the A1 attribution landed, so the pins are on the current sentences.
    absent("T3", tex, "end-to-end equivalent to full prefill")
    present("T3", tex, "We attribute that residual rather than only reporting it")
    present("T3", tex, "near-ties resolved differently at bf16")

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
        stats = load_report("cluster_stats_audit3.json")
        key = "8B_4B_seed0"
        entry = stats.get("fourarm", {}).get(key)
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
        oa = stats.get("outaware", {}).get("phaseB_outaware_1.7B_0.6B_seed0", {})
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
        tax = load_report("error_taxonomy_selfdiag.json").get("students") or {}
        counts = tax.get("1.7B", {}).get("counts", {})
        if counts.get("wrong_entity") != 32 or counts.get("correct") != 24:
            FAILURES.append("A7: 1.7B taxonomy changed: %s" % counts)
        for bad in ("refusal_or_hedge", "repetition_loop", "no_content",
                    "hit_length_limit"):
            if counts.get(bad):
                FAILURES.append("A7: 1.7B failures include %s=%d"
                                % (bad, counts[bad]))

    # ------------------------------- A1: audit3 par.3 (GPU-verified validator)
    present("A1", tex, "first-token top-1 agreement $0.929$--$0.982$")
    present("A1", tex, "$0.001$--$0.005$ nats")
    present("A1", tex, "reproduces the reference bit for bit")
    present("A1", tex, "attention kernel path")
    present("A1", tex, "The same\nvalidator holds on the second domain")
    present("A1", tex, "first-token top-1\nagreement is $0.933$")
    absent("A1", tex, "end-to-end equivalent to full prefill")

    res = load_report("phaseB_evalcheck_residual_seed0.json").get("students") or []
    want = {"0.6B": (0.929, 0.0047, 1.344), "1.7B": (0.929, 0.0015, 0.875),
            "4B": (0.982, 0.0010, 0.969)}
    for entry in res:
        top1, kl, mx = want[entry["student"]]
        if not close(entry["first_top1_agreement"], top1, 0.001):
            FAILURES.append("A1: %s top-1 agreement changed (%.4f)"
                            % (entry["student"], entry["first_top1_agreement"]))
        if not close(entry["kl_median"], kl, 0.0002):
            FAILURES.append("A1: %s median KL changed (%.5f)"
                            % (entry["student"], entry["kl_median"]))
        if not close(entry["first_logits_max_abs_err_max"], mx, 0.001):
            FAILURES.append("A1: %s max logit error changed (%.4f)"
                            % (entry["student"], entry["first_logits_max_abs_err_max"]))
    attrib = load_report("phaseB_evalcheck_attrib.json").get("default_attn") or {}
    for key in ("roundtrip", "position_ids"):
        blk = attrib.get(key)
        if blk is None:
            FAILURES.append("A1: attribution block %s is missing" % key)
            continue
        if blk["max_abs_logit_err_max"] > 1e-6 or blk["kl_median"] > 1e-9:
            FAILURES.append("A1: attribution %s is no longer exact (%s)"
                            % (key, blk))
    squad = load_report("phaseB_evalcheck_squad.json").get("students") or []
    if not squad:
        FAILURES.append("A1: the SQuAD validator report carries no students")
    elif not close(squad[0]["first_top1_agreement"], 0.933, 0.002):
        FAILURES.append("A1: SQuAD validator top-1 changed (%.4f)"
                        % squad[0]["first_top1_agreement"])

    # ------------- A2: audit3 par.6/7 (20-epoch adapter, three seeds, row dumps)
    absent("A2", tex, "Both causality runs use a shorter schedule")
    present("A2", tex, "$\\mathbf{0.905{\\pm}0.083}$")
    present("A2", tex, "$\\mathbf{0.321{\\pm}0.129}$")
    present("A2", tex, "the mean margin $+0.095$ falls below the $0.10$")
    present("A2", tex, "\\textbf{Adapter training variance.}")
    # Report-derived margins (audit3 par.6/7). Mean margins: +0.750000
    # (1.7B_0.6B) and +0.095238 (8B_0.6B).
    want_margins = {"1.7B_0.6B": (0.714286, 0.678571, 0.857143),
                    "8B_0.6B": (0.178571, 0.071429, 0.035714)}
    want_joint = {"1.7B_0.6B": (0.857143, 0.857143, 1.0),
                  "8B_0.6B": (0.428571, 0.357143, 0.178571)}
    means = {}
    for pair, expected in want_margins.items():
        margins = []
        for seed in (0, 1, 2):
            rep = load_report(f"phaseB_adapter_causal20_{pair}_seed{seed}.json")
            if rep.get("epochs") != 20:
                FAILURES.append("A2: %s seed%d is not a 20-epoch run"
                                % (pair, seed))
            rows = (rep.get("rows_by_condition") or {}).get("adapter(joint)")
            if not rows:
                FAILURES.append("A2: %s seed%d has no adapter rows" % (pair, seed))
                continue
            mean = lambda a: sum(x[a + "_em"] for x in rows) / len(rows)
            if not close(mean("Joint"), want_joint[pair][seed], 0.001):
                FAILURES.append("A2: %s seed%d correct-KV EM changed (%.6f, "
                                "want %.6f)"
                                % (pair, seed, mean("Joint"),
                                   want_joint[pair][seed]))
            margins.append(mean("Joint") - max(mean("WrongJoint"),
                                              mean("RandKV"), mean("ZeroKV")))
        if len(margins) != 3:
            FAILURES.append("A2: %s has %d margins, expected 3"
                            % (pair, len(margins)))
            continue
        for seed, (got, want) in enumerate(zip(margins, expected)):
            if not close(got, want, 0.001):
                FAILURES.append("A2: %s seed%d margin changed (%+.6f, want %+.6f)"
                                % (pair, seed, got, want))
        avg = sum(margins) / 3
        means[pair] = avg
        if not close(avg, sum(expected) / 3, 0.001):
            FAILURES.append("A2: %s mean margin changed (%+.6f, want %+.6f)"
                            % (pair, avg, sum(expected) / 3))
    # Real bounds instead of the old `avg < 0.75`, which passed at exactly
    # 0.750000: the equal-depth gain must stay large, and the flagship margin
    # must stay inside the pre-registered noise band the paper reports.
    if means.get("1.7B_0.6B") is not None and means["1.7B_0.6B"] < 0.70:
        FAILURES.append("A2: equal-depth margin dropped to %+.3f"
                        % means["1.7B_0.6B"])
    if means.get("8B_0.6B") is not None and means["8B_0.6B"] >= 0.10:
        FAILURES.append("A2: flagship margin %+.3f now passes the 0.10 gate but "
                        "the paper still reports it as unresolved"
                        % means["8B_0.6B"])
    # The paper text must carry the report-derived margins: per-seed $+0.679$
    # (1.7B seed1), $+0.179$ (8B seed0) and the mean margin $+0.095$.
    for literal in ("$+0.679$", "$+0.179$", "$+0.095$"):
        present("A2", tex, literal)

    # ------------------- A3: audit3 par.8 (within-domain second-domain repair)
    present("A3", tex, "Mapper retrained on SQuAD")
    present("A3", tex, "Adapter retrained on SQuAD")
    present("A3", tex, "every\ntransfer arm---K-only, V-only under each mapper objective, and joint---answers")
    for split in (0, 1, 2):
        rep = load_report(f"phaseB_squadwithin_split{split}.json")
        for arm, block in (rep.get("summary") or {}).items():
            if arm != "Self" and block["EM"] > 0:
                FAILURES.append("A3: split%d arm %s is no longer zero (%.3f)"
                                % (split, arm, block["EM"]))
        for arm, block in (rep.get("summary_adapted") or {}).items():
            if arm != "adapted_Self" and block["EM"] > 0:
                FAILURES.append("A3: split%d adapted arm %s is no longer zero"
                                % (split, arm))

    # -------------------------- A4: audit3 par.11 (error budget, five variants)
    present("A4", tex, "\\label{tab:errorbudget}")
    # REVISION_PLAN4 T7 rewrote the caption to disclose the per-run range.
    present("A4", tex, "Five variants and six runs are too few for a $p$-value")
    budget = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            rep = load_report(f"phaseB_errorbudget_{pair}_seed{seed}.json")
            for name, block in (rep.get("summary") or {}).items():
                budget.setdefault(name, []).append(block)
    if len(budget) != 5:
        FAILURES.append("A4: expected five mapper variants, found %d" % len(budget))
    else:
        means = {n: {k: sum(b[k] for b in v) / len(v)
                     for k in ("e_raw_mean", "e_attn_mean", "EM")}
                 for n, v in budget.items()}
        if not (means["affine"]["e_raw_mean"] < means["outaware"]["e_raw_mean"]
                and means["affine"]["EM"] < means["outaware"]["EM"]):
            FAILURES.append("A4: the raw/consumption contrast no longer holds: %s"
                            % means)
        if means["outaware"]["e_attn_mean"] > 0.2:
            FAILURES.append("A4: outaware attention-output error rose to %.3f"
                            % means["outaware"]["e_attn_mean"])

    if FAILURES:
        print("FAIL: audit3 paper-side assertions")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("OK: all audit3 paper-side assertions hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
