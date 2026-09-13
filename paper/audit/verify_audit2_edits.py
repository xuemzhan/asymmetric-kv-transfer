# -*- coding: utf-8 -*-
"""One-to-one checks for the audit2 paper-side edits.

Every assertion maps to an audit2 finding via the traceability table in
paper/audit/REVISION_PLAN2.md. Run from anywhere:

    python paper/audit/verify_audit2_edits.py

Exit code 0 = all assertions hold. This guard exists because a single withdrawn
claim appeared in four places (Abstract, Introduction, Experimental Setup,
Results) plus a table and a figure; editing one site and missing another leaves
the paper self-contradictory.

PART I (W19b) covers the edit-only findings. PART II (W20) covers the findings
that needed new GPU runs (B1--B4, C1, C4); each new number asserted below is
traceable to a reports/*.json file named in the assertion tag.
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "paper", "main.tex")
FIG = os.path.join(ROOT, "paper", "figures", "gen_paper_figures.py")

FAILURES: list[str] = []


def absent(tag: str, text: str, needle: str) -> None:
    n = text.count(needle)
    if n:
        FAILURES.append("%s: %r must be absent, found %d" % (tag, needle, n))


def present(tag: str, text: str, needle: str) -> None:
    if needle not in text:
        FAILURES.append("%s: %r must be present" % (tag, needle))


def main() -> int:
    tex = io.open(TEX, encoding="utf-8").read()
    fig = io.open(FIG, encoding="utf-8").read()

    # ------------------------------------------------------------------ PART I
    # audit2 par.4 - the permutation-invariant control is withdrawn document-wide.
    absent("A1", tex, "token-shuffled")
    absent("A5", tex, "Token-shuffled")
    present("A1", tex, "against $0.899\\pm0.010$ for the student's own cache")
    present("A4", tex, "attention is invariant to a shared")
    present("A4", tex, "is not a content control")
    absent("A6", fig, "SHUF_STUDENT")

    # audit2 par.16 - the adapter trains on gold answers; it is not label-free.
    absent("A2", tex, "trained only on")
    present("A2", tex, "cross-entropy on calibration answers under injected teacher KV, restores")
    present("A2", tex, "cross-entropy on calibration answers under injected teacher KV while the")

    # audit2 par.8 - one adapter per training condition, shared across the four arms.
    present("A7", tex, "trains its own adapter from scratch")

    # audit2 par.14 - no unfinished-experiment admission.
    absent("A9", tex, "still running")

    # audit2 par.5 - state-space vs consumption-space compatibility.
    present("X5", tex, "The practical message separates two levels of compatibility")
    present("X5", tex, "\\emph{state space}, matching raw KV representations is not sufficient")
    present("X5", tex, "\\emph{consumption space}, alignment is what matters")
    # X6 originally pinned "the failure is a compatibility failure, and it is worth
    # separating two levels". audit3 (par.16) replaced that sentence with the
    # four-layer statement, so the guard for the new wording lives in
    # verify_audit3_edits.py (tag T1b); the state/consumption split it encoded is
    # still asserted here.
    present("X6", tex, "\\emph{(ii) Value transfer.}")
    present("X6", tex, "Alignment in raw representation space is\ninsufficient")
    present("X7", tex, "The failure lies in how the receiving model reads the state")

    # ----------------------------------------------------------------- PART II
    # B1 (reports/phaseB_evalcheck_seed0.json) - audit2 par.3, evaluator validator.
    absent("B1", tex, "six-item")
    absent("B1", tex, "was not retained")
    present("B1", tex, "normalized-EM agreement $0.929$--$0.982$")
    present("B1", tex, "complete $56$-item Self test set of three student sizes")
    # audit3/A1 rewrote the validator paragraph and kept the same three numbers
    # with different wording; the A1 assertions in verify_audit3_edits.py pin
    # them against the residual report.
    present("B1", tex, "the largest absolute logit error is $1.344$, $0.875$,")
    present("B1", tex, "\\label{fig:evalcheck}")
    present("B1", fig, "def fig_evalcheck():")
    present("B1", fig, "fig_evalcheck()")

    # B4 (reports/phaseB_controls_8B_0.6B_seed{0_fixed,1,2}.json) - par.4.
    present("B4", tex, "Breaking the key--value correspondence does register")
    present("B4", tex, "to $0.000$ in all three seeds, and the mirror control")
    present("B4", tex, "Shuffled K (student)")
    present("B4", tex, "Same permutation, K and V (student)")
    present("B4", fig, "SHUFFLES = [")
    absent("B4", tex, "we do not report a same-permutation arm")

    # B3 (reports/phaseB_adapter_*_o_proj_seed*.json) - par.7, adapted Self frame.
    absent("B3", tex, "the Self column was not retained")
    present("B3", tex, "measured against an \\emph{adapted} Self of $0.940\\pm0.041$")
    present("B3", tex, "$0.976{\\pm}0.021$")

    # B2 (reports/phaseB_adapter_causal_*.json) - par.6, content causality.
    present("B2", tex, "\\label{tab:causal}")
    # audit3/A2 replaced the 10-epoch single-seed causality with the 20-epoch
    # three-seed frozen-adapter test; the current numbers and the "unresolved"
    # verdict are pinned by tag A2 in verify_audit3_edits.py.
    absent("B2", tex, "Both causality runs use a shorter schedule")

    # C1 (reports/phaseB_alignment_repaired_1.7B_0.6B_seed0.json) - par.9.
    absent("C1", tex, "Layer alignment alone does not explain the failure")
    absent("C1", tex, "does not destroy transfer")
    present("C1", tex, "\\subsection{Layer alignment matters once the consumer reads the state}")
    present("C1", tex, "V-only EM falls to $0.214$")
    present("C1", tex, "It is a floor effect of the mapper")
    present("C1", tex, "Alignment is a precondition that the ordinary affine mapper")
    present("C1", fig, "OUTAWARE_PROP_SEEDS")

    # C4 (reports/phaseB_squad_{outaware,adapter}_seed0.json) - par.13.
    present("C4", tex, "\\label{tab:squadrepair}")
    present("C4", tex, "V-only, and joint EM at $0.000$ and $0.033$")
    # audit3/A3 merged the cross-domain bullet into the task-distribution
    # limitation and added the within-domain SQuAD result (tag A3).
    present("C4", tex, "compatibility being bound to the task distribution")

    # C2 (reports/phaseB_selfdiag_seed0.json) - par.12, 1.7B Self anomaly.
    # audit3 (par.13, tag A7) replaced this sentence with a rule-based taxonomy
    # of the archived generations; the stronger wording is asserted here and the
    # per-class counts are pinned in verify_audit3_edits.py.
    present("C2", tex, "not a formatting or extraction artifact")
    present("C2", tex, "Classifying the archived")

    # Stale-claim sweep: strings that described the superseded reading.
    for stale in ("$0.018$ & $89\\%$",
                  "was run at seed 0 only and was not replicated across",
                  "is the most effective and the most content-specific",
                  "the post-hoc $(8, 12)$ pair reaches $0.22$"):
        absent("STALE", tex, stale)

    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("OK: all audit2 paper-side assertions hold")
    # ============================================================
    # PART III (W21) - end-to-end self-review corrections.
    # Each assertion pins a number that was wrong or inconsistent in the
    # W20b text; sources are named next to the assertion.
    # ============================================================

    # E1 (reports/phaseB_fourarm_seed{0,1,2}.json): 4B->0.6B V-only dLL std is
    # 0.0547 over three seeds, i.e. 0.05 at two decimals. tab:factorial already
    # printed 0.05; tab:main printed 0.06.
    present("E1", tex, "$-1.22{\\pm}0.05$")
    absent("E1", tex, "$-1.22{\\pm}0.06$")

    # E2 (reports/phaseB_adapter_1.7B_0.6B_{o_proj_,}seed{0,1,2}.json): four
    # 1.7B adapter cells were mis-rounded; the 8B block was already correct.
    present("E2", tex, "$0.113{\\pm}0.052$ & $0.000$")
    present("E2", tex, "$0.202{\\pm}0.203$")
    present("E2", tex, "$0.560{\\pm}0.021$ & $0.345{\\pm}0.055$")
    present("E2", tex, "yields $0.202/0.345/0.179$")
    present("E2", tex, "intermediate, at $0.560/0.345/0.458$")
    absent("E2", tex, "$0.113{\\pm}0.051$")
    absent("E2", tex, "$0.559{\\pm}0.020$")
    absent("E2", tex, "$0.203{\\pm}0.203$")

    # E3 (reports/phaseB_alignment_8B_0.6B_seed{0,1,2}.json): V-only EM is 0.000
    # in 17 of 18 map x seed cells (learned top-1 V selection, seed 2 = 0.107)
    # and K-only peaks at 0.179 (learned top-1 V selection, seed 1).
    present("E3", tex, "except one\nlearned-selection cell ($0.107$), and K-only never exceeds $0.179$")
    absent("E3", tex, "for every map and seed, and K-only\nnever exceeds $0.107$")

    # E4 (reports/phaseB_controls_{1.7B,8B}_0.6B_*.json): the 1.7B control suite,
    # its identity V-only control and its random-KV control all have three seeds;
    # only the repaired-regime scrambling, the projection ablation and the two
    # causality runs are seed 0.
    present("E4", tex, "Four results are seed 0 only")
    present("E4", tex, "were both run for all three seeds")
    absent("E4", tex, "the identity\n  V-only control, the random-KV control")

    # E5 (reports/phaseB_evalcheck_seed0.json): max |first-token logit error| is
    # 1.34375 / 0.875 / 0.96875; the paper now states one value everywhere.
    absent("E5", tex, "logit error at most\n$1.35$")
    absent("E5", tex, "to within $1.34$ in absolute value")
    absent("E5", tex, "error $\\le1.34$")
    present("E5", tex, "first-token logit error at most\n$1.344$")

    # E6 (reports/phaseB_mechanism_{pair}_seed{0,1,2}.json): the routing
    # total-variation in the text is the three-seed mean 0.167 / 0.251, matching
    # the three-seed means printed for top-1 agreement and cosine.
    present("E6", tex, "$0.167$ and $0.251$")
    absent("E6", tex, "$0.165$")
    absent("E6", tex, "$0.245$")

    # E7 (reports/archive/g0_seed{0,1,2}.json + reports/g0_v2_summary.json): the
    # pre-registered gate was defined on the 42/14 split and re-run at 70/56.
    present("E7", tex, "on the earlier calibration/evaluation split ($42$/$14$)")
    present("E7", tex, "re-ran its likelihood clause")
    present("E7", tex, "($\\dll = -0.045$, $-0.219$, $-0.415$")

    # E8 (reports/phaseB_squad_{seed0,outaware_seed0}.json): the +4.62 vs +1.05
    # comparison is seed 0 on both sides; the three-seed default mean is +0.97.
    present("E8", tex, "$+4.62$ at seed 0, against $+1.05$ for the default protocol")
    absent("E8", tex, "$+4.62$ against $+1.05$ under the default")

    # E9 (Table adapter): the adapted Self is 0.940 +- 0.041 on both pairs, so
    # the unadapted 0.899 is not a safe ceiling for either pair.
    present("E9", tex, "belongs to the adapter rather\nthan to the transfer")
    absent("E9", tex, "understated\nthe gap on the flagship pair and overstated it on no pair")

    return 0


if __name__ == "__main__":
    sys.exit(main())
