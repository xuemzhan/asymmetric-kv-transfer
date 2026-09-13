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
    present("X6", tex, "the failure is a compatibility failure, and it is worth separating two")
    present("X7", tex, "The failure lies in how the receiving model reads the state")

    # ----------------------------------------------------------------- PART II
    # B1 (reports/phaseB_evalcheck_seed0.json) - audit2 par.3, evaluator validator.
    absent("B1", tex, "six-item")
    absent("B1", tex, "was not retained")
    present("B1", tex, "normalized-EM agreement $0.929$--$0.982$")
    present("B1", tex, "complete $56$-item Self test set of three student sizes")
    present("B1", tex, "logits is $1.34$, $0.88$, and $0.97$")
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
    present("B2", tex, "we report the flagship causality as")
    present("B2", tex, "$0.304$ for correct KV against $0.143$--$0.161$")

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
    present("C4", tex, "\\textbf{Cross-domain repair.}")

    # C2 (reports/phaseB_selfdiag_seed0.json) - par.12, 1.7B Self anomaly.
    present("C2", tex, "formatting artifact we can see")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
