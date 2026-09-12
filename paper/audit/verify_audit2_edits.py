# -*- coding: utf-8 -*-
"""One-to-one checks for the audit2 paper-side edits (REVISION_PLAN2.md, PART II-A).

Every assertion maps to an audit2 finding via the traceability table in
paper/audit/REVISION_PLAN2.md. Run from anywhere:

    python paper/audit/verify_audit2_edits.py

Exit code 0 = all assertions hold. This guard exists because a single withdrawn
claim appeared in four places (Abstract, Introduction, Experimental Setup,
Results) plus a table and a figure; editing one site and missing another leaves
the paper self-contradictory.
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

    # audit2 \u00a74 - the permutation-invariant control is withdrawn document-wide.
    absent("A1", tex, "token-shuffled")
    absent("A5", tex, "Token-shuffled")
    present("A1", tex, "against $0.899\\pm0.010$ for the student's own cache; the same evaluator")
    present("A1", tex, "against $0.899\\pm0.010$ for the student's own")
    present("A4", tex, "reordering the token axis of $K$ and $V$ by the \\emph{same}")
    present("A4", tex, "attention is invariant to a shared")
    present("A6", tex, "student's own cache scores $0.899$")
    absent("A6", fig, "SHUF_STUDENT")

    # audit2 \u00a716 - the adapter trains on gold answers; it is not label-free.
    absent("A2", tex, "trained only on")
    present("A2", tex, "cross-entropy on calibration answers under injected teacher KV, restores")
    present("A2", tex, "cross-entropy on calibration answers under injected teacher KV while the")

    # audit2 \u00a79 - layer alignment alone does not explain the failure.
    absent("A3", tex, "is not the controlling")
    present("A3", tex, "so layer alignment alone does not explain the failure")
    present("A3", tex, "Layer alignment alone does not explain the failure: on the equal-depth")
    present("A8", tex, "\\subsection{Layer alignment alone does not explain the failure}")

    # audit2 \u00a78 - one adapter per training condition, shared across the four arms.
    present("A7", tex, "trains its own adapter from scratch")

    # audit2 \u00a714 - no unfinished-experiment admission.
    absent("A9", tex, "still running")
    present("A9", tex, "This ablation is preliminary (seed 0, two of")

    # audit2 \u00a75 - state-space vs consumption-space compatibility.
    present("X5", tex, "The practical message separates two levels of compatibility")
    present("X5", tex, "\\emph{state space}, matching raw KV representations is not sufficient")
    present("X5", tex, "\\emph{consumption space}, alignment is what matters")
    present("X6", tex, "the failure is a compatibility failure, and it is worth separating two")
    present("X7", tex, "The failure lies in how the receiving model reads the state")

    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("OK: all audit2 paper-side assertions hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())