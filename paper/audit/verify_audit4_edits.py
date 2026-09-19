# -*- coding: utf-8 -*-
"""One-to-one checks for the audit4 paper-side edits (REVISION_PLAN4 PART I).

Run from anywhere:

    python paper/audit/verify_audit4_edits.py

Exit code 0 = all assertions hold. The guard exists because audit4's four
findings that cost nothing to fix are all of the "a reviewer can check this in
the repository" kind, and three of them were contradictions inside the paper:

  T1  the title says "compatible consumer" while the paper's claim is now
      functional compatibility (and the old title leaked into README /
      arxiv metadata);
  T2  the contribution list still had the three-item consumer-side framing;
  T3  the four arms had no algebraic frame;
  T4  the factorial (likelihood) table and the cost paragraph moved to
      appendices, and the cost paragraph's mapper parameterisation disagreed
      with the per-(layer, head) mapper defined in the Method;
  T5  "the joint arm is the weakest arm everywhere" is refuted by Table 1
      itself, and Related Work still blamed a "standard evaluation" for a
      defect in our own earlier implementation;
  T6  the task-conditioned naming must carry its qualifiers;
  T7  the rank correlations were reported only as pooled values, while the
      report JSONs carry a run-to-run range the paper did not disclose.

The numeric assertions read reports/phaseB_errorbudget_*.json and recompute the
pooled and per-run correlations, so the text cannot drift away from the reports.
"""
from __future__ import annotations

import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "paper", "main.tex")
ARXIV_TEX = os.path.join(ROOT, "paper", "arxiv_submission", "main.tex")
README = os.path.join(ROOT, "README.md")
ARXIV_META = os.path.join(ROOT, "paper", "arxiv_metadata.md")
REPORTS = os.path.join(ROOT, "reports")
ARCHIVE = os.path.join(ROOT, "paper", "archive")

OLD_TITLE = "Needs a Compatible Consumer"
NEW_TITLE = "Needs Functional Compatibility"
V1_DOCS = ("BRIEF.md", "project_context.md", "architecture.md",
           "v3_data_baseline.md")

FAILURES: list[str] = []


def absent(tag: str, text: str, needle: str) -> None:
    n = text.count(needle)
    if n:
        FAILURES.append("%s: %r must be absent, found %d" % (tag, needle, n))


def present(tag: str, text: str, needle: str) -> None:
    if needle not in text:
        FAILURES.append("%s: %r must be present" % (tag, needle))


def read(path: str) -> str:
    """Read a guard input; a missing file is a FAILURE, not a crash."""
    try:
        return io.open(path, encoding="utf-8").read()
    except OSError as exc:
        FAILURES.append("FILE: %s could not be read (%s)" % (path, exc))
        return ""


def load_report(name: str) -> dict:
    """Load reports/<name>; missing or corrupt is a FAILURE, not a crash.

    A guard that raises cannot be told apart from a guard that failed, so the
    artifact is recorded in FAILURES and an empty dict is returned.
    """
    path = os.path.join(REPORTS, name)
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


def close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol


def spearman(x, y) -> float:
    import numpy as np
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        raise ValueError("degenerate rank vector")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    tex = read(TEX)
    arxiv = read(ARXIV_TEX)
    readme = read(README)
    meta = read(ARXIV_META)

    # ------------------------------------------------- T1: title and metadata
    present("T1", tex, "\\title{Cross-Model KV Transfer Needs Functional")
    present("T1", tex, "pdftitle={Cross-Model KV Transfer Needs Functional")
    present("T1", arxiv, "\\title{Cross-Model KV Transfer Needs Functional")
    present("T1", readme, NEW_TITLE)
    present("T1", meta, NEW_TITLE)
    for name, text in (("main.tex", tex), ("arxiv_submission/main.tex", arxiv),
                       ("README.md", readme), ("arxiv_metadata.md", meta)):
        absent("T1", text, OLD_TITLE)
    absent("T1", tex, "Consumption-Side Fix")

    # ---------------------------------------------------- T2: contributions
    for item in ("\\textbf{An evaluation principle.}",
                 "\\textbf{A functional compatibility decomposition.}",
                 "\\textbf{Consumer-space alignment.}",
                 "\\textbf{Repairability and its boundary.}"):
        present("T2", tex, item)
    absent("T2", tex, "\\textbf{A consumer-side diagnosis.}")
    absent("T2", tex, "\\textbf{A consumption-side fix.}")
    # The general principle and our own implementation defect must stay split.
    present("T2", tex, "This claim holds independently of")
    present("T2", tex, "we did not audit third-party implementations")

    # ------------------------------------- T3: compatibility decomposition
    present("T3", tex, "\\subsection{Functional compatibility decomposition}")
    present("T3", tex, "\\label{sec:decomp}")
    present("T3", tex, "\\label{eq:decomp}")
    present("T3", tex, "routing term")
    present("T3", tex, "consumption term")
    present("T3", tex, "not a claim that task performance decomposes additively")
    present("T3", tex, "does not say which\nterm limits task performance")

    # ------------------------------------------- T4: appendices and cost fix
    absent("T4", tex, "2050")
    absent("T4", tex, "\\subsection{Cost}")
    present("T4", tex, "\\appendix")
    present("T4", tex, "\\label{app:factorial}")
    present("T4", tex, "\\label{app:cost}")
    present("T4", tex, "\\label{tab:factorial}")
    present("T4", tex, "near $260$ tokens")
    present("T4", tex, "$7.40$M")
    # The appendices must follow the bibliography, not interrupt the main text.
    if "\\appendix" not in tex or "\\end{thebibliography}" not in tex:
        FAILURES.append("T4: main.tex must carry both \\appendix and the "
                        "bibliography")
    elif tex.index("\\appendix") < tex.index("\\end{thebibliography}"):
        FAILURES.append("T4: the appendix block must follow the bibliography")
        FAILURES.append("T4: the appendix block must follow the bibliography")
    # Recompute the heuristic from the per-(layer, head) mapper definition.
    params = 2 * 28 * 8 * (128 * 128 + 128)          # K + V, 28 layers, 8 heads
    mib = params * 4 / 1024 / 1024                   # fp32
    kv_kib = 28 * 2 * 8 * 128 * 2 / 1024             # fp16, K + V, per token
    cross = mib / (kv_kib / 1024)
    if not close(params / 1e6, 7.40, 0.005):
        FAILURES.append("T4: per-head mapper parameter count changed (%d)" % params)
    if not close(mib, 28.2, 0.05):
        FAILURES.append("T4: mapper footprint changed (%.2f MiB)" % mib)
    if not close(kv_kib, 112.0, 0.01):
        FAILURES.append("T4: KV bytes per token changed (%.1f KiB)" % kv_kib)
    if not close(cross, 260, 2):
        FAILURES.append("T4: crossover changed (%.1f tokens)" % cross)

    # ------------------------------------------------------ T5: wording fixes
    absent("T5", tex, "weakest arm everywhere")
    absent("T5", tex, "standard evaluation can report transfer")
    absent("T5", tex, "we show that the\nstandard evaluation")
    present("T5", tex, "both forms of\ncompatibility and is therefore consistently fragile")
    present("T5", tex, "our legacy task\nevaluator could report apparent transfer")
    present("T5", tex, "teacher-forced likelihood alone is insufficient evidence of")

    # ------------------------------------------------ T6: naming discipline
    present("T6", tex, "\\emph{task-conditioned} functional compatibility")
    present("T6", tex, "task-conditioned rather than a property of the model pair")
    present("T6", tex, "SQuAD supplies only fifteen calibration")
    present("T6", tex, "calibration volume")

    # ------------------------------------------- T7: correlation disclosure
    present("T7", tex, "per-run range")
    present("T7", tex, "pooled over five variants and six")
    present("T7", tex, "$[-0.60,-0.10]$")
    present("T7", tex, "$[-1.00,-0.80]$")
    budget = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            rep = load_report("phaseB_errorbudget_%s_seed%d.json" % (pair, seed))
            if not rep:
                continue
            budget[(pair, seed)] = rep
    if len(budget) == 6:
        names = ["raw", "affine", "outaware", "outaware-shuffled", "woaware"]
        pooled = {}
        for key in ("e_raw_mean", "e_attn_mean", "e_wo_mean", "EM"):
            pooled[key] = [sum(b["summary"][n][key] for b in budget.values())
                           / len(budget) for n in names]
        checks = {"e_raw_mean": (-0.10, -0.60, -0.10),
                  "e_attn_mean": (-1.00, -1.00, -0.80),
                  "e_wo_mean": (-0.80, -1.00, -0.80)}
        for key, (want_pooled, want_lo, want_hi) in checks.items():
            got = spearman(pooled[key], pooled["EM"])
            if not close(got, want_pooled, 0.005):
                FAILURES.append("T7: pooled rho for %s changed (%.3f)"
                                % (key, got))
            runs = [spearman([b["summary"][n][key] for n in names],
                             [b["summary"][n]["EM"] for n in names])
                    for b in budget.values()]
            if not close(min(runs), want_lo, 0.005) or not close(max(runs), want_hi, 0.005):
                FAILURES.append("T7: per-run rho range for %s changed (%s)"
                                % (key, [round(r, 2) for r in runs]))

    # --------------------------------------------------- T9: stale v1 docs
    for name in V1_DOCS:
        if os.path.exists(os.path.join(ROOT, "paper", name)):
            FAILURES.append("T9: paper/%s must be archived" % name)
        archived = os.path.join(ARCHIVE,
                                name[:-3] + "_v1_stale.md")
        if not os.path.isfile(archived):
            FAILURES.append("T9: %s must exist" % archived)
        elif "STALE" not in read(archived)[:400]:
            FAILURES.append("T9: %s needs a STALE header" % archived)

    # ---------------------------- T10: labels exist, no hard-coded sections
    hard = re.findall(r"Section~[0-9]", tex)
    if hard:
        FAILURES.append("T10: hard-coded section references remain: %s"
                        % sorted(set(hard)))
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        FAILURES.append("T10: duplicate labels: %s" % dupes)
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    undefined = sorted(refs - set(labels))
    if undefined:
        FAILURES.append("T10: references without a label: %s" % undefined)
    for needed in ("sec:decomp", "app:factorial", "app:cost", "sec:repaired-layer"):
        if needed == "sec:repaired-layer":
            # must NOT exist: an earlier draft of the appendix pointed at it
            if needed in labels:
                FAILURES.append("T10: stale label %s" % needed)
        elif needed not in labels:
            FAILURES.append("T10: missing label %s" % needed)

    # ------------------------------------------- arxiv bundle stays in sync
    if tex != arxiv:
        FAILURES.append("SYNC: paper/main.tex and arxiv_submission/main.tex differ")

    if FAILURES:
        print("FAIL: audit4 paper-side assertions")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("OK: all audit4 paper-side assertions hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
