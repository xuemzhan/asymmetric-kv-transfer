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


def _ranks_average(values) -> "np.ndarray":
    """1-based average ranks: ties share the mean of the ranks they span.

    W31: the pre-W31 helper used ordinal ranks (np.argsort(np.argsort(x))),
    which break ties by position in the input, so the same data in a different
    order gave a different correlation. Every rank correlation in the repository
    is an average-rank value now; see METRIC_CORRECTION.md section 14.
    """
    import numpy as np
    x = np.asarray(values, dtype=float)
    n = len(x)
    ranks = np.empty(n, dtype=float)
    if n == 0:
        return ranks
    order = np.argsort(x, kind="stable")
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j + 2)
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    import numpy as np
    rx = _ranks_average(x)
    ry = _ranks_average(y)
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
    # W32: the "fifteen calibration documents" concession is superseded by the
    # calibration-mix control, which excludes volume as the binding factor.
    present("T6", tex, "Fifteen synthetic documents")
    present("T6", tex, "calibration volume")

    # ------------------------------------------- T7: correlation disclosure
    present("T7", tex, "per-run range")
    # W32: the pooled-only phrasing moved behind the sweep; the five-variant
    # numbers are still printed in the Table 4 caption and in the Discussion.
    present("T7", tex, "five mapper variants and six")
    present("T7", tex, "$-0.267$")
    present("T7", tex, "$[-0.60,+0.05]$")
    present("T7", tex, "$[-1.00,-0.80]$")
    budget = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            rep = load_report("phaseB_errorbudget_%s_seed%d.json" % (pair, seed))
            if not rep:
                continue
            budget[(pair, seed)] = rep
            if rep.get("spearman", {}).get("rank_method") != "average":
                FAILURES.append("W31: %s spearman block does not declare "
                                "rank_method=average" % ("phaseB_errorbudget_%s_seed%d.json"
                                                         % (pair, seed)))
    if len(budget) == 6:
        names = ["raw", "affine", "outaware", "outaware-shuffled", "woaware"]
        pooled = {}
        for key in ("e_raw_mean", "e_attn_mean", "e_wo_mean", "EM"):
            pooled[key] = [sum(b["summary"][n][key] for b in budget.values())
                           / len(budget) for n in names]
        checks = {"e_raw_mean": (-0.10, -0.60, 0.05),
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

    # --------------------- W32: PART II experiments in the paper (A1/A2/A3)
    # REVISION_PLAN4 left these as TODO(audit4) placeholders until the GPU data
    # existed; every GPU number the paper now cites is recomputed here from
    # reports/ and compared with what the paper prints.
    present("A1", tex, "routing-aware")
    present("A1", tex, "\\label{tab:routingkey}")
    absent("A1", tex, "we measure it rather than intervening on it")
    present("A2", tex, "\\label{tab:sweep}")
    present("A2", tex, "figures/fig_sweep.pdf")
    present("A2", tex, "$198$")
    present("A3", tex, "calibration-mix control that holds the held-out documents")
    present("A3", tex, "Fifteen synthetic documents")

    routing = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            rep = load_report("phaseB_routingkey_%s_seed%d.json" % (pair, seed))
            if rep:
                routing[(pair, seed)] = rep
    if len(routing) != 6:
        FAILURES.append("A1: expected six routingkey reports, found %d" % len(routing))
    else:
        table = {("1.7B_0.6B", "K-affine"): (-0.82, 0.030, 0.167, 0.700),
                 ("1.7B_0.6B", "K-routing"): (3.79, 0.000, 0.139, 0.872),
                 ("1.7B_0.6B", "K-routing-shuf"): (3.85, 0.006, 0.205, 0.842),
                 ("1.7B_0.6B", "K-raw"): (0.31, 0.030, 0.170, 0.745),
                 ("8B_0.6B", "K-affine"): (2.62, 0.000, 0.254, 0.581),
                 ("8B_0.6B", "K-routing"): (2.26, 0.030, 0.153, 0.839),
                 ("8B_0.6B", "K-routing-shuf"): (1.35, 0.042, 0.214, 0.813),
                 ("8B_0.6B", "K-raw"): (-0.98, 0.000, 0.773, 0.032)}
        for (pair, arm), (w_dll, w_em, w_tv, w_top1) in table.items():
            reps = [routing[(pair, s)] for s in (0, 1, 2)]
            got = (sum(r["summary"][arm]["delta_vs_self"] for r in reps) / 3.0,
                   sum(r["summary"][arm]["EM"] for r in reps) / 3.0,
                   sum(r["routing"][arm]["tv_mean"] for r in reps) / 3.0,
                   sum(r["routing"][arm]["top1_mean"] for r in reps) / 3.0)
            for name, g, w, tol in (("dLL", got[0], w_dll, 0.005),
                                    ("EM", got[1], w_em, 0.0005),
                                    ("routing TV", got[2], w_tv, 0.0005),
                                    ("top-1", got[3], w_top1, 0.0005)):
                if not close(g, w, tol):
                    FAILURES.append("A1: %s %s %s is %.4f, the paper prints %.4f"
                                    % (pair, arm, name, g, w))
        for pair, want_routing, want_any in (("1.7B_0.6B", 0.0000, 0.0357),
                                             ("8B_0.6B", 0.0536, 0.0893)):
            reps = [routing[(pair, s)] for s in (0, 1, 2)]
            mx_routing = max(r["summary"]["K-routing"]["EM"] for r in reps)
            mx_any = max(r["summary"][a]["EM"] for r in reps
                         for a in r["summary"] if a != "Self")
            if not close(mx_routing, want_routing, 0.0002):
                FAILURES.append("A1: %s worst routing-arm EM is %.4f, the paper's "
                                "floor bound says %.4f" % (pair, mx_routing, want_routing))
            if not close(mx_any, want_any, 0.0002):
                FAILURES.append("A1: %s worst key-arm EM is %.4f, the paper says %.4f"
                                % (pair, mx_any, want_any))
        shuf = sum(routing[("8B_0.6B", s)]["summary"]["K-routing-shuf"]["EM"]
                   for s in (0, 1, 2)) / 3.0
        real = sum(routing[("8B_0.6B", s)]["summary"]["K-routing"]["EM"]
                   for s in (0, 1, 2)) / 3.0
        if shuf < real:
            FAILURES.append("A1: the 8B shuffled control (%.4f) is now below the "
                            "routing mapper (%.4f), breaking the paper's "
                            "\"not worse\" claim" % (shuf, real))
        for (pair, seed), rep in routing.items():
            if abs(rep.get("selfcheck_max_abs_diff_w1_vs_affine", 1.0)) > 1e-9:
                FAILURES.append("A1: %s seed %d unit-weight self-check is not exact"
                                % (pair, seed))

    sweep = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            rep = load_report("phaseB_mappersweep_%s_seed%d.json" % (pair, seed))
            if rep:
                sweep[(pair, seed)] = rep
    if len(sweep) != 6:
        FAILURES.append("A2: expected six mappersweep reports, found %d" % len(sweep))
    else:
        for pair, want in (("1.7B_0.6B", (0.918, -0.964, -0.962)),
                           ("8B_0.6B", (0.877, -0.919, -0.918))):
            pts = {k: [c[k] for s in (0, 1, 2) for c in sweep[(pair, s)]["configs"]]
                   for k in ("e_raw", "e_attn", "e_wo")}
            em = [c["EM"] for s in (0, 1, 2) for c in sweep[(pair, s)]["configs"]]
            for k, w in zip(("e_raw", "e_attn", "e_wo"), want):
                got = spearman(pts[k], em)
                if not close(got, w, 0.001):
                    FAILURES.append("A2: %s pooled-99 rho(%s) is %.4f, the paper "
                                    "prints %.3f" % (pair, k, got, w))
        pts = {k: [c[k] for (pair, s) in sweep for c in sweep[(pair, s)]["configs"]]
               for k in ("e_raw", "e_attn", "e_wo")}
        em = [c["EM"] for (pair, s) in sweep for c in sweep[(pair, s)]["configs"]]
        if len(em) != 198:
            FAILURES.append("A2: pooled point count is %d, the paper says 198" % len(em))
        for k, w in (("e_raw", -0.267), ("e_attn", -0.799), ("e_wo", -0.823)):
            got = spearman(pts[k], em)
            if not close(got, w, 0.001):
                FAILURES.append("A2: pooled-198 rho(%s) is %.4f, the paper prints %.3f"
                                % (k, got, w))
        for (pair, seed), rep in sweep.items():
            got = spearman([c["e_raw"] for c in rep["configs"]],
                           [c["EM"] for c in rep["configs"]])
            if got is None or got <= 0.0:
                FAILURES.append("A2: %s seed %d raw rho is %r, so the paper's "
                                "'positively related within every run' claim fails"
                                % (pair, seed, got))

    for cell, (pat, want_max_em, want_adapted) in (
            ("C1", ("phaseB_squadwithin_split%d.json", 0.000, 0.489)),
            ("C2", ("phaseB_squadwithin_synthetic_split%d.json", 0.000, 0.556)),
            ("C3", ("phaseB_squadwithin_squad_synthetic_split%d.json", 0.067, 0.422))):
        reps = [load_report(pat % s) for s in (0, 1, 2)]
        if not all(reps):
            FAILURES.append("A3: %s reports missing" % cell)
            continue
        held = [r["summary"]["Self"]["EM"] for r in reps]
        if not close(sum(held) / 3.0, 0.311, 0.001):
            FAILURES.append("A3: %s held-out Self is %.4f, the paper says 0.311"
                            % (cell, sum(held) / 3.0))
        adapted = [r["summary_adapted"]["adapted_Self"]["EM"] for r in reps]
        if not close(sum(adapted) / 3.0, want_adapted, 0.001):
            FAILURES.append("A3: %s adapted Self is %.4f, the paper prints %.3f"
                            % (cell, sum(adapted) / 3.0, want_adapted))
        per_split = [max(v["EM"] for k, v in r["summary"].items() if k != "Self")
                     for r in reps]
        if not close(max(per_split), want_max_em, 0.001):
            FAILURES.append("A3: %s best transfer-arm EM per split %s, the paper "
                            "prints %.3f"
                            % (cell, [round(x, 3) for x in per_split], want_max_em))
        if cell == "C3" and not close(sum(per_split) / 3.0, 0.044, 0.001):
            FAILURES.append("A3: C3 mean-over-splits EM is %.4f, the paper says 0.044"
                            % (sum(per_split) / 3.0))

    # W32 re-check: the arXiv metadata mirror must follow the paper -- abstract
    # sentences, the figure list, and the table/figure counts of the comments field.
    present("META", meta, "routing-aware key mapper that shrinks routing divergence")
    present("META", meta, "figures/fig_sweep.pdf")
    present("META", meta, "85 documents mixing both still fail")
    present("META", meta, "Sweeping the mapper")
    present("META", meta, "over 198 configurations")
    present("META", meta, "%d tables, %d figures" % (tex.count("\\begin{table}"),
                                                      tex.count("\\begin{figure}")))

    # ------------------- W31: rank method, A2 report integrity (PART II A2/A4)
    # The stored A2 correlations must be average-rank values recomputed from the
    # very configs stored in the same file, the gate must carry the disjointness
    # verdict, and the aggregate must equal the pooled recomputation.
    a2_runs = {}
    for pair in ("1.7B_0.6B", "8B_0.6B"):
        for seed in (0, 1, 2):
            name = "phaseB_mappersweep_%s_seed%d.json" % (pair, seed)
            rep = load_report(name)
            if not rep:
                continue
            a2_runs[(pair, seed)] = rep
            if rep.get("rank_method") != "average":
                FAILURES.append("W31: %s does not declare rank_method=average" % name)
            gate = rep.get("gate", {})
            if "bootstrap_intervals_disjoint" not in gate:
                FAILURES.append("W31: %s gate lacks bootstrap_intervals_disjoint" % name)
            elif gate.get("branch") != "strong statement preserved":
                FAILURES.append("W31: %s gate branch is %r, not the disjoint branch"
                                % (name, gate.get("branch")))
            for k in ("e_raw", "e_attn", "e_wo"):
                got = spearman([c[k] for c in rep["configs"]],
                               [c["EM"] for c in rep["configs"]])
                if not close(got, rep.get("rho", {}).get(k), 1e-9):
                    FAILURES.append("W31: %s rho[%s]=%r is not the average-rank value "
                                    "of its own configs (%.6f)"
                                    % (name, k, rep.get("rho", {}).get(k), got))
    if len(a2_runs) == 6:
        agg = load_report("phaseB_mappersweep_aggregate.json")
        if not agg:
            FAILURES.append("W31: the A2 aggregate report is missing")
        else:
            order = []
            for lab in agg.get("running", []):
                rid = lab.split("/")[0]
                if rid not in order:
                    order.append(rid)
            by_id = {"%s_s%d" % (p, s): r for (p, s), r in a2_runs.items()}
            if sorted(order) != sorted(by_id):
                FAILURES.append("W31: aggregate `running` names runs %r" % order)
            else:
                rebuilt = [("%s/%s" % (rid, c["key"]), c) for rid in order
                           for c in by_id[rid]["configs"]]
                if [lab for lab, _ in rebuilt] != agg.get("running"):
                    FAILURES.append("W31: aggregate point order does not match its "
                                    "stored `running` labels")
                for k in ("e_raw", "e_attn", "e_wo"):
                    got = spearman([c[k] for _, c in rebuilt],
                                   [c["EM"] for _, c in rebuilt])
                    if not close(got, agg.get("rho_pooled", {}).get(k), 1e-9):
                        FAILURES.append("W31: aggregate rho_pooled[%s]=%r is not the "
                                        "average-rank value of the stored points (%.6f)"
                                        % (k, agg.get("rho_pooled", {}).get(k), got))
        trail = load_report("rank_correlation_recompute.json")
        if not trail:
            FAILURES.append("W31: reports/rank_correlation_recompute.json is missing")
        elif not (trail.get("migration", {}).get("n_changed_fields") or 0) > 0:
            FAILURES.append("W31: the rank-method migration trail records no changes")

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
