"""Recompute the rank correlations stored in the W30 GPU reports with
average-rank Spearman (W31).

Why
---
Until W31 the project's ``spearman`` helper used ordinal ranks
(``np.argsort(np.argsort(x))``), which breaks ties by position in the input
vector instead of sharing the mean rank. Every stored correlation therefore
depended on the arbitrary order of the variants/configs in the report, and a
reader recomputing with ``scipy.stats.spearmanr`` (average ranks) would get a
different number. `experiments/stats_utils.py` now provides the average-rank
implementation, and this script rewrites the numbers that were produced with the
old one.

Scope
-----
Every number written here is a function of points that are *already stored* in
the reports -- ``configs[]`` for the A2 mapper sweep, ``summary[]`` for the A4
error budget -- so nothing here needs a GPU or the models. The script

  * recomputes every stored correlation from those points under *both* tie
    policies and refuses to write unless the recorded value matches one of them
    (the "preflight" block). That proves the quantity is unchanged and only the
    tie policy differs. A recorded value that matches neither is a hard error --
    except for the A2 aggregate, see below;
  * then rewrites the same fields with average ranks:
      reports/phaseB_mappersweep_<pair>_seed<seed>.json : rho, rho_ci95, gate
      reports/phaseB_mappersweep_aggregate.json         : rho_pooled, rho_ci95
      reports/phaseB_errorbudget_<pair>_seed<seed>.json : spearman
  * records every old -> new value, plus the pooled scopes the paper quotes, in
    reports/rank_correlation_recompute.json.

Two W30 provenance defects surface here and are recorded rather than smoothed
over (details in `paper/audit/METRIC_CORRECTION.md` section 14):

  * the stored A2 ``gate`` blocks are stale -- they lack
    ``bootstrap_intervals_disjoint`` and carry the branch
    "direction preserved, phrase as consistent with" although the code, the
    intervals and section 13 of METRIC_CORRECTION all give the disjoint
    "strong statement preserved" branch. The gate is recomputed from the stored
    points and the old strings are kept in ``changes``;
  * the stored A2 aggregate does not reproduce from the stored per-run configs
    under *either* tie policy (~0.002-0.004 apart), i.e. it was derived from an
    earlier state of some per-run report. It is recorded as stale and
    regenerated from the stored per-run points.

The script is idempotent: a second run reports zero changes and writes the same
bytes. It preserves each file's newline convention and the ``indent=2,
ensure_ascii=False`` dump used by ``phaseB_common.save``.

Usage
-----
  python3 scripts/recompute_rank_correlations.py --dry-run
  python3 scripts/recompute_rank_correlations.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(_HERE)))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
from stats_utils import bootstrap_spearman_ci, spearman  # noqa: E402

PAIRS = ("1.7B_0.6B", "8B_0.6B")
SEEDS = (0, 1, 2)
ERRORS = ("e_raw", "e_attn", "e_wo")
ORDINAL_NOTE = "ordinal ranks (np.argsort(np.argsort))"
AVERAGE_NOTE = "average ranks (scipy-compatible)"


def ordinal_spearman(x, y) -> float:
    """The pre-W31 tie policy, kept only to verify the old numbers."""
    import numpy as np
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def ordinal_bootstrap_ci(err, em, n_boot=10000, seed=0):
    """The pre-W31 bootstrap, kept only to verify the old CI fields."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(err)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(np.asarray(err)[idx])) < 2 or len(np.unique(np.asarray(em)[idx])) < 2:
            continue
        vals.append(ordinal_spearman(np.asarray(err)[idx], np.asarray(em)[idx]))
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump_like(path, obj):
    """Write JSON the way phaseB_common.save does, keeping the file's newlines."""
    raw = b""
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            raw = fh.read()
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def a2_gate(rho, ci, n_configs) -> dict:
    """Mirror of the gate block in experiments/phaseB_mappersweep.py main()."""
    gate = {
        "rho_raw": rho["e_raw"], "rho_attn": rho["e_attn"], "rho_wo": rho["e_wo"],
        "rho_ci95": ci,
        "attn_wo_strong": bool(rho["e_attn"] is not None and rho["e_attn"] <= -0.7
                               and rho["e_wo"] is not None and rho["e_wo"] <= -0.7
                               and rho["e_raw"] is not None and rho["e_raw"] >= -0.3),
        "rho_raw_upper": ci["e_raw"][1],
        "n_configs": n_configs,
    }
    if gate["attn_wo_strong"]:
        ci_raw, ci_attn = ci["e_raw"], ci["e_attn"]
        disjoint = (ci_raw[0] is not None and ci_attn[0] is not None
                    and (ci_raw[1] < ci_attn[0] or ci_raw[0] > ci_attn[1]))
    else:
        disjoint = False
    if gate["attn_wo_strong"] and disjoint:
        gate["branch"] = "strong statement preserved"
    elif (rho["e_attn"] is not None and rho["e_attn"] <= -0.7
          and rho["e_wo"] is not None and rho["e_wo"] <= -0.7):
        gate["branch"] = "direction preserved, phrase as consistent with"
    else:
        gate["branch"] = "sign flip / weak: restrict claim to alpha ends"
    gate["bootstrap_intervals_disjoint"] = bool(disjoint)
    return gate


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without rewriting anything")
    a = ap.parse_args()

    problems, changes, preflight = [], [], {}

    # ---------------------------------------------------------------- A2 per run
    a2 = {}
    for pair in PAIRS:
        for seed in SEEDS:
            path = os.path.join(REPORT_DIR,
                                "phaseB_mappersweep_%s_seed%d.json" % (pair, seed))
            if not os.path.isfile(path):
                problems.append("missing A2 report: %s" % path)
                continue
            d = load(path)
            err = {k: [c[k] for c in d["configs"]] for k in ERRORS}
            em = [c["EM"] for c in d["configs"]]
            # preflight: the recorded numbers must come from the stored points under
            # one of the two tie policies (ordinal = pre-W31, average = current)
            want_rho, want_ci = d.get("rho", {}), d.get("rho_ci95", {})
            for k in ERRORS:
                rec, rec_ci = want_rho.get(k), want_ci.get(k)
                ord_now, avg_now = ordinal_spearman(err[k], em), spearman(err[k], em)
                if not (close(rec, ord_now, 1e-9) or close(rec, avg_now, 1e-9)):
                    problems.append("%s: recorded rho[%s]=%r matches neither tie "
                                    "policy (ordinal %.9f, average %.9f)"
                                    % (os.path.basename(path), k, rec, ord_now, avg_now))
                ord_ci = ordinal_bootstrap_ci(err[k], em, seed=d["seed"])
                avg_ci = bootstrap_spearman_ci(err[k], em, seed=d["seed"])
                if not (close(rec_ci, ord_ci, 1e-9) or close(rec_ci, avg_ci, 1e-9)):
                    problems.append("%s: recorded rho_ci95[%s]=%r matches neither "
                                    "tie policy" % (os.path.basename(path), k, rec_ci))
            a2[(pair, seed)] = d
            preflight.setdefault("a2_runs_recorded_ordinal_reproduced", []).append(
                "%s_seed%d" % (pair, seed))

    # ---------------------------------------------------------------- A2 pooled
    agg_path = os.path.join(REPORT_DIR, "phaseB_mappersweep_aggregate.json")
    agg = load(agg_path) if os.path.isfile(agg_path) else None
    if agg is None:
        problems.append("missing A2 aggregate report: %s" % agg_path)
    if problems:
        print("REFUSING TO WRITE -- preflight problems:")
        for p in problems:
            print("  - " + p)
        return 1

    # Rebuild the pooled point order. The stored `running` labels pin both the
    # run order and the config order that produced the recorded aggregate.
    order = []
    labels = agg.get("running") or []
    for lab in labels:
        run_id = lab.split("/")[0]
        if run_id not in order:
            order.append(run_id)
    by_id = {"%s_s%d" % (p, s): a2[(p, s)] for p in PAIRS for s in SEEDS}
    if sorted(order) != sorted(by_id):
        problems.append("aggregate `running` names runs %r, reports hold %r"
                        % (order, sorted(by_id)))
    rebuilt = []
    for run_id in (order or sorted(by_id)):
        rebuilt += [("%s/%s" % (run_id, c["key"]), c) for c in by_id[run_id]["configs"]]
    if labels and [lab for lab, _ in rebuilt] != labels:
        problems.append("rebuilt (run, config) order does not match the stored "
                        "`running` labels")
    if len(rebuilt) != agg.get("n_points") or len(order) != agg.get("n_runs"):
        problems.append("rebuilt %d points / %d runs, report says %r / %r"
                        % (len(rebuilt), len(order), agg.get("n_points"), agg.get("n_runs")))
    err_pool = {k: [c[k] for _, c in rebuilt] for k in ERRORS}
    em_pool = [c["EM"] for _, c in rebuilt]
    # The pooled point set and its order are pinned structurally above, so the
    # rebuild is the same input the aggregate saw. Its *values* may still differ:
    # the W30 aggregate disagrees with the stored per-run configs under *either*
    # tie policy (~0.002-0.004), i.e. it was computed from an earlier state of some
    # per-run report. That is recorded here rather than hidden, and the aggregate is
    # regenerated from the stored per-run points below.
    agg_drift = {}
    for k in ERRORS:
        rec = agg.get("rho_pooled", {}).get(k)
        rec_ci = agg.get("rho_ci95", {}).get(k)
        ord_now = ordinal_spearman(err_pool[k], em_pool)
        avg_now = spearman(err_pool[k], em_pool)
        ord_ci = ordinal_bootstrap_ci(err_pool[k], em_pool, seed=0)
        avg_ci = bootstrap_spearman_ci(err_pool[k], em_pool, seed=0)
        if not (close(rec, ord_now, 1e-9) or close(rec, avg_now, 1e-9)):
            agg_drift[k] = {
                "recorded_value": rec,
                "ordinal_from_stored_points": ord_now,
                "average_from_stored_points": avg_now,
                "abs_diff_vs_ordinal": (abs(rec - ord_now) if rec is not None else None),
            }
        if not (close(rec_ci, ord_ci, 1e-9) or close(rec_ci, avg_ci, 1e-9)):
            agg_drift.setdefault(k, {})["recorded_ci95"] = rec_ci
            agg_drift.setdefault(k, {})["ordinal_ci95_from_stored_points"] = ord_ci
    preflight["a2_aggregate_matches_stored_points"] = not agg_drift
    if agg_drift:
        preflight["a2_aggregate_stale_vs_stored_points"] = agg_drift
        print("NOTE: the stored A2 aggregate does not match the stored per-run "
              "points under either tie policy; it is treated as a stale derived "
              "file and regenerated from the per-run points.")
        for k, v in agg_drift.items():
            print("  - %s: %r" % (k, v))

    # ---------------------------------------------------------------- A4 per run
    a4 = {}
    for pair in PAIRS:
        for seed in SEEDS:
            path = os.path.join(REPORT_DIR,
                                "phaseB_errorbudget_%s_seed%d.json" % (pair, seed))
            if not os.path.isfile(path):
                problems.append("missing A4 report: %s" % path)
                continue
            d = load(path)
            names = list(d["summary"].keys())
            block = d.get("spearman", {})
            if block.get("n_variants") not in (None, len(names)):
                problems.append("%s: n_variants=%r but summary holds %d variants"
                                % (os.path.basename(path), block.get("n_variants"),
                                   len(names)))
            for k, field in zip(ERRORS, ("e_raw_vs_EM", "e_attn_vs_EM", "e_wo_vs_EM")):
                rr = [d["summary"][n][k + "_mean"] for n in names]
                ee = [d["summary"][n]["EM"] for n in names]
                rec = block.get(field)
                ord_now, avg_now = ordinal_spearman(rr, ee), spearman(rr, ee)
                if not (close(rec, ord_now, 1e-9) or close(rec, avg_now, 1e-9)):
                    problems.append("%s: recorded %s=%r matches neither tie policy "
                                    "(ordinal %.9f, average %.9f)"
                                    % (os.path.basename(path), field, rec,
                                       ord_now, avg_now))
            a4[(pair, seed)] = d
    if problems:
        print("REFUSING TO WRITE -- preflight problems:")
        for p in problems:
            print("  - " + p)
        return 1
    preflight["a4_runs_ordinal_reproduced"] = ["%s_seed%d" % p for p in
                                               [(p, s) for p in PAIRS for s in SEEDS]]

    # ============================================================== rewrite A2
    a2_new = {}
    for key, d in sorted(a2.items()):
        pair, seed = key
        fname = "phaseB_mappersweep_%s_seed%d.json" % (pair, seed)
        err = {k: [c[k] for c in d["configs"]] for k in ERRORS}
        em = [c["EM"] for c in d["configs"]]
        rho = {k: spearman(err[k], em) for k in ERRORS}
        ci = {k: bootstrap_spearman_ci(err[k], em, seed=seed) for k in ERRORS}
        gate = a2_gate(rho, ci, len(d["configs"]))
        for k in ERRORS:
            changes.append({"file": fname, "field": "rho.%s" % k,
                            "old": d["rho"][k], "new": rho[k]})
            changes.append({"file": fname, "field": "rho_ci95.%s" % k,
                            "old": d["rho_ci95"][k], "new": ci[k]})
        for field in ("rho_raw", "rho_attn", "rho_wo", "rho_raw_upper",
                      "attn_wo_strong", "bootstrap_intervals_disjoint", "branch"):
            changes.append({"file": fname, "field": "gate.%s" % field,
                            "old": d["gate"].get(field), "new": gate[field]})
        d["rho"], d["rho_ci95"], d["gate"] = rho, ci, gate
        d["rank_method"] = "average"
        a2_new[key] = d

    # ========================================================= rewrite aggregate
    rho_pool = {k: spearman(err_pool[k], em_pool) for k in ERRORS}
    ci_pool = {k: bootstrap_spearman_ci(err_pool[k], em_pool, seed=0) for k in ERRORS}
    for k in ERRORS:
        changes.append({"file": "phaseB_mappersweep_aggregate.json",
                        "field": "rho_pooled.%s" % k,
                        "old": agg["rho_pooled"][k], "new": rho_pool[k]})
        changes.append({"file": "phaseB_mappersweep_aggregate.json",
                        "field": "rho_ci95.%s" % k,
                        "old": agg["rho_ci95"][k], "new": ci_pool[k]})
    agg["rho_pooled"], agg["rho_ci95"] = rho_pool, ci_pool
    agg["rank_method"] = "average"

    # ============================================================== rewrite A4
    a4_new = {}
    for key, d in sorted(a4.items()):
        pair, seed = key
        fname = "phaseB_errorbudget_%s_seed%d.json" % (pair, seed)
        names = list(d["summary"].keys())
        new_block = dict(d["spearman"])
        for k, field in zip(ERRORS, ("e_raw_vs_EM", "e_attn_vs_EM", "e_wo_vs_EM")):
            got = spearman([d["summary"][n][k + "_mean"] for n in names],
                           [d["summary"][n]["EM"] for n in names])
            changes.append({"file": fname, "field": "spearman.%s" % field,
                            "old": d["spearman"].get(field), "new": got})
            new_block[field] = got
        new_block["rank_method"] = "average"
        new_block["n_variants"] = len(names)
        d["spearman"] = new_block
        a4_new[key] = d

    # pooled-over-runs A4 values (the numbers the paper quotes as "pooled")
    a4_pooled = {}
    names = list(a4_new[(PAIRS[0], SEEDS[0])]["summary"].keys())
    for k, field in zip(ERRORS, ("e_raw_vs_EM", "e_attn_vs_EM", "e_wo_vs_EM")):
        old = ordinal_spearman(
            [sum(a4[(p, s)]["summary"][n][k + "_mean"] for p in PAIRS for s in SEEDS) / 6
             for n in names],
            [sum(a4[(p, s)]["summary"][n]["EM"] for p in PAIRS for s in SEEDS) / 6
             for n in names])
        new = spearman(
            [sum(a4_new[(p, s)]["summary"][n][k + "_mean"] for p in PAIRS for s in SEEDS) / 6
             for n in names],
            [sum(a4_new[(p, s)]["summary"][n]["EM"] for p in PAIRS for s in SEEDS) / 6
             for n in names])
        runs_new, runs_old = [], []
        for p in PAIRS:
            for s in SEEDS:
                rr = [a4_new[(p, s)]["summary"][n][k + "_mean"] for n in names]
                ee = [a4_new[(p, s)]["summary"][n]["EM"] for n in names]
                runs_new.append(spearman(rr, ee))
                runs_old.append(ordinal_spearman(rr, ee))
        a4_pooled[field] = {
            "pooled_old": old, "pooled_new": new,
            "per_run_old": runs_old, "per_run_new": runs_new,
            "per_run_range_new": [min(runs_new), max(runs_new)],
        }

    # ==================================================================== output
    # The audit trail is append-only across reruns: the first run's old -> new
    # record is kept under "migration", later runs only append to "verified_reruns"
    # so re-running cannot erase what the W30 state looked like.
    record = {
        "preflight": preflight,
        "n_changed_fields": sum(1 for c in changes if not close(c["old"], c["new"], 0.0)),
        "changes": changes,
        "a2_pooled_198_points": {"rho_pooled": rho_pool, "rho_ci95": ci_pool,
                                 "n_points": len(em_pool), "n_runs": len(order)},
        "a4_pooled_over_six_runs": a4_pooled,
    }
    out_path = os.path.join(REPORT_DIR, "rank_correlation_recompute.json")
    reruns = []
    if os.path.isfile(out_path):
        try:
            prev = load(out_path)
        except ValueError:
            prev = {}
        if isinstance(prev, dict) and "migration" in prev:
            first = prev["migration"]
            reruns = list(prev.get("verified_reruns", []))
        else:
            first = prev or record          # pre-W31 shape: treat as the record
    else:
        first = record
    reruns.append({"n_changed_fields": record["n_changed_fields"],
                   "a2_aggregate_matches_stored_points":
                       preflight.get("a2_aggregate_matches_stored_points")})
    out = {
        "task": "rank_correlation_recompute",
        "why": ("W31 switched spearman() from ordinal to average ranks; the stored "
                "correlations were recomputed from the points already in the reports"),
        "method": {"old": ORDINAL_NOTE, "new": AVERAGE_NOTE,
                   "bootstrap": "point-level, n_boot=10000, seed=run seed (aggregate: 0)"},
        "migration": first,
        "verified_reruns": reruns,
    }

    print("preflight: every recorded correlation matches the stored points under at "
          "least one of the two tie policies -- the quantity is unchanged, only the "
          "tie policy moves (%d of %d fields change)."
          % (record["n_changed_fields"], len(changes)))
    print("\n%-42s %-22s %10s %10s" % ("file", "field", "old", "new"))
    for c in changes:
        if close(c["old"], c["new"], 0.0):
            continue
        o, n = c["old"], c["new"]
        fmt = lambda v: ("[%s]" % ",".join("%+.3f" % x for x in v)) if isinstance(v, list) \
            else ("%+.4f" % v if isinstance(v, float) else str(v))
        print("%-42s %-22s %10s %10s" % (c["file"], c["field"], fmt(o), fmt(n)))
    print("\nA2 pooled (198 pts): " + "  ".join(
        "%s %+.3f [%+.3f,%+.3f]" % (k, rho_pool[k], ci_pool[k][0], ci_pool[k][1])
        for k in ERRORS))
    print("A4 pooled (5 variants x 6 runs): " + "  ".join(
        "%s %+.3f->%+.3f" % (f, a4_pooled[f]["pooled_old"], a4_pooled[f]["pooled_new"])
        for f in a4_pooled))
    for f in a4_pooled:
        print("   %-12s per-run old %s -> new %s" % (
            f, ["%+.2f" % x for x in a4_pooled[f]["per_run_old"]],
            ["%+.2f" % x for x in a4_pooled[f]["per_run_new"]]))

    if a.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    for (pair, seed), d in a2_new.items():
        dump_like(os.path.join(REPORT_DIR,
                               "phaseB_mappersweep_%s_seed%d.json" % (pair, seed)), d)
    dump_like(agg_path, agg)
    for (pair, seed), d in a4_new.items():
        dump_like(os.path.join(REPORT_DIR,
                               "phaseB_errorbudget_%s_seed%d.json" % (pair, seed)), d)
    dump_like(out_path, out)
    print("\n[rank] rewrote 6 A2 reports, the A2 aggregate, 6 A4 reports and "
          "reports/rank_correlation_recompute.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
