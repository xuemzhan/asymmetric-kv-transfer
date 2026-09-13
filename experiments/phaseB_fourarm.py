"""Corrected four-arm scan for all pairs (fixes the legacy greedy-EM artifact).

Runs Self / K-only / V-only / Joint with phaseB_common.score_arm (fresh cache
for generation + no double-feed) and saves per-sample rows, per-sample and
document-clustered CIs.

Usage:
  python3 experiments/phaseB_fourarm.py --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    capture_all,
    doc_groups,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    map_teacher,
    save,
    score_arm,
    stack_kv,
    summarize_rows,
)


def run_pair(pair: str, seed: int, n_calib: int, n_eval: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = load_data(seed, "train")[:n_calib]
    test = load_data(seed, "test")[:n_eval]

    # Teacher first, capture, then free before loading the student (8B+4B OOM fix).
    teacher, tok_t, t_layers = load_teacher(pair)
    calib_t = capture_all(teacher, tok_t, train)
    eval_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()

    student, tok_s, s_layers = load_student(pair)
    calib_s = capture_all(student, tok_s, train)
    eval_s = capture_all(student, tok_s, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    del ct, cs, calib_t, calib_s
    torch.cuda.empty_cache()
    mapped = [map_teacher(mk, mv, eval_t[i], lmap) for i in range(len(test))]

    rows = []
    for i, s in enumerate(test):
        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"],
               "doc_id": s.get("project", s["doc"][:40])}
        arms = {
            "Self": KV(k=eval_s[i].k, v=eval_s[i].v),
            "K-only": KV(k=mapped[i].k, v=eval_s[i].v),
            "V-only": KV(k=eval_s[i].k, v=mapped[i].v),
            "Joint": KV(k=mapped[i].k, v=mapped[i].v),
        }
        for name, kv in arms.items():
            ll, em = score_arm(student, tok_s, kv, s)
            row[name] = ll
            row[name + "_em"] = float(em)
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(test)}]", flush=True)

    summary = summarize_rows(rows, ["K-only", "V-only", "Joint"], base="Self",
                             seed=seed, groups=doc_groups(test))
    return {
        "pair": pair, "seed": seed, "n_calib": n_calib, "n_eval": n_eval,
        "self_EM": float(np.mean([r["Self_em"] for r in rows])),
        "summary": summary, "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    out = a.output or f"/workspace/v3/reports/phaseB_fourarm_seed{a.seed}.json"
    results = []
    for pair in a.pairs:
        print(f"=== {pair} ===", flush=True)
        results.append(run_pair(pair, a.seed, a.n_calib, a.n_eval))
        # incremental save so an OOM on a later pair does not lose earlier work
        save({"task": "phaseB_fourarm_corrected", "seed": a.seed,
              "n_calib": a.n_calib, "n_eval": a.n_eval,
              "pair_results": results}, out)
    report = {"task": "phaseB_fourarm_corrected", "seed": a.seed,
              "n_calib": a.n_calib, "n_eval": a.n_eval,
              "pair_results": results}
    save(report, out)
    print("\n=== corrected four-arm ===")
    print(f"{'pair':10s} {'SelfEM':>7s} {'K_EM':>6s} {'V_EM':>6s} {'J_EM':>6s} "
          f"{'K_dLL':>7s} {'V_dLL':>7s} {'J_dLL':>7s}")
    for r in results:
        s = r["summary"]
        print(f"{r['pair']:10s} {r['self_EM']:7.3f} {s['K-only']['EM']:6.3f} "
              f"{s['V-only']['EM']:6.3f} {s['Joint']['EM']:6.3f} "
              f"{s['K-only']['delta_vs_self']:7.2f} "
              f"{s['V-only']['delta_vs_self']:7.2f} "
              f"{s['Joint']['delta_vs_self']:7.2f}")


if __name__ == "__main__":
    main()
