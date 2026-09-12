"""B1: Decouple capability gap from layer alignment / mapping quality.

Reviewer §2: the six-pair scan confounds parameter gap, layer-count mismatch,
and proportional-mapping error --- V only "succeeds" on the two equal-layer
pairs, so capability-gated V transfer is not identified.

This script holds the pair fixed and varies ONLY the layer map (alignment),
then repeats across pairs with different capability gaps. It yields the
x=capability-gap / y=alignment-quality phase diagram requested by the reviewer.

Maps evaluated per pair:
  - proportional                (published baseline)
  - offset +/- k                (misaligned but same layer count)
  - cyclic shift                (equal-layer permutation)
  - random permutation          (destroys alignment)
  - learned top-1 (K-based)     (unequal-layer pairs only)
  - learned top-1 (V-based)     (unequal-layer pairs only)

Usage:
  python3 phaseB_alignment.py --pair 1.7B_0.6B --seed 0
  python3 phaseB_alignment.py --pair 8B_0.6B --seed 0
  python3 phaseB_alignment.py --pair 8B_4B --seed 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    capture_pair_kv,
    capture_all,
    fit_mapper,
    layer_map_identity,
    layer_map_learned_topk,
    layer_map_offset,
    layer_map_permuted,
    layer_map_proportional,
    layer_map_shift,
    doc_groups,
    load_data,
    load_pair,
    map_teacher,
    save,
    score_arm,
    stack_kv,
    summarize_rows,
)


def build_maps(pair: str, t_layers: int, s_layers: int, ct, cs, seed: int):
    maps = {"proportional": layer_map_proportional(t_layers, s_layers)}
    if t_layers == s_layers:
        maps["identity"] = layer_map_identity(t_layers, s_layers)
    for d in (3, -3):
        maps[f"offset{d:+d}"] = layer_map_offset(t_layers, s_layers, d)
    maps["permuted"] = layer_map_permuted(t_layers, s_layers, seed=seed + 7)

    learned = {}
    if t_layers != s_layers:
        for kind in ("K", "V"):
            t0 = time.time()
            lm, info = layer_map_learned_topk(t_layers, s_layers, ct, cs, k=1, kind=kind)
            learned[f"learned_top1_{kind}"] = lm
            print(f"  [learned {kind}] computed in {time.time()-t0:.1f}s", flush=True)
    maps.update(learned)
    return maps


def alignment_score(lmap, t_layers: int, s_layers: int) -> float:
    """Mean absolute deviation of the map from the proportional map (lower=better aligned)."""
    prop = [min(t_layers - 1, round(s * t_layers / s_layers)) for s in range(s_layers)]
    v = [np.mean([abs(float(t) - prop[s]) for t in lmap[s]]) for s in range(s_layers)]
    return float(np.mean(v))


def run(pair: str, seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_data(seed, "train")[:n_calib]
    test = load_data(seed, "test")[:n_eval]
    print("[B1] capture KV (teacher then student) ...", flush=True)
    (calib_t, eval_t, calib_s, eval_s, student, tok_s,
     t_layers, s_layers) = capture_pair_kv(pair, train, test)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)

    # Self baseline
    self_rows = []
    for i, s in enumerate(test):
        ll, em = score_arm(student, tok_s, eval_s[i], s, want_em=True)
        self_rows.append((ll, em))
    self_ll = float(np.mean([r[0] for r in self_rows]))
    self_em = float(np.mean([r[1] for r in self_rows]))
    print(f"[B1] Self LL={self_ll:.3f} EM={self_em:.3f}", flush=True)

    maps = build_maps(pair, t_layers, s_layers, ct, cs, seed)

    results = {}
    for name, lmap in maps.items():
        mk = fit_mapper("K", ct, cs, lmap)
        mv = fit_mapper("V", ct, cs, lmap)
        rows = []
        for i, s in enumerate(test):
            row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"]}
            row["Self"] = self_rows[i][0]
            row["Self_em"] = float(self_rows[i][1])
            m = map_teacher(mk, mv, eval_t[i], lmap)
            ll_k, em_k = score_arm(student, tok_s, KV(k=m.k, v=eval_s[i].v), s)
            ll_v, em_v = score_arm(student, tok_s, KV(k=eval_s[i].k, v=m.v), s)
            ll_j, em_j = score_arm(student, tok_s, KV(k=m.k, v=m.v), s)
            row.update({
                "K-only": ll_k, "K-only_em": float(em_k),
                "V-only": ll_v, "V-only_em": float(em_v),
                "Joint": ll_j, "Joint_em": float(em_j),
            })
            rows.append(row)
        summary = summarize_rows(rows, ["K-only", "V-only", "Joint"], base="Self",
                                 seed=seed, groups=doc_groups(test))
        results[name] = {
            "layer_map": lmap,
            "alignment_deviation": alignment_score(lmap, t_layers, s_layers),
            "summary": summary,
        }
        print(f"  {name:16s} align={results[name]['alignment_deviation']:.2f} "
              f"K_d={summary['K-only']['delta_vs_self']:+.2f} "
              f"V_d={summary['V-only']['delta_vs_self']:+.2f} "
              f"J_d={summary['Joint']['delta_vs_self']:+.2f} "
              f"V_EM={summary['V-only']['EM']:.2f}", flush=True)
        del rows

    param_ratio = {"8B_0.6B": 13.3, "4B_1.7B": 2.35, "4B_0.6B": 6.7,
                   "8B_1.7B": 4.7, "1.7B_0.6B": 2.8, "8B_4B": 2.0}[pair]
    report = {
        "task": "phaseB_alignment",
        "pair": pair, "seed": seed,
        "param_ratio": param_ratio,
        "t_layers": t_layers, "s_layers": s_layers,
        "self_LL": self_ll, "self_EM": self_em,
        "n_calib": n_calib, "n_eval": n_eval,
        "results": results,
    }
    save(report, output)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="1.7B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    out = a.output or f"/workspace/v3/reports/phaseB_alignment_{a.pair}_seed{a.seed}.json"
    run(a.pair, a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
