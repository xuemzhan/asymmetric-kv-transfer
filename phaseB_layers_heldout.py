"""B2: Held-out V-hotspot discovery.

Reviewer §8: L8/L12 were selected by scanning the test set, so the reported
+2.15 (L8+L12) may carry post-selection bias, and scanning 28 layers is 28
comparisons.

Protocol (honest):
  1. Fit the Affine mapper on the training split.
  2. Scan all student layers with single-layer V-only injection on the
     VALIDATION split (28 samples).
  3. Freeze the top-2 validation layers and evaluate them once on the TEST
     split (56 samples).
  4. Compare with the legacy L8/L12 selection and with the test-oracle top-2
     (to quantify the selection bias).

Usage:
  python3 phaseB_layers_heldout.py --pair 8B_0.6B --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    capture_all,
    capture_pair_kv,
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


def scan_layers(student, tok_s, eval_s, mapped, samples, layers, base_lls, seed: int):
    """Single-layer V-only delta LL (inject only mapped teacher V at one layer)."""
    out = []
    for sl in layers:
        lls = []
        for i, s in enumerate(samples):
            v = eval_s[i].v.copy()
            v[sl] = mapped[i].v[sl]
            ll, _ = score_arm(student, tok_s, KV(k=eval_s[i].k, v=v), s, want_em=False)
            lls.append(ll)
        lls = np.array(lls)
        out.append({
            "layer": int(sl),
            "V_only_delta_LL": float((lls - base_lls).mean()),
            "V_only_LL_mean": float(lls.mean()),
        })
    return out


def evaluate_layers(student, tok_s, eval_s, mapped, samples, base_rows, layer_lists, seed: int):
    results = {}
    for name, layers in layer_lists.items():
        rows = []
        for i, s in enumerate(samples):
            v = eval_s[i].v.copy()
            for sl in layers:
                v[sl] = mapped[i].v[sl]
            ll, em = score_arm(student, tok_s, KV(k=eval_s[i].k, v=v), s, want_em=True)
            rows.append({"Self": base_rows[i][0], "Self_em": float(base_rows[i][1]),
                         "V-only": ll, "V-only_em": float(em)})
        summary = summarize_rows(rows, ["V-only"], base="Self", seed=seed,
                                 groups=doc_groups(samples))
        results[name] = {"layers": [int(x) for x in layers], **summary["V-only"]}
    return results


def run(pair: str, seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_data(seed, "train")[:n_calib]
    val = load_data(seed, "val")
    test = load_data(seed, "test")[:n_eval]
    print("[B2] capture teacher KV ...", flush=True)
    teacher, tok_t, t_layers = load_teacher(pair)
    calib_t = capture_all(teacher, tok_t, train)
    val_t = capture_all(teacher, tok_t, val)
    test_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()
    print("[B2] capture student KV ...", flush=True)
    student, tok_s, s_layers = load_student(pair)
    calib_s = capture_all(student, tok_s, train)
    val_s = capture_all(student, tok_s, val)
    test_s = capture_all(student, tok_s, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    del ct, cs
    torch.cuda.empty_cache()

    val_mapped = [map_teacher(mk, mv, val_t[i], lmap) for i in range(len(val))]
    test_mapped = [map_teacher(mk, mv, test_t[i], lmap) for i in range(len(test))]

    # baselines
    val_base = [score_arm(student, tok_s, val_s[i], val[i], want_em=False)[0]
                for i in range(len(val))]
    val_base = np.array(val_base)
    test_base = [score_arm(student, tok_s, test_s[i], test[i], want_em=True)
                 for i in range(len(test))]

    print("[B2] scan all layers on validation ...", flush=True)
    val_scan = scan_layers(student, tok_s, val_s, val_mapped, val,
                           list(range(s_layers)), val_base, seed)
    val_ranked = sorted(val_scan, key=lambda r: -r["V_only_delta_LL"])
    top2_val = [r["layer"] for r in val_ranked[:2]]
    rank_l8 = next((j for j, r in enumerate(val_ranked) if r["layer"] == 8), None)
    rank_l12 = next((j for j, r in enumerate(val_ranked) if r["layer"] == 12), None)

    print("[B2] scan all layers on test (oracle reference) ...", flush=True)
    test_base_ll = np.array([r[0] for r in test_base])
    test_scan = scan_layers(student, tok_s, test_s, test_mapped, test,
                            list(range(s_layers)), test_base_ll, seed)
    test_ranked = sorted(test_scan, key=lambda r: -r["V_only_delta_LL"])
    top2_test = [r["layer"] for r in test_ranked[:2]]

    # evaluate selected sets on test
    layer_lists = {
        "legacy_L8L12": [8, 12],
        f"val_top1_L{top2_val[0]}": [top2_val[0]],
        f"val_top2_L{top2_val[0]}_L{top2_val[1]}": top2_val,
        f"testoracle_L{top2_test[0]}_L{top2_test[1]}": top2_test,
        "V_ALL": list(range(s_layers)),
    }
    print("[B2] evaluate selected sets on test ...", flush=True)
    test_eval = evaluate_layers(student, tok_s, test_s, test_mapped, test,
                                test_base, layer_lists, seed)

    report = {
        "task": "phaseB_layers_heldout",
        "pair": pair, "seed": seed,
        "t_layers": t_layers, "s_layers": s_layers,
        "n_calib": n_calib, "n_val": len(val), "n_eval": len(test),
        "layer_map": lmap,
        "val_scan": val_scan,
        "test_scan": test_scan,
        "val_top2": top2_val,
        "test_top2_oracle": top2_test,
        "L8_val_rank": rank_l8, "L12_val_rank": rank_l12,
        "test_eval": test_eval,
        "self_val_LL": float(val_base.mean()),
        "self_test_LL": float(test_base_ll.mean()),
        "self_test_EM": float(np.mean([r[1] for r in test_base])),
    }
    save(report, output)
    print("\n=== B2 held-out hotspot (pair=%s seed=%d) ===" % (pair, seed))
    print(f"val top2: {top2_val} (L8 rank={rank_l8}, L12 rank={rank_l12})")
    print(f"test oracle top2: {top2_test}")
    for name, r in test_eval.items():
        print(f"  {name:28s} dLL={r['delta_vs_self']:+.3f} "
              f"CI{np.round(r['delta_ci95'],2)} p={r['wilcoxon_p']:.2e} EM={r['EM']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="8B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    out = a.output or f"/workspace/v3/reports/phaseB_layers_heldout_{a.pair}_seed{a.seed}.json"
    run(a.pair, a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
