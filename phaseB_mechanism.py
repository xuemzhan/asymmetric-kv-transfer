"""B3: Mechanism probe --- routing mismatch vs weight-bound consumption.

Reviewer §5: standard attention is A = softmax(Q K^T), O = A V W_O. V-only can
fail because (a) mapped teacher V is mismatched with student routing A_S, or
(b) mapped teacher V is mismatched with the student's downstream consumption.

This script:
  1. Computes the student's attention maps under Self (A_S, student K) and
     K-only (A'_T, mapped teacher K) and compares them (routing diagnostic).
  2. Fits an "attention-output-aware" V mapper: instead of matching V in
     V-space, it matches the attention output A_S V (i.e., weighted ridge
     regression in the directions the student actually reads). If this mapper
     recovers V-only performance, V failure is downstream-weight-bound; if it
     does not, the failure is in routing/content itself.

Usage:
  python3 phaseB_mechanism.py --pair 8B_0.6B --seed 0 --n-diagnostic 24
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
    load_pair,
    map_teacher,
    save,
    score_arm,
    stack_kv,
    summarize_rows,
)
from phase0_g0 import build_cache


def get_attn_map(model, tok, cache, query: str, n_doc: int):
    """Return list over layers of (H, S_q, n_doc) attention probabilities."""
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model(q_ids, past_key_values=cache, use_cache=True, output_attentions=True)
    attn = out.attentions
    maps = [a[0].float().cpu().numpy()[:, :, :n_doc] for a in attn]
    return maps


def compare_maps(a_s, a_t):
    """Per-layer routing divergence between two (H, S_q, n_doc) attention maps."""
    # mean over heads, mean over query positions -> doc-token distribution
    p = a_s.mean(axis=(0, 1))
    q = a_t.mean(axis=(0, 1))
    eps = 1e-12
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    tv = 0.5 * np.abs(p - q).sum()
    cos = float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q) + eps))
    # top-1 attention agreement across query positions and heads
    top_s = a_s.argmax(axis=-1)
    top_t = a_t.argmax(axis=-1)
    agree = float((top_s == top_t).mean())
    # entropy of mean doc distribution
    ent = lambda x: float(-(x * np.log(x + eps)).sum())
    return {"total_variation": float(tv), "cosine": cos,
            "top1_agreement": agree, "entropy_self": ent(p), "entropy_konly": ent(q)}


def fit_output_aware_mapper(calib_t, calib_s, attn_by_sample, layer_map,
                            lam=1e-3, target_s=None):
    """Weighted ridge regression matching attention output A @ V.

    For each (student layer, head): X = A_S @ V_T_head, Y = A_S @ V_S_head on
    calibration tokens; solve for W mapping X -> Y, then apply as V_T @ W.
    Returns dict W[(s,h)] (D,D) and bias[(s,h)].

    `target_s` overrides the target student KV (shuffled-target control); rows
    are truncated to the common document length.
    """
    W, b = {}, {}
    L_s = calib_s[0].v.shape[0]
    H = calib_s[0].v.shape[2]
    tgt = target_s if target_s is not None else calib_s
    for s in range(L_s):
        src_layer = layer_map[s][0]
        Xs = {h: [] for h in range(H)}
        Ys = {h: [] for h in range(H)}
        for i in range(len(calib_t)):
            A = attn_by_sample[i][s].mean(axis=0)   # (S_q, n_doc)
            Vt = calib_t[i].v[src_layer]            # (n_doc_t, H, D)
            Vs = tgt[i].v[s]                        # (n_doc_s, H, D)
            m = min(A.shape[1], Vt.shape[0], Vs.shape[0])
            A, Vt, Vs = A[:, :m], Vt[:m], Vs[:m]
            for h in range(H):
                Xs[h].append(A @ Vt[:, h, :])
                Ys[h].append(A @ Vs[:, h, :])
        for h in range(H):
            X = np.concatenate(Xs[h], axis=0).astype(np.float64)
            Y = np.concatenate(Ys[h], axis=0).astype(np.float64)
            Xm, Ym = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
            Xc, Yc = X - Xm, Y - Ym
            W[(s, h)] = np.linalg.solve(
                Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
            b[(s, h)] = (Ym - Xm @ W[(s, h)]).ravel()
    return W, b


def apply_output_aware(kv_t, layer_map, W, b):
    L_s = len(layer_map)
    L_t, S, H, D = kv_t.k.shape
    v = np.zeros((L_s, S, H, D), dtype=np.float64)
    for s in range(L_s):
        src = kv_t.v[layer_map[s][0]].astype(np.float64)
        for h in range(H):
            v[s, :, h, :] = src[:, h, :] @ W[(s, h)] + b[(s, h)]
    return v.astype(np.float32)


def run(pair: str, seed: int, n_calib: int, n_eval: int, n_diagnostic: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_data(seed, "train")[:n_calib]
    test = load_data(seed, "test")[:n_eval]
    print("[B3] capture KV (teacher then student) ...", flush=True)
    (calib_t, eval_t, calib_s, eval_s, student, tok_s,
     t_layers, s_layers) = capture_pair_kv(pair, train, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    del ct, cs
    torch.cuda.empty_cache()

    mapped_t = [map_teacher(mk, mv, eval_t[i], lmap) for i in range(len(test))]

    # ---- routing diagnostic on a subset -----------------------------------
    nd = min(n_diagnostic, len(test))
    print(f"[B3] attention diagnostic on {nd} samples ...", flush=True)
    routing = []
    for i in range(nd):
        s = test[i]
        n_doc = eval_s[i].k.shape[1]
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        a_s = get_attn_map(student, tok_s, build_cache(KV(k=eval_s[i].k, v=eval_s[i].v)), q, n_doc)
        a_t = get_attn_map(student, tok_s, build_cache(KV(k=mapped_t[i].k, v=eval_s[i].v)), q, n_doc)
        per_layer = [compare_maps(a_s[l], a_t[l]) for l in range(len(a_s))]
        routing.append({"id": s["id"], "per_layer": per_layer})
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{nd}]", flush=True)
    routing_agg = {
        key: [float(np.mean([r["per_layer"][l][key] for r in routing]))
              for l in range(s_layers)]
        for key in ("total_variation", "cosine", "top1_agreement")
    }

    # ---- attention-output-aware V mapper ----------------------------------
    print("[B3] fitting attention-output-aware V mapper ...", flush=True)
    attn_calib = []
    for i, s in enumerate(train):
        n_doc = calib_s[i].k.shape[1]
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        attn_calib.append(get_attn_map(
            student, tok_s, build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)), q, n_doc))
    W, b = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap,
                                   lam=1e-3)
    out_aware = [KV(k=eval_s[i].k,
                    v=apply_output_aware(eval_t[i], lmap, W, b))
                 for i in range(len(test))]

    # ---- compare V-only performance: Affine vs output-aware ----------------
    rows = []
    for i, s in enumerate(test):
        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"]}
        ll, em = score_arm(student, tok_s, KV(k=eval_s[i].k, v=eval_s[i].v), s)
        row["Self"], row["Self_em"] = ll, float(em)
        ll, em = score_arm(student, tok_s, KV(k=eval_s[i].k, v=mapped_t[i].v), s)
        row["V-only_Affine"], row["V-only_Affine_em"] = ll, float(em)
        ll, em = score_arm(student, tok_s, out_aware[i], s)
        row["V-only_OutAware"], row["V-only_OutAware_em"] = ll, float(em)
        rows.append(row)

    groups = doc_groups(test)
    summary = summarize_rows(rows, ["V-only_Affine", "V-only_OutAware"],
                             base="Self", seed=seed, groups=groups)
    report = {
        "task": "phaseB_mechanism",
        "pair": pair, "seed": seed,
        "n_calib": n_calib, "n_eval": n_eval, "n_diagnostic": nd,
        "routing_agg_by_layer": routing_agg,
        "routing_per_sample": routing,
        "summary": summary,
        "rows": rows,
    }
    save(report, output)
    print("\n=== B3 mechanism (pair=%s seed=%d) ===" % (pair, seed))
    print("routing (mean over layers):",
          {k: round(float(np.mean(v)), 4) for k, v in routing_agg.items()})
    for k, r in summary.items():
        print(f"  {k:18s} dLL={r['delta_vs_self']:+.3f} p={r['wilcoxon_p']:.2e} EM={r['EM']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="8B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--n-diagnostic", type=int, default=24)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    out = a.output or f"/workspace/v3/reports/phaseB_mechanism_{a.pair}_seed{a.seed}.json"
    run(a.pair, a.seed, a.n_calib, a.n_eval, a.n_diagnostic, out)


if __name__ == "__main__":
    main()
