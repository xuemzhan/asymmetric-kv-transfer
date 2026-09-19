"""A1: routing-aware K mapper (audit4 section 4, REVISION_PLAN4 II.A1).

Motivation
----------
`phaseB_mechanism` diagnoses key-side failure through routing divergence but
never intervenes on it. This script makes the key side symmetric with the
value side: fit the key map where the student actually looks, then measure
whether closing the routing gap repairs the K arm.

Default mapper (closed form, same de-RoPE convention as `AffineMapper`)
----------------------------------------------------------------------
For each student layer l / KV head h, fit the weighted ridge regression

    min_Z  sum_j w_j || de_rope(K_T)[j] Z - K_S[j] ||^2 + lam ||Z||^2

where the token weight w_j is the total attention mass the student pays to
document position j under its own (Self) cache, summed over the query heads in
the KV head's group and over query positions. w_j == 1 recovers the published
affine K mapper exactly; the script asserts that numerically (self-consistency
gate) before any routing number is interpreted.

Arms
----
  K-affine        : published AffineMapper K baseline
  K-routing       : weighted (attention-mass) K ridge
  K-routing-shuf  : same fit with targets from another document (content control)
  K-raw           : no learned map, proportional layer selection (identity)

Pre-registered gate (REVISION_PLAN4 II.A1, three branches, no post-hoc pick):
  * routing TV drops >= 30% vs affine AND K-only EM clustered CI95 excludes 0
      -> intervention holds ("routing-space alignment partially repairs the key arm")
  * routing TV drops but K-only EM stays at the floor
      -> stronger: routing alignment alone is insufficient
  * routing TV does not drop
      -> mapper-family limitation (honest negative intervention)

Usage:
  python3 experiments/phaseB_routingkey.py --pair 1.7B_0.6B --seed 0 \
    --n-calib 70 --n-eval 56
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    build_cache,
    capture_all,
    de_rope_k,
    doc_groups,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    query_of,
    raw_map_teacher,
    save,
    score_arm,
    stack_kv,
    summarize_rows,
)
from phaseB_mechanism import compare_maps, get_attn_map

_ROOTS = (os.environ.get("V3_ROOT")
          or ("/workspace/v3"
              if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
              else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOTS, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOTS, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")


def group_heads(n_q: int, n_kv: int) -> dict:
    reps = n_q // n_kv
    return {h: [h * reps + g for g in range(reps)] for h in range(n_kv)}


def concat_attention_weights(attn_by_sample, lengths, group_of_head, L_s):
    """Per-layer concatenated weight vectors w (S_total, H_kv).

    `attn_by_sample[i]` is the per-layer list of (H_q, S_q, n_doc) maps returned
    by `get_attn_map`; `lengths[i]` is that sample's document length (the number
    of KV positions the attention map indexes).
    """
    H_kv = None
    blocks = None
    for i, attn in enumerate(attn_by_sample):
        Li = lengths[i]
        if blocks is None:
            H_kv = len(group_of_head)
            blocks = [np.zeros((sum(lengths), H_kv), dtype=np.float64)
                      for _ in range(L_s)]
        off = int(np.sum(lengths[:i]))
        for s in range(L_s):
            A = attn[s]                                  # (H_q, S_q, n_doc)
            m = min(Li, A.shape[-1])
            for h, qs in group_of_head.items():
                w = A[qs][:, :, :m].sum(axis=(0, 1))     # (m,)
                blocks[s][off:off + m, h] = w
    return blocks


def fit_weighted_k_mapper(calib_t, calib_s, weight_blocks, layer_map,
                          lam: float = 1e-3, target_s=None):
    """Weighted per-(layer, head) centred ridge in the AffineMapper convention.

    `weight_blocks[s]` is (S_total, H_kv). Passing all-ones reproduces
    `fit_mapper("K", ...)` (asserted by the caller). `target_s` overrides the
    student targets (used for the shuffled-document control).
    """
    ct = stack_kv(calib_t)
    cs = stack_kv(calib_s)
    L_s, S, H, D = cs.k.shape
    pos = np.arange(S, dtype=np.float64)
    tgt = target_s if target_s is not None else cs
    Z, b = {}, {}
    for s in range(L_s):
        src = layer_map[s][0]
        x_all = de_rope_k(ct.k[src], pos).astype(np.float64)
        y_all = tgt.k[s].astype(np.float64)
        for h in range(H):
            x, y = x_all[:, h, :], y_all[:, h, :]
            w = (weight_blocks[s][:, h] if weight_blocks is not None
                 else np.ones(S))
            sw = float(w.sum()) + 1e-12
            xm = (w[:, None] * x).sum(0) / sw
            ym = (w[:, None] * y).sum(0) / sw
            xc, yc = x - xm, y - ym
            A_ = xc.T @ (w[:, None] * xc) + lam * np.eye(D, dtype=np.float64)
            B_ = xc.T @ (w[:, None] * yc)
            Z[(s, h)] = np.linalg.solve(A_, B_)
            b[(s, h)] = ym - xm @ Z[(s, h)]
    return Z, b


def apply_weighted_k(kv_t, layer_map, Z, b):
    L_s, S, H, D = len(layer_map), kv_t.k.shape[1], kv_t.k.shape[2], kv_t.k.shape[3]
    pos = np.arange(S, dtype=np.float64)
    out = np.zeros((L_s, S, H, D), dtype=np.float32)
    for s in range(L_s):
        src = layer_map[s][0]
        x = de_rope_k(kv_t.k[src], pos).astype(np.float64)
        for h in range(H):
            out[s, :, h, :] = (x[:, h, :] @ Z[(s, h)] + b[(s, h)]).astype(np.float32)
    return out


def shuffled_target(calib_s, calib_t, L_s, train):
    """Student KV whose document targets come from a *different* document.

    The synthetic corpus repeats each document across ~7 questions, so
    `(i + 1) % n` would usually hit the same document and make the control a
    no-op. Group by document instead and map each group to the next unique
    document. Each target block is truncated/repeated to the source block
    length so the concatenation lines up with the (global-position) de-RoPE
    input, mirroring `fit_weighted_k_mapper`.
    """
    groups = doc_groups(train)
    uniq = np.unique(groups)
    gmap = {int(g): int(uniq[(j + 1) % len(uniq)]) for j, g in enumerate(uniq)}
    target_idx = []
    for i, g in enumerate(groups):
        members = np.where(groups == gmap[int(g)])[0]
        target_idx.append(int(members[i % len(members)]))
    n = len(calib_t)
    total = int(sum(calib_t[i].k.shape[1] for i in range(n)))
    H, D = calib_s[0].k.shape[2], calib_s[0].k.shape[3]
    out = []
    for s in range(L_s):
        y = np.zeros((total, H, D), dtype=calib_s[0].k.dtype)
        off = 0
        for i in range(n):
            Li = calib_t[i].k.shape[1]
            src = calib_s[target_idx[i]].k[s]
            m = min(Li, src.shape[0])
            y[off:off + m] = src[:m]
            if m < Li:
                y[off + m:off + Li] = src.mean(0)
            off += Li
        out.append(y)
    return KV(k=np.stack(out), v=calib_s[0].v)


def routing_stats(student, tok_s, self_kv, mapped_kv, sample, n_doc):
    q = query_of(sample)
    a_s = get_attn_map(student, tok_s, build_cache(self_kv), q, n_doc)
    a_m = get_attn_map(student, tok_s, build_cache(mapped_kv), q, n_doc)
    per_layer = [compare_maps(a_s[l], a_m[l]) for l in range(len(a_s))]
    return per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="1.7B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train = load_data(a.seed, "train")[: a.n_calib]
    test = load_data(a.seed, "test")[: a.n_eval]

    print("[A1] capture teacher then student ...", flush=True)
    teacher, tok_t, t_layers = load_teacher(a.pair)
    calib_t = capture_all(teacher, tok_t, train)
    eval_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(a.pair)
    calib_s = capture_all(student, tok_s, train)
    eval_s = capture_all(student, tok_s, test)
    if tok_t.get_vocab() != tok_s.get_vocab():
        print("[A1] WARNING: teacher/student tokenizers differ", flush=True)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)

    print("[A1] affine K baseline ...", flush=True)
    mk = fit_mapper("K", ct, cs, lmap, lam=a.lam)

    print("[A1] calibration self-attention maps ...", flush=True)
    attn_calib, calib_lengths = [], []
    for i, s in enumerate(train):
        n_doc = calib_s[i].k.shape[1]
        calib_lengths.append(n_doc)
        attn_calib.append(get_attn_map(
            student, tok_s, build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)),
            query_of(s), n_doc))
    n_q = attn_calib[0][0].shape[0]
    n_kv = calib_s[0].k.shape[2]
    ghead = group_heads(n_q, n_kv)
    weight_blocks = concat_attention_weights(attn_calib, calib_lengths, ghead,
                                             s_layers)

    print("[A1] routing-aware K fit ...", flush=True)
    Z, b = fit_weighted_k_mapper(calib_t, calib_s, weight_blocks, lmap,
                                 lam=a.lam)
    # self-consistency: all-ones weights must reproduce the affine baseline
    Z1, b1 = fit_weighted_k_mapper(calib_t, calib_s, None, lmap, lam=a.lam)
    selfcheck = 0.0
    for s in range(s_layers):
        for h in range(n_kv):
            selfcheck = max(selfcheck, float(np.max(np.abs(Z1[(s, h)]
                                                          - mk.W[("K", s, h)]))))
            selfcheck = max(selfcheck, float(np.max(np.abs(b1[(s, h)]
                                                          - mk.bias[("K", s, h)]))))
    print(f"[A1] self-check max|routing(w=1) - affine| = {selfcheck:.3e}",
          flush=True)
    assert selfcheck < 1e-8, "routing-aware fit with w=1 is not the affine K fit"

    print("[A1] shuffled-target K fit ...", flush=True)
    tgt_shuf = shuffled_target(calib_s, calib_t, s_layers, train)
    Zs, bs = fit_weighted_k_mapper(calib_t, calib_s, weight_blocks, lmap,
                                   lam=a.lam, target_s=tgt_shuf)
    del Z1, b1, weight_blocks, attn_calib, ct, cs, calib_t, calib_s
    gc.collect()

    affine_k = [mk.transform(eval_t[i].k, lmap, kv_kind="K",
                             positions=np.arange(eval_t[i].k.shape[1],
                                                dtype=np.float64),
                             de_rope_fn=de_rope_k).astype(np.float32)
                for i in range(len(test))]
    routing_k = [apply_weighted_k(eval_t[i], lmap, Z, b) for i in range(len(test))]
    shuf_k = [apply_weighted_k(eval_t[i], lmap, Zs, bs) for i in range(len(test))]
    raw_k = [raw_map_teacher(eval_t[i], lmap, s_layers).k for i in range(len(test))]

    arms_k = {
        "K-affine": affine_k,
        "K-routing": routing_k,
        "K-routing-shuf": shuf_k,
        "K-raw": raw_k,
    }

    print("[A1] scoring arms (LL + corrected EM + routing) ...", flush=True)
    rows = []
    routing_layer_acc = {name: [] for name in arms_k}
    for i, s in enumerate(test):
        n_doc = eval_s[i].k.shape[1]
        row = {"id": s["id"]}
        self_kv = KV(k=eval_s[i].k, v=eval_s[i].v)
        ll, em = score_arm(student, tok_s, self_kv, s)
        row["Self"], row["Self_em"] = ll, float(em)
        for name, km in arms_k.items():
            ll, em = score_arm(student, tok_s, KV(k=km[i], v=eval_s[i].v), s)
            row[name], row[name + "_em"] = ll, float(em)
            pl = routing_stats(student, tok_s, self_kv,
                               KV(k=km[i], v=eval_s[i].v), s, n_doc)
            row[name + "_routing"] = {
                "tv_mean": float(np.mean([p["total_variation"] for p in pl])),
                "top1_mean": float(np.mean([p["top1_agreement"] for p in pl])),
                "cosine_mean": float(np.mean([p["cosine"] for p in pl])),
            }
            routing_layer_acc[name].append(
                [p["total_variation"] for p in pl])
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  [{i + 1}/{len(test)}]", flush=True)

    groups = doc_groups(test)
    keys = list(arms_k)
    summary = summarize_rows(rows, keys, base="Self", seed=a.seed, groups=groups)

    routing_agg = {}
    for name in keys:
        arr = np.array(routing_layer_acc[name])          # (n_eval, L_s)
        routing_agg[name] = {
            "tv_mean": float(arr.mean()),
            "tv_by_layer": arr.mean(axis=0).tolist(),
            "top1_mean": float(np.mean([r[name + "_routing"]["top1_mean"]
                                        for r in rows])),
            "cosine_mean": float(np.mean([r[name + "_routing"]["cosine_mean"]
                                          for r in rows])),
        }

    # gate evaluation (pre-registered branches)
    tv_aff = routing_agg["K-affine"]["tv_mean"]
    tv_rou = routing_agg["K-routing"]["tv_mean"]
    drop = (tv_aff - tv_rou) / max(tv_aff, 1e-12)
    em_ci = summary["K-routing"].get("EM_ci95_clustered") or [0.0, 0.0]
    em_excludes_zero = em_ci[0] > 0.0
    if drop >= 0.30 and em_excludes_zero:
        branch = "intervention holds"
    elif drop >= 0.30:
        branch = "routing gap closed but task arm still at the floor"
    elif drop > 0.0:
        branch = "small routing improvement, below the 30% gate"
    else:
        branch = "no routing improvement (mapper-family limitation)"
    gate = {
        "tv_affine": float(tv_aff), "tv_routing": float(tv_rou),
        "relative_tv_drop": float(drop), "tv_drop_gate": 0.30,
        "routing_em_ci95_clustered": [float(x) for x in em_ci],
        "routing_em_ci_excludes_zero": bool(em_excludes_zero),
        "branch": branch,
    }

    report = {
        "task": "phaseB_routingkey",
        "pair": a.pair, "seed": a.seed,
        "n_calib": a.n_calib, "n_eval": a.n_eval, "lam": a.lam,
        "n_query_heads": int(n_q), "n_kv_heads": int(n_kv),
        "selfcheck_max_abs_diff_w1_vs_affine": float(selfcheck),
        "routing": routing_agg,
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    out = a.output or (f"{REPORT_DIR}/phaseB_routingkey_{a.pair}"
                       f"_seed{a.seed}.json")
    save(report, out)
    print(f"\n=== A1 routing-aware K (pair={a.pair} seed={a.seed}) ===")
    for name in keys:
        r = summary[name]
        print(f"  {name:16s} dLL={r['delta_vs_self']:+.3f} "
              f"EM={r['EM']:.3f} TV={routing_agg[name]['tv_mean']:.4f} "
              f"top1={routing_agg[name]['top1_mean']:.3f}")
    print(f"  gate: {gate}")


if __name__ == "__main__":
    main()
