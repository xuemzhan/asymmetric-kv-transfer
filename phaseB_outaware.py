"""Optimization: proper W_O-aware V mapper + content-specificity control.

Extends B3 (attention-output-aware V mapper) in two ways the reviewer asked for:

1. W_O-aware objective. The student attention output per query head q is
   O_q = (A_q V_h) W_O_q, with W_O_q the q-slice of o_proj. We fit, per KV head
   h, a single map M_h satisfying M_h W_O_q ~= N_q for every query head q in
   the group, where N_q is the ridge solution of A_q V_T,h -> A_q V_S,h W_O_q.
   This keeps the mapping conservative in the subspace o_proj actually reads.

2. Content-specificity control. The same attention-output-aware mapper is
   refit with *shuffled* targets (A V_T -> A V_S of a different document). If
   the real mapper recovers EM but the shuffled one does not, the recovered
   signal is document-specific content, not a generic "student-like cache".

Usage:
  python3 phaseB_outaware.py --pair 1.7B_0.6B --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    build_cache,
    capture_all,
    doc_groups,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    map_teacher,
    score_arm,
    stack_kv,
)
from phaseB_mechanism import fit_output_aware_mapper, get_attn_map


def fit_wo_aware_mapper(student, calib_t, calib_s, attn_by_sample, layer_map, lam=1e-3):
    """Per-KV-head linear map M_h satisfying M_h W_O_q ~= N_q for all q heads.

    Linear only (no intercept): the downstream o_proj consumes V linearly, so
    the intercept is unidentifiable through W_O.
    """
    L_s = calib_s[0].v.shape[0]
    H_kv = calib_s[0].v.shape[2]
    D = calib_s[0].v.shape[3]
    Wm = {}
    for l in range(L_s):
        src = layer_map[l][0]
        W = student.model.layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy()
        H_q = W.shape[1] // D
        groups = H_q // H_kv
        for h in range(H_kv):
            Ns, Ws = [], []
            for gi in range(groups):
                q = h * groups + gi
                Wq = W[:, q * D:(q + 1) * D].T.astype(np.float64)  # (D, hidden)
                Xs, Ys = [], []
                for i in range(len(calib_t)):
                    A = attn_by_sample[i][l][q]              # (Sq, n_doc)
                    Vt = calib_t[i].v[src][:, h, :]          # (n_doc, D)
                    Vs = calib_s[i].v[l][:, h, :]
                    Xs.append(A @ Vt)
                    Ys.append((A @ Vs) @ Wq)
                X = np.concatenate(Xs, 0)
                Y = np.concatenate(Ys, 0)
                Xm, Ym = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
                N = np.linalg.solve((X - Xm).T @ (X - Xm) + lam * np.eye(D),
                                    (X - Xm).T @ (Y - Ym))
                Ns.append(N)
                Ws.append(Wq)
            Nstack = np.concatenate(Ns, axis=1)   # (D, groups*hidden)
            Wstack = np.concatenate(Ws, axis=1)   # (D, groups*hidden)
            Wm[(l, h)] = Nstack @ np.linalg.pinv(Wstack)   # (D, D)
    return Wm


def apply_wo_aware(kv_t, layer_map, Wm):
    L_s = len(layer_map)
    L_t, S, H, D = kv_t.k.shape
    v = np.zeros((L_s, S, H, D), dtype=np.float64)
    for l in range(L_s):
        src = layer_map[l][0]
        for h in range(H):
            v[l, :, h, :] = kv_t.v[src][:, h, :].astype(np.float64) @ Wm[(l, h)]
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="1.7B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train = load_data(a.seed, "train")[:a.n_calib]
    test = load_data(a.seed, "test")[:a.n_eval]
    teacher, tok_t, t_layers = load_teacher(a.pair)
    calib_t = capture_all(teacher, tok_t, train)
    eval_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(a.pair)
    calib_s = capture_all(student, tok_s, train)
    eval_s = capture_all(student, tok_s, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    mapped = [map_teacher(mk, mv, eval_t[i], lmap) for i in range(len(test))]

    def attn_of(samples, kvs):
        out = []
        for i, s in enumerate(samples):
            n_doc = kvs[i].k.shape[1]
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            out.append(get_attn_map(student, tok_s,
                                   build_cache(KV(k=kvs[i].k, v=kvs[i].v)), q, n_doc))
        return out

    print("[opt] calibration attention ...", flush=True)
    attn_calib = attn_of(train, calib_s)

    print("[opt] fitting output-aware (real targets) ...", flush=True)
    Wr, br = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap, lam=1e-3)

    # content-specificity control: target V from a *different document*
    groups = doc_groups(train)
    uniq = np.unique(groups)
    gmap = {g: uniq[(j + 1) % len(uniq)] for j, g in enumerate(uniq)}
    target_idx = []
    for i, g in enumerate(groups):
        members = np.where(groups == gmap[g])[0]
        target_idx.append(int(members[i % len(members)]))
    calib_s_shuf = [calib_s[j] for j in target_idx]
    print("[opt] fitting output-aware (shuffled targets) ...", flush=True)
    Ws, bs = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap,
                                     lam=1e-3, target_s=calib_s_shuf)

    print("[opt] fitting W_O-aware ...", flush=True)
    Wwo = fit_wo_aware_mapper(student, calib_t, calib_s, attn_calib, lmap)

    def apply_oa(kv_t, W, b):
        L_s = len(lmap)
        _, S, H, D = kv_t.k.shape
        v = np.zeros((L_s, S, H, D), dtype=np.float64)
        for l in range(L_s):
            src = lmap[l][0]
            for h in range(H):
                v[l, :, h, :] = kv_t.v[src][:, h, :].astype(np.float64) @ W[(l, h)] + b[(l, h)]
        return v.astype(np.float32)

    v_real = [apply_oa(eval_t[i], Wr, br) for i in range(len(test))]
    v_shuf = [apply_oa(eval_t[i], Ws, bs) for i in range(len(test))]
    v_wo = [apply_wo_aware(eval_t[i], lmap, Wwo) for i in range(len(test))]

    rows = []
    for i, s in enumerate(test):
        arms = {
            "Self": KV(k=eval_s[i].k, v=eval_s[i].v),
            "V-Affine": KV(k=eval_s[i].k, v=mapped[i].v),
            "V-OutAware": KV(k=eval_s[i].k, v=v_real[i]),
            "V-OutAware-shuf": KV(k=eval_s[i].k, v=v_shuf[i]),
            "V-WOAware": KV(k=eval_s[i].k, v=v_wo[i]),
        }
        row = {"id": s["id"]}
        for name, kv in arms.items():
            ll, em = score_arm(student, tok_s, kv, s)
            row[name] = ll
            row[name + "_em"] = float(em)
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(test)}]", flush=True)

    out = a.output or f"/workspace/v3/reports/phaseB_outaware_{a.pair}_seed{a.seed}.json"
    import json
    with open(out, "w") as f:
        json.dump({"task": "phaseB_outaware", "pair": a.pair, "seed": a.seed,
                   "n_calib": a.n_calib, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"[opt] saved: {out}")
    for name in ["Self", "V-Affine", "V-OutAware", "V-OutAware-shuf", "V-WOAware"]:
        em = np.mean([r[name + "_em"] for r in rows])
        ll = np.mean([r[name] for r in rows])
        print(f"  {name:18s} EM={em:.3f} LL={ll:.3f}")


if __name__ == "__main__":
    main()
