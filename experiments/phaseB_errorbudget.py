"""A4: state-space vs consumption-space error budget (audit3 section 11).

The paper's central mechanism claim is that raw representation alignment is
insufficient and alignment in the receiver's consumption space is what tracks
task performance, but `tab:mappers` only reports EM. This script puts the
missing quantities next to it, for every mapper variant on the same run:

  e_raw  = ||V_hat - V_S||_F^2            / ||V_S||_F^2
  e_attn = ||A_S V_hat - A_S V_S||_F^2    / ||A_S V_S||_F^2
  e_wo   = ||(A_S V_hat - A_S V_S) W_O||_F^2 / ||A_S V_S W_O||_F^2

with A_S the student's own attention map (Self cache) and W_O the student's
`o_proj` weight sliced per query head. Each variant is also scored for
corrected V-only EM.

Reading (pre-registered, REVISION_PLAN3 II.A4):
  if the affine mapper minimises e_raw while its EM is the lowest, and
  e_attn / e_wo track the EM ordering, the paper's "where you align matters"
  statement is backed by numbers; if e_wo does not track EM, the statement must
  be downgraded in the text.

Usage:
  python3 experiments/phaseB_errorbudget.py --pair 1.7B_0.6B --seed 0 \
    --n-calib 70 --n-eval 56
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from phaseB_common import (
    HEAD_DIM,
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
    pos_of,
    raw_map_teacher,
    score_arm,
    stack_kv,
)
from phaseB_mechanism import apply_output_aware, fit_output_aware_mapper, get_attn_map
from phaseB_outaware import apply_wo_aware, fit_wo_aware_mapper


def apply_oa(kv_t, layer_map, W, b):
    L_s = len(layer_map)
    _, S, H, D = kv_t.k.shape
    v = np.zeros((L_s, S, H, D), dtype=np.float64)
    for l in range(L_s):
        src = layer_map[l][0]
        for h in range(H):
            v[l, :, h, :] = (kv_t.v[src][:, h, :].astype(np.float64)
                             @ W[(l, h)] + b[(l, h)])
    return v.astype(np.float32)


def error_budget(v_hat: np.ndarray, v_s: np.ndarray, attn_layers: list,
                 wo_slices: dict, group_of_head: dict) -> dict:
    """Raw / attention-output / o_proj-space relative errors for one sample."""
    num_raw = den_raw = 0.0
    num_attn = den_attn = 0.0
    num_wo = den_wo = 0.0
    n_layers = min(v_hat.shape[0], v_s.shape[0])
    for l in range(n_layers):
        A_l = attn_layers[l]                    # (H_q, S_q, n_doc)
        n_kv_heads = v_hat.shape[2]
        n_doc = min(A_l.shape[-1], v_hat.shape[1], v_s.shape[1])
        diff = (v_hat[l][:n_doc].astype(np.float64)
                - v_s[l][:n_doc].astype(np.float64))
        num_raw += float((diff ** 2).sum())
        den_raw += float((v_s[l][:n_doc].astype(np.float64) ** 2).sum())
        for h in range(n_kv_heads):
            for q in group_of_head[h]:
                A = A_l[q][:, :n_doc].astype(np.float64)
                if A.shape[0] == 0:
                    continue
                x = A @ v_hat[l][:n_doc, h, :].astype(np.float64)
                y = A @ v_s[l][:n_doc, h, :].astype(np.float64)
                d = x - y
                num_attn += float((d ** 2).sum())
                den_attn += float((y ** 2).sum())
                Wq = wo_slices[(l, q)]          # (hidden, D)
                dw = d @ Wq
                yw = y @ Wq
                num_wo += float((dw ** 2).sum())
                den_wo += float((yw ** 2).sum())
    eps = 1e-12
    return {
        "e_raw": num_raw / (den_raw + eps),
        "e_attn": num_attn / (den_attn + eps),
        "e_wo": num_wo / (den_wo + eps),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


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
    train = load_data(a.seed, "train")[: a.n_calib]
    test = load_data(a.seed, "test")[: a.n_eval]

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

    print("[budget] calibration attention maps ...", flush=True)
    attn_calib = []
    for i, s in enumerate(train):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        attn_calib.append(get_attn_map(
            student, tok_s, build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)),
            q, calib_s[i].k.shape[1]))
    W_oa, b_oa = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap)

    groups = doc_groups(train)
    uniq = np.unique(groups)
    gmap = {g: uniq[(j + 1) % len(uniq)] for j, g in enumerate(uniq)}
    target_idx = []
    for i, g in enumerate(groups):
        members = np.where(groups == gmap[g])[0]
        target_idx.append(int(members[i % len(members)]))
    calib_s_shuf = [calib_s[j] for j in target_idx]
    W_sh, b_sh = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap,
                                         target_s=calib_s_shuf)
    print("[budget] fitting W_O-aware V mapper ...", flush=True)
    W_wo = fit_wo_aware_mapper(student, calib_t, calib_s, attn_calib, lmap)
    del ct, cs
    torch.cuda.empty_cache()

    # student weight-side structure (identical for every layer of a model)
    D = int(getattr(student.config, "head_dim", None) or HEAD_DIM)
    n_kv = student.config.num_key_value_heads
    n_q = student.config.num_attention_heads
    group_of_head = {h: [h * (n_q // n_kv) + g for g in range(n_q // n_kv)]
                     for h in range(n_kv)}
    wo_slices = {}
    for l in range(s_layers):
        Wl = student.model.layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for q in range(n_q):
            wo_slices[(l, q)] = Wl[:, q * D:(q + 1) * D].astype(np.float64)

    variants = {
        "raw": [raw_map_teacher(eval_t[i], lmap, s_layers).v for i in range(len(test))],
        "affine": [mv.transform(eval_t[i].v, lmap, kv_kind="V",
                                positions=pos_of(eval_t[i]),
                                de_rope_fn=None).astype(np.float32)
                   for i in range(len(test))],
        "outaware": [apply_oa(eval_t[i], lmap, W_oa, b_oa) for i in range(len(test))],
        "outaware-shuffled": [apply_oa(eval_t[i], lmap, W_sh, b_sh)
                              for i in range(len(test))],
        "woaware": [apply_wo_aware(eval_t[i], lmap, W_wo) for i in range(len(test))],
    }

    print("[budget] evaluation attention maps (student Self cache) ...", flush=True)
    rows = []
    for i, s in enumerate(test):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        attn = get_attn_map(student, tok_s,
                            build_cache(KV(k=eval_s[i].k, v=eval_s[i].v)),
                            q, eval_s[i].k.shape[1])
        row = {"id": s["id"]}
        for name, v_hat in variants.items():
            stats = error_budget(v_hat[i], eval_s[i].v.astype(np.float32),
                                 attn, wo_slices, group_of_head)
            _, em = score_arm(student, tok_s, KV(k=eval_s[i].k, v=v_hat[i]), s)
            row[name] = {**stats, "EM": float(em)}
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(test)}]", flush=True)

    summary = {}
    for name in variants:
        e_raw = np.array([r[name]["e_raw"] for r in rows])
        e_attn = np.array([r[name]["e_attn"] for r in rows])
        e_wo = np.array([r[name]["e_wo"] for r in rows])
        ems = np.array([r[name]["EM"] for r in rows])
        summary[name] = {
            "e_raw_mean": float(e_raw.mean()), "e_raw_std": float(e_raw.std()),
            "e_attn_mean": float(e_attn.mean()), "e_attn_std": float(e_attn.std()),
            "e_wo_mean": float(e_wo.mean()), "e_wo_std": float(e_wo.std()),
            "EM": float(ems.mean()), "EM_std": float(ems.std()),
        }

    names = list(variants)
    em_vec = np.array([summary[n]["EM"] for n in names])
    corr = {
        "e_raw_vs_EM": spearman(np.array([summary[n]["e_raw_mean"] for n in names]), em_vec),
        "e_attn_vs_EM": spearman(np.array([summary[n]["e_attn_mean"] for n in names]), em_vec),
        "e_wo_vs_EM": spearman(np.array([summary[n]["e_wo_mean"] for n in names]), em_vec),
        "n_variants": len(names),
        "note": "Spearman over 5 mapper variants in one run; n is too small for a p-value.",
    }

    report = {
        "task": "phaseB_errorbudget",
        "pair": a.pair, "seed": a.seed,
        "n_calib": a.n_calib, "n_eval": a.n_eval,
        "self_EM": float(np.mean([
            score_arm(student, tok_s, KV(k=eval_s[i].k, v=eval_s[i].v), test[i])[1]
            for i in range(len(test))])),
        "summary": summary,
        "spearman": corr,
        "rows": rows,
    }
    out = a.output or (f"/workspace/v3/reports/phaseB_errorbudget_{a.pair}"
                       f"_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[budget] saved: {out}")
    for name in names:
        s = summary[name]
        print(f"  {name:18s} e_raw={s['e_raw_mean']:.4f} e_attn={s['e_attn_mean']:.4f} "
              f"e_wo={s['e_wo_mean']:.4f} EM={s['EM']:.3f}")
    print(f"  spearman: {corr}")


if __name__ == "__main__":
    main()
