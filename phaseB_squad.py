"""Corrected SQuAD (second-domain) four-arm evaluation, cross-domain mapper.

Mirrors phase7_second_domain.py but uses the corrected evaluator
(phaseB_common.score_arm): fresh cache for generation + no double-feed. The
mapper is calibrated on the synthetic OOD training split and evaluated on SQuAD,
so the mapper never sees the evaluation domain.

Usage:
  python3 phaseB_squad.py --seed 0 --n-calib 70 --n-eval 30
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from phaseB_common import (
    KV,
    build_cache,
    capture_all,
    de_rope_k,
    fit_mapper,
    greedy_answer_fixed,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    map_teacher,
    query_of,
    score_arm,
    stack_kv,
    summarize_rows,
)
from phaseB_mechanism import (apply_output_aware, fit_output_aware_mapper,
                              get_attn_map)
from phase0_g0 import answer_loglik, exact_match

DATA = "/workspace/v3/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--v-mapper", default="affine", choices=["affine", "outaware"])
    ap.add_argument("--adapter", action="store_true",
                    help="train a rank-8 adapter on OOD calib, eval on SQuAD")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    calib = load_data(a.seed, "train")[:a.n_calib]
    eval_set = json.load(open(f"{DATA}/squad_test_seed0.json"))[:a.n_eval]
    probe = json.load(open(f"{DATA}/nq_open_probe_seed0.json"))

    pair = "8B_0.6B"
    teacher, tok_t, t_layers = load_teacher(pair)
    calib_t = capture_all(teacher, tok_t, calib)
    eval_t = capture_all(teacher, tok_t, eval_set)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(pair)
    calib_s = capture_all(student, tok_s, calib)
    eval_s = capture_all(student, tok_s, eval_set)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    Woa = boa = mv = None
    if a.v_mapper == "outaware":
        attn_calib = []
        for i in range(len(calib)):
            attn_calib.append(get_attn_map(
                student, tok_s,
                build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)),
                query_of(calib[i]), calib_s[i].k.shape[1]))
        Woa, boa = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap)
    else:
        mv = fit_mapper("V", ct, cs, lmap)
    pos_calib = np.arange(ct.k.shape[1], dtype=np.float64)
    del ct, cs
    torch.cuda.empty_cache()

    def map_kv_at(kv_t):
        pos = np.arange(kv_t.k.shape[1], dtype=np.float64)
        km = mk.transform(kv_t.k, lmap, kv_kind="K", positions=pos, de_rope_fn=de_rope_k)
        if a.v_mapper == "outaware":
            vm = apply_output_aware(kv_t, lmap, Woa, boa)
        else:
            vm = mv.transform(kv_t.v, lmap, kv_kind="V", positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    mapped = [map_kv_at(eval_t[i]) for i in range(len(eval_set))]
    mapped_calib = [map_kv_at(calib_t[i]) for i in range(len(calib))]

    if a.adapter:
        from phaseB_adapter import install_adapters, remove_adapters, train_adapter
        doc_states = [KV(k=mapped_calib[i].k, v=mapped_calib[i].v)
                      for i in range(len(calib))]
        adapters = install_adapters(student, a.rank, ("o_proj",))
        train_adapter(student, tok_s, calib, doc_states, a.epochs, 1e-3, a.seed,
                      adapters, tag="squad")

    rows = []
    for i, s in enumerate(eval_set):
        row = {"id": s["id"], "answer": s["answer"]}
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
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_set)}]", flush=True)

    summary = summarize_rows(rows, ["K-only", "V-only", "Joint"], base="Self",
                             seed=a.seed)

    # novelty probe with corrected greedy
    p_lls, p_ems = [], []
    for s in probe:
        q = "Question: " + s["q"] + "\nAnswer:"
        p_ems.append(float(exact_match(
            greedy_answer_fixed(student, tok_s, None, q), s["answer"])))
        p_lls.append(answer_loglik(student, tok_s, None, q, s["answer"]))
    probe_s = {"n": len(probe), "em": float(np.mean(p_ems)),
               "mean_ll": float(np.mean(p_lls))}

    if a.adapter:
        remove_adapters(student, ("o_proj",))
    report = {
        "task": "phaseB_squad_corrected",
        "pair": pair, "seed": a.seed,
        "v_mapper": a.v_mapper, "adapter": bool(a.adapter),
        "calib_domain": "OOD(train_v2)", "eval_domain": "SQuAD",
        "n_calib": a.n_calib, "n_eval": a.n_eval,
        "self_EM": float(np.mean([r["Self_em"] for r in rows])),
        "summary": summary, "novelty_probe_student": probe_s, "rows": rows,
    }
    out = a.output or f"/workspace/v3/reports/phaseB_squad_seed{a.seed}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[squad] saved: {out}")
    print(f"Self EM={report['self_EM']:.3f}")
    for k in ["K-only", "V-only", "Joint"]:
        r = summary[k]
        print(f"  {k:7s} EM={r['EM']:.3f} dLL={r['delta_vs_self']:+.3f} p={r['wilcoxon_p']:.2e}")


if __name__ == "__main__":
    main()
