"""A3: within-domain repair on the second domain (audit3 section 8).

The archived SQuAD result calibrates the mapper and the adapter on the
synthetic distribution and evaluates on SQuAD without retraining, so it shows
that the fix does not transfer *across* domains. This script asks the other
question: does the repair mechanism replicate *inside* the second domain?

Protocol
--------
SQuAD has 30 items over 30 distinct documents, so we split documents (never
items): `--split-seed D` draws 15 calibration documents and holds out the other
15. Everything (K mapper, V mappers, optional adapter) is fit on SQuAD
calibration only; the held-out split is never touched.

Pre-registered gate (REVISION_PLAN3 II.A3):
  held-out V-only (or Joint) >= 0.5 x Self_heldout with a document bootstrap
  CI95 excluding 0 in >= 2 of 3 splits, and the shuffled-target control below
  half of the real mapper  => the mechanism replicates within a domain and only
  the repair parameters are domain-specific;
  all arms ~ 0            => consumer compatibility is bound to the task
  distribution, a stronger negative result.

Usage:
  python3 experiments/phaseB_squad_within.py --split-seed 0 --n-calib 15 --n-eval 15 \
    --v-mappers affine,outaware,wo --adapter
"""
from __future__ import annotations

import argparse
import json

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
    pos_of,
    layer_map_proportional,
    load_student,
    load_teacher,
    map_teacher,
    query_of,
    score_arm,
    stack_kv,
    summarize_rows,
)
from phaseB_mechanism import apply_output_aware, fit_output_aware_mapper, get_attn_map
from phaseB_outaware import apply_wo_aware, fit_wo_aware_mapper

SQUAD_PATH = "/workspace/v3/data/squad_test_seed0.json"


def split_documents(samples: list, split_seed: int, n_calib: int):
    """Document-disjoint calibration/evaluation split (each item has its own doc)."""
    perm = np.random.RandomState(split_seed).permutation(len(samples))
    calib_idx = sorted(int(i) for i in perm[:n_calib])
    eval_idx = sorted(int(i) for i in perm[n_calib:])
    return calib_idx, eval_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="8B_0.6B", choices=list(PAIRS))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=15)
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--v-mappers", default="affine,outaware,wo",
                    help="comma-separated: affine,outaware,wo")
    ap.add_argument("--adapter", action="store_true",
                    help="also train a rank-8 adapter on SQuAD calibration")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    v_mappers = tuple(v.strip() for v in a.v_mappers.split(",") if v.strip())

    torch.manual_seed(a.split_seed)
    np.random.seed(a.split_seed)

    with open(SQUAD_PATH) as f:
        squad = json.load(f)
    calib_idx, eval_idx = split_documents(squad, a.split_seed, a.n_calib)
    calib = [squad[i] for i in calib_idx]
    eval_set = [squad[i] for i in eval_idx][: a.n_eval]
    calib_docs = {s["doc"] for s in calib}
    eval_docs = {s["doc"] for s in eval_set}
    assert not (calib_docs & eval_docs), "calibration and held-out documents overlap"

    print(f"[within] split {a.split_seed}: calib={len(calib)} docs, "
          f"held-out={len(eval_set)} docs (disjoint)", flush=True)
    teacher, tok_t, t_layers = load_teacher(a.pair)
    calib_t = capture_all(teacher, tok_t, calib)
    eval_t = capture_all(teacher, tok_t, eval_set)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(a.pair)
    calib_s = capture_all(student, tok_s, calib)
    eval_s = capture_all(student, tok_s, eval_set)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)

    print("[within] calibration attention maps ...", flush=True)
    attn_calib = []
    for i, s in enumerate(calib):
        attn_calib.append(get_attn_map(
            student, tok_s,
            build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)),
            query_of(s), calib_s[i].k.shape[1]))

    def apply_oa(kv_t, W, b):
        L_s = len(lmap)
        _, S, H, D = kv_t.k.shape
        v = np.zeros((L_s, S, H, D), dtype=np.float64)
        for l in range(L_s):
            src = lmap[l][0]
            for h in range(H):
                v[l, :, h, :] = (kv_t.v[src][:, h, :].astype(np.float64)
                                 @ W[(l, h)] + b[(l, h)])
        return v.astype(np.float32)

    # The affine V mapper is always fit: it defines the Joint arm and the
    # adapter's training states, and it costs one ridge solve.
    mv = fit_mapper("V", ct, cs, lmap)
    W_oa = b_oa = W_sh = b_sh = W_wo = None
    if "outaware" in v_mappers:
        print("[within] fitting output-aware V mapper (real targets) ...", flush=True)
        W_oa, b_oa = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap)
        groups = doc_groups(calib)
        uniq = np.unique(groups)
        gmap = {g: uniq[(j + 1) % len(uniq)] for j, g in enumerate(uniq)}
        target_idx = []
        for i, g in enumerate(groups):
            members = np.where(groups == gmap[g])[0]
            target_idx.append(int(members[i % len(members)]))
        calib_s_shuf = [calib_s[j] for j in target_idx]
        print("[within] fitting output-aware V mapper (shuffled targets) ...",
              flush=True)
        W_sh, b_sh = fit_output_aware_mapper(calib_t, calib_s, attn_calib, lmap,
                                             target_s=calib_s_shuf)
    if "wo" in v_mappers:
        print("[within] fitting W_O-aware V mapper ...", flush=True)
        W_wo = fit_wo_aware_mapper(student, calib_t, calib_s, attn_calib, lmap)
    del ct, cs
    torch.cuda.empty_cache()

    def v_affine(kv_t):
        return mv.transform(kv_t.v, lmap, kv_kind="V", positions=pos_of(kv_t),
                            de_rope_fn=None).astype(np.float32)

    def k_mapped(kv_t):
        return mk.transform(kv_t.k, lmap, kv_kind="K", positions=pos_of(kv_t),
                            de_rope_fn=de_rope_k).astype(np.float32)

    mapped = [KV(k=k_mapped(eval_t[i]), v=v_affine(eval_t[i]))
              for i in range(len(eval_set))]
    v_oa = [apply_oa(eval_t[i], W_oa, b_oa) for i in range(len(eval_set))] \
        if W_oa is not None else None
    v_sh = [apply_oa(eval_t[i], W_sh, b_sh) for i in range(len(eval_set))] \
        if W_sh is not None else None
    v_wo = [apply_wo_aware(eval_t[i], lmap, W_wo) for i in range(len(eval_set))] \
        if W_wo is not None else None
    mapped_calib = [map_teacher(mk, mv, calib_t[i], lmap) for i in range(len(calib))]

    def rows_with(tag_prefix=""):
        rows = []
        for i, s in enumerate(eval_set):
            arms = {
                "Self": KV(k=eval_s[i].k, v=eval_s[i].v),
                "K-only": KV(k=mapped[i].k, v=eval_s[i].v),
            }
            arms["V-only-affine"] = KV(k=eval_s[i].k, v=v_affine(eval_t[i]))
            arms["Joint-affine"] = KV(k=mapped[i].k, v=v_affine(eval_t[i]))
            if v_oa is not None:
                arms["V-only-outaware"] = KV(k=eval_s[i].k, v=v_oa[i])
                arms["V-only-outaware-shuf"] = KV(k=eval_s[i].k, v=v_sh[i])
            if v_wo is not None:
                arms["V-only-woaware"] = KV(k=eval_s[i].k, v=v_wo[i])
                arms["Joint-woaware"] = KV(k=mapped[i].k, v=v_wo[i])
            row = {"id": s["id"], "answer": s["answer"]}
            for name, kv in arms.items():
                ll, em = score_arm(student, tok_s, kv, s)
                row[tag_prefix + name] = ll
                row[tag_prefix + name + "_em"] = float(em)
            rows.append(row)
            if (i + 1) % 5 == 0:
                print(f"  [{tag_prefix or 'no-adapter'} {i + 1}/{len(eval_set)}]",
                      flush=True)
        return rows

    rows_no = rows_with()
    rows_ad = None
    if a.adapter:
        from phaseB_adapter import install_adapters, remove_adapters, train_adapter
        print(f"[within] training rank-{a.rank} adapter on SQuAD calibration "
              f"({a.epochs} epochs) ...", flush=True)
        doc_states = [KV(k=mapped_calib[i].k, v=mapped_calib[i].v)
                      for i in range(len(calib))]
        adapters = install_adapters(student, a.rank, ("o_proj",))
        train_adapter(student, tok_s, calib, doc_states, a.epochs, 1e-3,
                      a.split_seed, adapters, tag="within-squad")
        rows_ad = rows_with(tag_prefix="adapted_")
        remove_adapters(student, ("o_proj",))

    base_keys = [k for k in rows_no[0] if not k.endswith("_em") and k != "id"
                 and k != "answer"]
    summary = summarize_rows(rows_no, base_keys, base="Self", seed=a.split_seed)
    adapted_keys = ["adapted_" + k for k in base_keys]
    if rows_ad is not None:
        summary_adapted = summarize_rows(rows_ad, adapted_keys,
                                         base="adapted_Self", seed=a.split_seed)
    else:
        summary_adapted = None

    self_em = float(np.mean([r["Self_em"] for r in rows_no]))
    report = {
        "task": "phaseB_squad_within",
        "pair": a.pair,
        "split_seed": a.split_seed,
        "v_mappers": list(v_mappers),
        "adapter": bool(a.adapter),
        "rank": a.rank, "epochs": a.epochs,
        "calib_domain": "SQuAD", "eval_domain": "SQuAD (held-out documents)",
        "n_calib": len(calib), "n_eval": len(eval_set),
        "calib_ids": [s["id"] for s in calib],
        "eval_ids": [s["id"] for s in eval_set],
        "self_EM": self_em,
        "summary": summary,
        "summary_adapted": summary_adapted,
        "rows": rows_no,
        "rows_adapted": rows_ad,
    }
    out = a.output or (f"/workspace/v3/reports/phaseB_squadwithin_split{a.split_seed}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[within] saved: {out}")
    print(f"Self EM (held-out) = {self_em:.3f}")
    for k in base_keys:
        r = summary[k]
        ci = r.get("EM_ci95")
        if ci:
            print(f"  {k:24s} EM={r['EM']:.3f} CI95=[{ci[0]:.3f}, {ci[1]:.3f}]")
        else:
            print(f"  {k:24s} EM={r['EM']:.3f}")
    if summary_adapted is not None:
        print("  -- with within-domain adapter --")
        for k in adapted_keys:
            r = summary_adapted[k]
            print(f"  {k:24s} EM={r['EM']:.3f}")


if __name__ == "__main__":
    main()
