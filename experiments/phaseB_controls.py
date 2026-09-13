"""B4: Control experiments for the flagship pair.

Answers reviewer §12: are the observed gains example-specific teacher-state
transfer, or generic distribution/calibration effects?

Controls (all on the same fitted Affine mapper and the same eval samples):
  - Self / K-only / V-only / Joint          (published four arms)
  - Wrong-document cache                    (mapped teacher KV from a different sample)
  - Zero cache                              (K and/or V set to 0)
  - Moment-matched Gaussian random cache    (per layer/head/dim mean+std of student KV)
  - Token-shuffled ("deranged") cache       (exact marginal moments, destroyed order)
  - Identity / no-map                       (raw proportional teacher layer, no affine)

Usage:
  python3 experiments/phaseB_controls.py --pair 8B_0.6B --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    KV,
    capture_all,
    de_rope_k,
    doc_groups,
    capture_pair_kv,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_pair,
    map_teacher,
    raw_map_teacher,
    save,
    score_arm,
    stack_kv,
    summarize_rows,
)


def derangement(n: int, rng: np.random.RandomState) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def wrong_doc_partner(samples: list, rng: np.random.RandomState) -> np.ndarray:
    """For each sample pick another sample whose DOC differs.

    The synthetic set repeats each document across ~7 questions; a wrong-sample
    control that lands on the same document is vacuous, so we derange at the
    document-group level and return a representative sample index per group.
    """
    docs = [s["doc"] for s in samples]
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        groups.setdefault(d, []).append(i)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("wrong-doc control needs >=2 distinct documents")
    dp = derangement(len(keys), rng)
    partner = np.empty(len(samples), dtype=int)
    for gi, key in enumerate(keys):
        src_group = groups[keys[dp[gi]]]
        for i in groups[key]:
            partner[i] = src_group[i % len(src_group)]
    return partner


def rand_like_from_stats(kv: KV, rng: np.random.RandomState) -> KV:
    """Gaussian random with per-(layer, head, dim) mean/std copied from kv."""
    k = np.empty_like(kv.k)
    v = np.empty_like(kv.v)
    for arr, dst in ((kv.k, k), (kv.v, v)):
        mu = arr.mean(axis=1, keepdims=True)
        sd = arr.std(axis=1, keepdims=True) + 1e-6
        dst[:] = (rng.standard_normal(arr.shape) * sd + mu).astype(np.float32)
    return KV(k=k, v=v)


def shuffled_kv(kv: KV, rng: np.random.RandomState) -> KV:
    """Token-order shuffle with one permutation shared across layers (deranged cache)."""
    L, S, H, D = kv.k.shape
    perm = rng.permutation(S)
    return KV(k=kv.k[:, perm].copy(), v=kv.v[:, perm].copy())


def run(pair: str, seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 12345)

    train = load_data(seed, "train")[:n_calib]
    test = load_data(seed, "test")[:n_eval]
    print("[B4] capture KV (teacher then student) ...", flush=True)
    (calib_t, eval_t, calib_s, eval_s, student, tok_s,
     t_layers, s_layers) = capture_pair_kv(pair, train, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    print("[B4] fit Affine mappers ...", flush=True)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    del ct, cs
    torch.cuda.empty_cache()

    mapped_t = [map_teacher(mk, mv, eval_t[i], lmap) for i in range(len(test))]
    raw_t = [raw_map_teacher(eval_t[i], lmap, s_layers) for i in range(len(test))]
    wrong_idx = wrong_doc_partner(test, rng)

    rows = []
    for i, s in enumerate(test):
        sk, sv = eval_s[i].k, eval_s[i].v
        mk_i, mv_i = mapped_t[i].k, mapped_t[i].v
        rk_i, rv_i = raw_t[i].k, raw_t[i].v

        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"]}

        arms = {
            "Self": KV(k=sk, v=sv),
            "K-only": KV(k=mk_i, v=sv),
            "V-only": KV(k=sk, v=mv_i),
            "Joint": KV(k=mk_i, v=mv_i),
            # wrong document (use the wrong sample's own K/V pair to keep lengths)
            "Wrong_K-only": KV(k=mapped_t[wrong_idx[i]].k, v=eval_s[wrong_idx[i]].v),
            "Wrong_V-only": KV(k=eval_s[wrong_idx[i]].k, v=mapped_t[wrong_idx[i]].v),
            "Wrong_Joint": KV(k=mapped_t[wrong_idx[i]].k, v=mapped_t[wrong_idx[i]].v),
            # zero
            "Zero_K": KV(k=np.zeros_like(sk), v=sv),
            "Zero_V": KV(k=sk, v=np.zeros_like(sv)),
            "Zero_KV": KV(k=np.zeros_like(sk), v=np.zeros_like(sv)),
            # identity / no-map
            "Identity_K-only": KV(k=rk_i, v=sv),
            "Identity_V-only": KV(k=sk, v=rv_i),
            "Identity_Joint": KV(k=rk_i, v=rv_i),
        }
        rand = rand_like_from_stats(eval_s[i], rng)
        shuf = shuffled_kv(eval_s[i], rng)
        arms["Rand_K"] = KV(k=rand.k, v=sv)
        arms["Rand_V"] = KV(k=sk, v=rand.v)
        arms["Rand_KV"] = KV(k=rand.k, v=rand.v)
        arms["Shuf_K"] = KV(k=shuf.k, v=sv)
        arms["Shuf_V"] = KV(k=sk, v=shuf.v)
        arms["Shuf_KV"] = KV(k=shuf.k, v=shuf.v)

        for name, kv in arms.items():
            ll, em = score_arm(student, tok_s, kv, s, want_em=True)
            row[name] = ll
            row[name + "_em"] = float(em)
        rows.append(row)
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(test)}]", flush=True)

    keys = [k for k in rows[0] if not k.endswith("_em")
            and k not in ("id", "hop", "answer")]
    groups = doc_groups(test)
    summary = summarize_rows(rows, keys, base="Self", seed=seed, groups=groups)

    report = {
        "task": "phaseB_controls",
        "pair": pair, "seed": seed,
        "n_calib": n_calib, "n_eval": n_eval,
        "layer_map": lmap,
        "summary": summary,
        "rows": rows,
    }
    save(report, output)

    print("\n=== B4 control summary (pair=%s seed=%d) ===" % (pair, seed))
    print(f"{'arm':16s} {'LL':>8s} {'dLL':>7s} {'p':>9s} {'EM':>6s}")
    for k in keys:
        r = summary[k]
        print(f"{k:16s} {r['LL_mean']:8.3f} {r['delta_vs_self']:7.3f} "
              f"{r['wilcoxon_p']:9.2e} {r['EM']:6.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="8B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    out = a.output or f"/workspace/v3/reports/phaseB_controls_{a.pair}_seed{a.seed}.json"
    run(a.pair, a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
