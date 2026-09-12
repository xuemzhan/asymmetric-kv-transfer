"""V3 Phase 4 v2：跨模型对缩放律 — 6 pairs 4-way ablation + bootstrap + Wilcoxon

v2 变更（相对 phase4_scaling_law.py）：
1. 数据：train_v2_seed{seed}.json / test_v2_seed{seed}.json（n_calib=70, n_eval=56）
2. summary 用 bootstrap_ci95（替换 z-interval）
3. 每 pair 加 Wilcoxon: K-only/V-only/Joint vs Self
4. 3 seeds 独立运行

用法：python3 phase4_scaling_law_v2.py --seed 0
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/apcs")
sys.path.insert(0, "/workspace/v3")
from phase0_g0 import load_model, capture_kv, build_cache, answer_loglik, \
    greedy_answer, exact_match, KV, HEAD_DIM, ROPE_THETA
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope
from stats_utils import bootstrap_ci95, paired_wilcoxon_test

DATA_DIR = "/workspace/v3/data"
REPORT_DIR = "/workspace/v3/reports"

MODEL_PAIRS = {
    "8B_0.6B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master",
        "t_layers": 36, "s_layers": 28,
    },
    "4B_1.7B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master",
        "t_layers": 36, "s_layers": 28,
    },
    "4B_0.6B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master",
        "t_layers": 36, "s_layers": 28,
    },
    "8B_1.7B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master",
        "t_layers": 36, "s_layers": 28,
    },
    "1.7B_0.6B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master",
        "t_layers": 28, "s_layers": 28,
    },
    "8B_4B": {
        "teacher": "/root/.cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master",
        "student": "/root/.cache/modelscope/models/Qwen--Qwen3-4B/snapshots/master",
        "t_layers": 36, "s_layers": 36,
    },
}


def layer_map_generic(t_layers: int, s_layers: int) -> list[list[int]]:
    return [[min(t_layers - 1, round(s * t_layers / s_layers))]
            for s in range(s_layers)]


def run_pair_v2(pair_name: str, pair: dict, calib: list, eval_set: list,
                seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)
    lmap = layer_map_generic(pair["t_layers"], pair["s_layers"])

    print(f"  loading teacher ({pair['t_layers']}L) ...")
    teacher, tok_t = load_model(pair["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]
    del teacher
    torch.cuda.empty_cache()

    print(f"  loading student ({pair['s_layers']}L) ...")
    student, tok_s = load_model(pair["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]

    def stack_kv(kvs):
        return KV(
            k=np.concatenate([kv.k for kv in kvs], axis=1),
            v=np.concatenate([kv.v for kv in kvs], axis=1),
        )

    calib_t_all = stack_kv(calib_t_kv)
    calib_s_all = stack_kv(calib_s_kv)
    pos_all = np.arange(calib_t_all.k.shape[1], dtype=np.float64)

    mk = AffineMapper(lam=1e-3)
    mv = AffineMapper(lam=1e-3)
    t0 = time.time()
    mk.fit(calib_t_all.k, calib_s_all.k, lmap, kv_kind="K",
           positions=pos_all, de_rope_fn=de_rope_k)
    mv.fit(calib_t_all.v, calib_s_all.v, lmap, kv_kind="V",
           positions=pos_all, de_rope_fn=None)
    print(f"  [Affine] fit K+V {time.time()-t0:.1f}s")

    def map_teacher_kv(kv_t, pos):
        km = mk.transform(kv_t.k, lmap, kv_kind="K",
                          positions=pos, de_rope_fn=de_rope_k)
        vm = mv.transform(kv_t.v, lmap, kv_kind="V",
                          positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    # 四路消融（每样本 LL 保存用于统计）
    rows = []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        pos_i = np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64)
        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"],
               "doc_id": s.get("project", s["doc"][:40])}

        c = build_cache(eval_s_kv[i])
        row["Self"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["Self_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        m = map_teacher_kv(eval_t_kv[i], pos_i)

        c = build_cache(KV(k=m.k, v=eval_s_kv[i].v))
        row["K-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["K-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        c = build_cache(KV(k=eval_s_kv[i].k, v=m.v))
        row["V-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["V-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        c = build_cache(KV(k=m.k, v=m.v))
        row["Joint"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["Joint_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        rows.append(row)

    # 汇总 + bootstrap CI95
    summary, em, wilcoxon = {}, {}, {}
    self_lls = np.array([r["Self"] for r in rows], dtype=float)
    for key in ["Self", "K-only", "V-only", "Joint"]:
        xs = np.array([r[key] for r in rows], dtype=float)
        mean, lo, hi = bootstrap_ci95(xs, seed=seed)
        summary[key] = {"mean": float(mean), "ci95": [float(lo), float(hi)], "n": len(xs)}
        em[key] = float(np.mean([r[f"{key}_em"] for r in rows]))
    for key in ["K-only", "V-only", "Joint"]:
        xs = np.array([r[key] for r in rows], dtype=float)
        stat, p, d = paired_wilcoxon_test(xs.tolist(), self_lls.tolist())
        dmean, dlo, dhi = bootstrap_ci95(xs - self_lls, seed=seed)
        wilcoxon[key] = {
            "statistic": float(stat), "p_value": float(p), "cohens_d": float(d),
            "delta_mean": float(dmean), "delta_ci95": [float(dlo), float(dhi)],
        }

    self_ll = summary["Self"]["mean"]
    delta_k = summary["K-only"]["mean"] - self_ll
    delta_v = summary["V-only"]["mean"] - self_ll
    delta_j = summary["Joint"]["mean"] - self_ll

    return {
        "pair": pair_name,
        "summary_ll": summary,
        "exact_match": em,
        "wilcoxon_vs_self": wilcoxon,
        "deltas": {
            "K-only": float(delta_k), "V-only": float(delta_v), "Joint": float(delta_j),
        },
        "ratio": {
            "K_V_ratio": float(delta_k / max(abs(delta_v), 1e-6)) if abs(delta_v) > 1e-6 else float('inf'),
            "efficiency": float(delta_k / delta_j) if abs(delta_j) > 1e-6 else float('inf'),
        },
        "n_eval": len(eval_set),
        "rows": rows,
    }


def run_phase4_v2(seed: int, n_calib: int, n_eval: int, pairs: list[str],
                  output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train_v2_seed{seed}.json"))
    test = json.load(open(f"{DATA_DIR}/test_v2_seed{seed}.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]
    print(f"[P4v2 seed{seed}] calib={len(calib)} eval={len(eval_set)} pairs={pairs}")

    pair_results = []
    for pair_name in pairs:
        if pair_name not in MODEL_PAIRS:
            print(f"  SKIP unknown pair: {pair_name}")
            continue
        print(f"=== {pair_name} ===")
        pr = run_pair_v2(pair_name, MODEL_PAIRS[pair_name], calib, eval_set, seed)
        pair_results.append(pr)
        print(f"  Self={pr['summary_ll']['Self']['mean']:.2f} "
              f"K={pr['summary_ll']['K-only']['mean']:.2f} "
              f"V={pr['summary_ll']['V-only']['mean']:.2f} "
              f"J={pr['summary_ll']['Joint']['mean']:.2f} | "
              f"pK={pr['wilcoxon_vs_self']['K-only']['p_value']:.2e} "
              f"pV={pr['wilcoxon_vs_self']['V-only']['p_value']:.2e}")

    report = {
        "task": "phase4_scaling_law_v2",
        "seed": seed,
        "n_calib": n_calib, "n_eval": n_eval,
        "pairs": pairs,
        "pair_results": pair_results,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved: {output}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--pairs", nargs="+",
                    default=["8B_0.6B", "4B_1.7B", "4B_0.6B", "8B_1.7B", "1.7B_0.6B", "8B_4B"])
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase4_scaling_law_v2_seed{a.seed}.json"
    run_phase4_v2(a.seed, a.n_calib, a.n_eval, a.pairs, out)


if __name__ == "__main__":
    main()
