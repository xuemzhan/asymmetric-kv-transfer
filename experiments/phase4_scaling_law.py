"""V3 Phase 4：规模定律 — 多模型对验证 K/V 传输性规律

P4 预测：g 是模型对的"性质"而非"噪声"，跨模型对稳定。

实现：在多个模型对上重复 Phase 0 实验（K-only/V-only/Joint），
验证：
1. K 传输普遍有效（所有模型对 K-only 显著提升）
2. V 传输普遍无效（所有模型对 V-only 不显著或负向）
3. K/V 传输效率比（ΔLL_K / ΔLL_V）跨模型对稳定

模型对：
- 8B → 0.6B（已完成）
- 4B → 1.7B
- 4B → 0.6B
- 8B → 1.7B
- 1.7B → 0.6B（新增：验证小模型对假设）
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/apcs")
from phase0_g0 import (
    load_model, capture_kv, build_cache, answer_loglik,
    greedy_answer, exact_match, layer_map_proportional, KV,
    TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
)
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope

DATA_DIR = "/workspace/v3/data"
REPORT_DIR = "/workspace/v3/reports"

# 多模型对
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


def run_g0_single_pair(pair_name: str, pair: dict,
                       calib: list, eval_set: list, seed: int) -> dict:
    """在单个模型对上运行 G0 实验。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)
    lmap = layer_map_generic(pair["t_layers"], pair["s_layers"])

    # 加载教师
    print(f"  loading teacher ({pair['t_layers']}L) ...")
    teacher, tok_t = load_model(pair["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]
    del teacher
    torch.cuda.empty_cache()

    # 加载学生
    print(f"  loading student ({pair['s_layers']}L) ...")
    student, tok_s = load_model(pair["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]

    # 校准堆叠 + 拟合 AffineMapper
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
    mk.fit(calib_t_all.k, calib_s_all.k, lmap, kv_kind="K",
           positions=pos_all, de_rope_fn=de_rope_k)
    mv.fit(calib_t_all.v, calib_s_all.v, lmap, kv_kind="V",
           positions=pos_all, de_rope_fn=None)

    def map_teacher_kv(kv_t, pos):
        km = mk.transform(kv_t.k, lmap, kv_kind="K",
                          positions=pos, de_rope_fn=de_rope_k)
        vm = mv.transform(kv_t.v, lmap, kv_kind="V",
                          positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    # 四路消融
    rows = []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        pos_i = np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64)
        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"]}

        # Self
        c = build_cache(eval_s_kv[i])
        row["Self"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["Self_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        # 映射教师 KV
        m = map_teacher_kv(eval_t_kv[i], pos_i)

        # K-only
        c = build_cache(KV(k=m.k, v=eval_s_kv[i].v))
        row["K-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["K-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        # V-only
        c = build_cache(KV(k=eval_s_kv[i].k, v=m.v))
        row["V-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["V-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        # Joint
        c = build_cache(KV(k=m.k, v=m.v))
        row["Joint"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["Joint_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        rows.append(row)

    # 汇总
    def agg(key):
        xs = np.array([r[key] for r in rows], dtype=float)
        n = len(xs)
        return {
            "mean": float(xs.mean()),
            "ci95": float(1.96 * xs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        }

    summary = {
        "Self": agg("Self"),
        "K-only": agg("K-only"),
        "V-only": agg("V-only"),
        "Joint": agg("Joint"),
    }

    em = {
        "Self": float(np.mean([r["Self_em"] for r in rows])),
        "K-only": float(np.mean([r["K-only_em"] for r in rows])),
        "V-only": float(np.mean([r["V-only_em"] for r in rows])),
        "Joint": float(np.mean([r["Joint_em"] for r in rows])),
    }

    # 增益计算
    self_ll = summary["Self"]["mean"]
    delta_k = summary["K-only"]["mean"] - self_ll
    delta_v = summary["V-only"]["mean"] - self_ll
    delta_j = summary["Joint"]["mean"] - self_ll

    return {
        "pair": pair_name,
        "summary_ll": summary,
        "exact_match": em,
        "deltas": {
            "K-only": delta_k,
            "V-only": delta_v,
            "Joint": delta_j,
        },
        "ratio": {
            "K_V_ratio": delta_k / max(abs(delta_v), 1e-6) if abs(delta_v) > 1e-6 else float('inf'),
            "efficiency": delta_k / delta_j if abs(delta_j) > 1e-6 else float('inf'),
        },
        "n_eval": len(eval_set),
    }


def run_phase4(seed: int, n_calib: int, n_eval: int, pairs: list[str],
               output: str) -> dict:
    """Phase 4 规模定律实验。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train.json"))
    test = json.load(open(f"{DATA_DIR}/test.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]

    results = {}
    for pair_name in pairs:
        if pair_name not in MODEL_PAIRS:
            print(f"  [SKIP] unknown pair: {pair_name}")
            continue

        print(f"\n[P4] === {pair_name} ===")
        pair = MODEL_PAIRS[pair_name]

        result = run_g0_single_pair(pair_name, pair, calib, eval_set, seed)
        results[pair_name] = result

        print(f"  Self: {result['summary_ll']['Self']['mean']:.3f}")
        print(f"  K-only: {result['summary_ll']['K-only']['mean']:.3f} (Δ={result['deltas']['K-only']:+.3f})")
        print(f"  V-only: {result['summary_ll']['V-only']['mean']:.3f} (Δ={result['deltas']['V-only']:+.3f})")
        print(f"  Joint: {result['summary_ll']['Joint']['mean']:.3f} (Δ={result['deltas']['Joint']:+.3f})")
        print(f"  K/V ratio: {result['ratio']['K_V_ratio']:.2f}")

    # 跨模型对分析
    k_deltas = [r["deltas"]["K-only"] for r in results.values()]
    v_deltas = [r["deltas"]["V-only"] for r in results.values()]
    ratios = [r["ratio"]["K_V_ratio"] for r in results.values()]

    cross_pair_analysis = {
        "mean_K_delta": float(np.mean(k_deltas)),
        "std_K_delta": float(np.std(k_deltas)),
        "mean_V_delta": float(np.mean(v_deltas)),
        "std_V_delta": float(np.std(v_deltas)),
        "mean_K_V_ratio": float(np.mean(ratios)),
        "std_K_V_ratio": float(np.std(ratios)),
        "K_always_positive": all(d > 0 for d in k_deltas),
        "V_always_negative": all(d < 0 for d in v_deltas),
        "pattern_consistent": all(d > 0 for d in k_deltas) and all(d < 0 for d in v_deltas),
    }

    report = {
        "task": "phase4_scaling_law",
        "seed": seed,
        "n_calib": n_calib,
        "n_eval": n_eval,
        "pairs": pairs,
        "results": results,
        "cross_pair_analysis": cross_pair_analysis,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n=== Phase 4 Summary ===")
    print(f"Pairs tested: {len(results)}")
    print(f"Mean K-only ΔLL: {cross_pair_analysis['mean_K_delta']:+.3f} ± {cross_pair_analysis['std_K_delta']:.3f}")
    print(f"Mean V-only ΔLL: {cross_pair_analysis['mean_V_delta']:+.3f} ± {cross_pair_analysis['std_V_delta']:.3f}")
    print(f"Mean K/V ratio: {cross_pair_analysis['mean_K_V_ratio']:.2f} ± {cross_pair_analysis['std_K_V_ratio']:.2f}")
    print(f"K always positive: {cross_pair_analysis['K_always_positive']}")
    print(f"V always negative: {cross_pair_analysis['V_always_negative']}")
    print(f"Pattern consistent: {cross_pair_analysis['pattern_consistent']}")
    print(f"saved: {output}")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--pairs", nargs="+", default=["8B_0.6B", "4B_1.7B", "4B_0.6B", "8B_1.7B", "1.7B_0.6B"])
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase4_scaling_law_seed{a.seed}.json"
    run_phase4(a.seed, a.n_calib, a.n_eval, a.pairs, out)


if __name__ == "__main__":
    main()
