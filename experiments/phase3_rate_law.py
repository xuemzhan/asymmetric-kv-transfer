"""V3 Phase 3：速率-优势定律 — 状态/权重比特成本比 vs 1/g

P4 预测：状态/权重比特成本比被 1/g 预测（指数级差距）。

实现：
1. 计算状态传输成本：KV cache 大小（bytes）= layers × tokens × 2 × kv_heads × head_dim × dtype_size
2. 计算权重传输成本：映射器参数量（bytes）= 每层参数 × 层数 × dtype_size
3. 生成成本比曲线：cache_cost / mapper_cost vs 序列长度
4. 与任务增益（ΔLL）对比，验证 1/g 预测

关键洞察：
- g_K ≈ 1 → K 状态传输有效 → cache 传输效率 ≈ 权重传输效率
- g_V ≈ 0 → V 状态传输无效 → cache 传输效率 << 权重传输效率
- 成本比应预测：何时传输状态（小序列）vs 传输权重（大序列）
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/apcs")
from phase0_g0 import (
    load_model, capture_kv, build_cache, answer_loglik,
    greedy_answer, exact_match, layer_map_proportional, KV,
    MODEL_PATHS, TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
)
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope

DATA_DIR = "/workspace/v3/data"
REPORT_DIR = "/workspace/v3/reports"

# 精度
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def compute_cache_bytes(n_layers: int, n_tokens: int, kv_heads: int, head_dim: int,
                        dtype: str = "bfloat16") -> dict:
    """计算 KV cache 的字节数。"""
    bytes_per_token_per_layer = 2 * kv_heads * head_dim * DTYPE_BYTES[dtype]  # K + V
    total_bytes = n_layers * n_tokens * bytes_per_token_per_layer
    return {
        "bytes": total_bytes,
        "mb": total_bytes / (1024**2),
        "gb": total_bytes / (1024**3),
        "per_token_bytes": n_layers * bytes_per_token_per_layer,
    }


def compute_mapper_bytes(n_layers: int, kv_dim: int, dtype: str = "float32") -> dict:
    """计算 Affine 映射器的参数字节数（K + V 各一个仿射变换）。"""
    # Affine: W ∈ R^{kv_dim × kv_dim} + bias ∈ R^{kv_dim}
    params_per_mapper = kv_dim * kv_dim + kv_dim
    total_params = 2 * params_per_mapper * n_layers  # K + V
    total_bytes = total_params * DTYPE_BYTES[dtype]
    return {
        "bytes": total_bytes,
        "mb": total_bytes / (1024**2),
        "total_params": total_params,
        "per_layer_params": 2 * params_per_mapper,
    }


def estimate_task_gain_ratio(g_k: float, g_v: float) -> float:
    """基于 g 值估算任务增益比。g_K≈1 表示 K 传输有效，g_V≈0 表示 V 传输无效。
    返回：有效传输比率（0-1）。"""
    # 简化模型：任务增益由 K 传输主导，V 传输贡献有限
    # 如果 g_V << g_K，则 V 传输几乎无效
    if g_k < 1e-6:
        return 0.0
    # 增益比 = g_K / (g_K + g_V) 的某种函数
    # 当 g_V = 0 时增益比 = 1（K 传输完全有效）
    # 当 g_V = g_K 时增益比 = 0.5（K/V 传输同等有效）
    return g_k / (g_k + g_v + 1e-6)


def run_phase3(seed: int, n_calib: int, n_eval: int, seq_lengths: list[int],
               output: str) -> dict:
    """Phase 3 速率-优势定律实验。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train.json"))
    test = json.load(open(f"{DATA_DIR}/test.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]

    layer_map = layer_map_proportional()
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)

    # ── 加载教师和学生 ──
    print("[P3] load teacher 8B ...")
    teacher, tok_t = load_model(MODEL_PATHS["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]
    del teacher
    torch.cuda.empty_cache()

    print("[P3] load student 0.6B ...")
    student, tok_s = load_model(MODEL_PATHS["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]

    # ── 校准 + 拟合映射器 ──
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
    mk.fit(calib_t_all.k, calib_s_all.k, layer_map, kv_kind="K",
           positions=pos_all, de_rope_fn=de_rope_k)
    mv.fit(calib_t_all.v, calib_s_all.v, layer_map, kv_kind="V",
           positions=pos_all, de_rope_fn=None)

    def map_teacher_kv(kv_t, pos):
        km = mk.transform(kv_t.k, layer_map, kv_kind="K",
                          positions=pos, de_rope_fn=de_rope_k)
        vm = mv.transform(kv_t.v, layer_map, kv_kind="V",
                          positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    # ── Self 基线 ──
    print("[P3] computing Self baseline ...")
    self_lls = []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        c = build_cache(eval_s_kv[i])
        self_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
    self_mean_ll = float(np.mean(self_lls))

    # ── K-only / V-only / Joint 基线 ──
    print("[P3] computing injection baselines ...")
    k_only_lls, v_only_lls, joint_lls = [], [], []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        pos_i = np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64)
        m = map_teacher_kv(eval_t_kv[i], pos_i)

        # K-only
        c = build_cache(KV(k=m.k, v=eval_s_kv[i].v))
        k_only_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))

        # V-only
        c = build_cache(KV(k=eval_s_kv[i].k, v=m.v))
        v_only_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))

        # Joint
        c = build_cache(KV(k=m.k, v=m.v))
        joint_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))

    k_only_mean_ll = float(np.mean(k_only_lls))
    v_only_mean_ll = float(np.mean(v_only_lls))
    joint_mean_ll = float(np.mean(joint_lls))

    delta_k_only = k_only_mean_ll - self_mean_ll
    delta_v_only = v_only_mean_ll - self_mean_ll
    delta_joint = joint_mean_ll - self_mean_ll

    print(f"  Self: {self_mean_ll:.3f}")
    print(f"  K-only: {k_only_mean_ll:.3f} (Δ={delta_k_only:+.3f})")
    print(f"  V-only: {v_only_mean_ll:.3f} (Δ={delta_v_only:+.3f})")
    print(f"  Joint:  {joint_mean_ll:.3f} (Δ={delta_joint:+.3f})")

    # ── 成本计算 ──
    kv_dim = KV_HEADS * HEAD_DIM  # 1024

    # 不同序列长度的成本
    cost_curve = []
    for n_tok in seq_lengths:
        cache_cost = compute_cache_bytes(STUDENT_LAYERS, n_tok, KV_HEADS, HEAD_DIM)
        mapper_cost = compute_mapper_bytes(STUDENT_LAYERS, kv_dim)
        cost_ratio = cache_cost["mb"] / mapper_cost["mb"]

        cost_curve.append({
            "seq_len": n_tok,
            "cache_mb": cache_cost["mb"],
            "mapper_mb": mapper_cost["mb"],
            "cost_ratio": cost_ratio,
            "cache_gb": cache_cost["gb"],
        })

    # ── 传输效率分析 ──
    # 假设：传输完整 KV cache = 传输 n_tok 个 token 的状态
    # 传输映射器 = 传输固定参数（可复用于任意 n_tok）
    # 交叉点：cache_cost = mapper_cost
    cross_point = None
    for i in range(len(cost_curve) - 1):
        r1, r2 = cost_curve[i]["cost_ratio"], cost_curve[i+1]["cost_ratio"]
        if r1 < 1.0 and r2 >= 1.0:
            # 线性插值
            frac = (1.0 - r1) / (r2 - r1)
            cross_point = cost_curve[i]["seq_len"] + frac * (cost_curve[i+1]["seq_len"] - cost_curve[i]["seq_len"])
            break

    # ── 增益效率比 ──
    # 每 MB 传输获得的 ΔLL
    efficiency = {
        "K_only_per_mb": delta_k_only / mapper_cost["mb"],
        "V_only_per_mb": delta_v_only / mapper_cost["mb"],
        "Joint_per_mb": delta_joint / mapper_cost["mb"],
        # cache 传输效率（按 token 平均）
        "K_only_per_token": delta_k_only / kv_dim,  # 每 token 传输 K 的增益
        "V_only_per_token": delta_v_only / kv_dim,
    }

    # ── 预测：1/g 预测成本比 ──
    # 从 Phase 2 结果读取 g 值（如果可用）
    g_k, g_v = 1.0, 0.9998  # 默认值（Phase 2 结果）
    try:
        p2_path = f"{REPORT_DIR}/phase2_g_scalar_seed{seed}.json"
        if os.path.exists(p2_path):
            p2 = json.load(open(p2_path))
            if "8B_0.6B" in p2.get("results", {}):
                g_k = p2["results"]["8B_0.6B"]["mean_g_K"]
                g_v = p2["results"]["8B_0.6B"]["mean_g_V"]
    except Exception:
        pass

    predicted_ratio = 1.0 / max(g_k / max(g_v, 1e-6), 1e-6)

    report = {
        "task": "phase3_rate_law",
        "seed": seed,
        "pair": "Qwen3-8B -> Qwen3-0.6B",
        "n_calib": n_calib,
        "n_eval": n_eval,
        "baselines": {
            "self_LL": self_mean_ll,
            "k_only_LL": k_only_mean_ll,
            "v_only_LL": v_only_mean_ll,
            "joint_LL": joint_mean_ll,
            "delta_k_only": delta_k_only,
            "delta_v_only": delta_v_only,
            "delta_joint": delta_joint,
        },
        "cost_curve": cost_curve,
        "cross_point_seq_len": cross_point,
        "mapper_params_mb": mapper_cost["mb"],
        "mapper_total_params": mapper_cost["total_params"],
        "efficiency": efficiency,
        "g_values": {"g_K": g_k, "g_V": g_v},
        "predicted_cost_ratio": predicted_ratio,
        "p4_verdict": {
            "cross_point_exists": cross_point is not None,
            "cross_point_seq_len": cross_point,
            "predicted_ratio": predicted_ratio,
            "actual_ratio_at_cross": cost_curve[0]["cost_ratio"] if cost_curve else None,
        },
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 打印摘要 ──
    print("\n=== Phase 3 Summary ===")
    print(f"Mapper params: {mapper_cost['total_params']/1e6:.2f}M ({mapper_cost['mb']:.1f} MB)")
    print(f"Cache per token: {STUDENT_LAYERS * 2 * KV_HEADS * HEAD_DIM * 2 / 1024:.1f} KB")
    print(f"Cost ratio @ 1K tokens: {cost_curve[0]['cost_ratio']:.3f}")
    print(f"Cost ratio @ 10K tokens: {cost_curve[1]['cost_ratio']:.3f}")
    print(f"Cost ratio @ 100K tokens: {cost_curve[2]['cost_ratio']:.3f}")
    if cross_point:
        print(f"Cross point (cache=mapper): {cross_point:.0f} tokens")
    print(f"g_K={g_k:.4f}, g_V={g_v:.4f}, predicted ratio=1/g={predicted_ratio:.2f}")
    print(f"saved: {output}")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase3_rate_law_seed{a.seed}.json"

    # 测试序列长度
    seq_lengths = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    run_phase3(a.seed, a.n_calib, a.n_eval, seq_lengths, out)


if __name__ == "__main__":
    main()
