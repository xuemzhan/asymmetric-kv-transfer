"""V3 Phase 2：标量 g — CCA 典型相关系数跨模型对扫描

P3 预测：存在标量 g：K 侧 ≈ 1、V 侧 ≈ 0，跨模型对/层稳定。
实现：对每个 (层, K/V) 配对，计算 CCA 第一典型相关系数 ρ₁ 作为 g 的代理。

模型对：
  - 8B → 0.6B（主对）
  - 4B → 1.7B（扩展对1）
  - 4B → 0.6B（扩展对2）
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")
sys.path.insert(0, APCS_DIR)
from phase0_g0 import (
    load_model, capture_kv, KV,
    MODEL_PATHS, TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
    layer_map_proportional,
)
from apcs.rope.runner import _rope_pairs, de_rope

# 扩展模型对
MODEL_PAIRS = {
    "8B_0.6B": {
        "teacher": os.path.join(MODELS_DIR, "Qwen--Qwen3-8B", "snapshots", "master"),
        "student": os.path.join(MODELS_DIR, "Qwen--Qwen3-0.6B", "snapshots", "master"),
        "t_layers": 36, "s_layers": 28,
    },
    "4B_1.7B": {
        "teacher": os.path.join(MODELS_DIR, "Qwen--Qwen3-4B", "snapshots", "master"),
        "student": os.path.join(MODELS_DIR, "Qwen--Qwen3-1.7B", "snapshots", "master"),
        "t_layers": 36, "s_layers": 28,
    },
    "4B_0.6B": {
        "teacher": os.path.join(MODELS_DIR, "Qwen--Qwen3-4B", "snapshots", "master"),
        "student": os.path.join(MODELS_DIR, "Qwen--Qwen3-0.6B", "snapshots", "master"),
        "t_layers": 36, "s_layers": 28,
    },
}


def layer_map_generic(t_layers: int, s_layers: int) -> list[list[int]]:
    return [[min(t_layers - 1, round(s * t_layers / s_layers))]
            for s in range(s_layers)]


def cca_first_corr(A: np.ndarray, B: np.ndarray, lam: float = 1e-3,
                    max_samples: int = 500) -> float:
    """CCA 第一典型相关系数 ρ₁。A, B: (N, D) 矩阵。
    当 N >> D 时 CCA 平凡得 ρ≈1，故先 subsample 到 max_samples。
    """
    n = A.shape[0]
    if n < 10:
        return 0.0
    # subsample 避免 N>>D 导致 CCA 平凡
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        A = A[idx]
        B = B[idx]
        n = max_samples
    # 中心化
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    # 协方差
    Cxx = (A.T @ A) / n + lam * np.eye(A.shape[1])
    Cyy = (B.T @ B) / n + lam * np.eye(B.shape[1])
    Cxy = (A.T @ B) / n
    Cyx = Cxy.T
    # generalized eigenproblem
    try:
        Cxx_inv = np.linalg.inv(Cxx)
        Cyy_inv = np.linalg.inv(Cyy)
        M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
        eigvals = np.linalg.eigvalsh(M)
        rho1 = float(np.clip(np.sqrt(max(eigvals.max(), 0)), 0, 1))
    except np.linalg.LinAlgError:
        rho1 = 0.0
    return rho1


def run_phase2(seed: int, n_calib: int, pairs: list[str], output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train.json"))
    calib = train[:n_calib]
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)

    results = {}
    for pair_name in pairs:
        pair = MODEL_PAIRS[pair_name]
        print(f"\n[P2] === {pair_name} ===")

        # 加载教师
        print(f"  loading teacher ({pair['t_layers']}L) ...")
        teacher, tok_t = load_model(pair["teacher"])
        calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
        del teacher
        torch.cuda.empty_cache()

        # 加载学生
        print(f"  loading student ({pair['s_layers']}L) ...")
        student, tok_s = load_model(pair["student"])
        calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
        del student
        torch.cuda.empty_cache()

        # 堆叠
        def stack_kv(kvs):
            return KV(
                k=np.concatenate([kv.k for kv in kvs], axis=1),
                v=np.concatenate([kv.v for kv in kvs], axis=1),
            )
        calib_t_all = stack_kv(calib_t_kv)
        calib_s_all = stack_kv(calib_s_kv)
        pos_all = np.arange(calib_t_all.k.shape[1], dtype=np.float64)

        lmap = layer_map_generic(pair["t_layers"], pair["s_layers"])

        # de-RoPE 教师 K 和学生 K（整层一次性）
        t_k_deroped = np.zeros_like(calib_t_all.k)
        s_k_deroped = np.zeros_like(calib_s_all.k)
        for sl, tl_list in enumerate(lmap):
            tl = tl_list[0]
            S = calib_t_all.k.shape[1]
            pos = np.arange(S, dtype=np.float64)
            t_k_deroped[sl] = de_rope_k(calib_t_all.k[sl], pos)
            s_k_deroped[sl] = de_rope_k(calib_s_all.k[sl], pos)

        # 逐层 CCA
        layer_g = []
        for sl in range(pair["s_layers"]):
            tl = lmap[sl][0]
            # K: 都已 de-RoPE
            N = calib_t_all.k.shape[1]
            D = HEAD_DIM
            H = KV_HEADS
            t_k = t_k_deroped[sl].reshape(N, H * D)
            s_k = s_k_deroped[sl].reshape(N, H * D)
            t_v = calib_t_all.v[sl].reshape(N, H * D)
            s_v = calib_s_all.v[sl].reshape(N, H * D)

            g_k = cca_first_corr(t_k, s_k)
            g_v = cca_first_corr(t_v, s_v)
            layer_g.append({
                "student_layer": sl,
                "teacher_layer": tl,
                "g_K": g_k,
                "g_V": g_v,
                "g_ratio": g_k / max(g_v, 1e-6),
            })

        # 汇总
        g_k_vals = [lg["g_K"] for lg in layer_g]
        g_v_vals = [lg["g_V"] for lg in layer_g]
        pair_summary = {
            "layers": layer_g,
            "mean_g_K": float(np.mean(g_k_vals)),
            "mean_g_V": float(np.mean(g_v_vals)),
            "std_g_K": float(np.std(g_k_vals)),
            "std_g_V": float(np.std(g_v_vals)),
            "mean_g_ratio": float(np.mean([lg["g_ratio"] for lg in layer_g])),
        }
        results[pair_name] = pair_summary
        print(f"  mean g_K={pair_summary['mean_g_K']:.4f} ± {pair_summary['std_g_K']:.4f}")
        print(f"  mean g_V={pair_summary['mean_g_V']:.4f} ± {pair_summary['std_g_V']:.4f}")
        print(f"  mean g_ratio={pair_summary['mean_g_ratio']:.2f}")

    # P3 判定
    p3_verdict = {}
    for pn, ps in results.items():
        k_high = ps["mean_g_K"] > 0.5  # K 侧高
        v_low = ps["mean_g_V"] < ps["mean_g_K"] * 0.5  # V 侧明显低于 K
        consistent = ps["std_g_K"] < 0.3 and ps["std_g_V"] < 0.3  # 跨层稳定
        p3_verdict[pn] = {
            "K_high": k_high,
            "V_low_relative_to_K": v_low,
            "cross_layer_stable": consistent,
            "P3_supported": k_high and v_low and consistent,
        }

    report = {
        "task": "phase2_g_scalar",
        "seed": seed,
        "n_calib": n_calib,
        "pairs": pairs,
        "results": results,
        "p3_verdict": p3_verdict,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[P2] saved: {output}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=42)
    ap.add_argument("--pairs", nargs="+", default=["8B_0.6B", "4B_1.7B", "4B_0.6B"])
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase2_g_scalar_seed{a.seed}.json"
    run_phase2(a.seed, a.n_calib, a.pairs, out)


if __name__ == "__main__":
    main()
