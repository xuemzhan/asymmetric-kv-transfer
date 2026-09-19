"""V3 Phase 1：因果基板定位 — 逐层 K-only / V-only 替换扫描 Δacc + 有效秩

G0 通过（V 不可传输），现在定位优势住在哪些层/头。
对每个学生层 l：
  K-only: 替换该层 K 为映射教师 K，其余层保持学生 K/V
  V-only: 替换该层 V 为映射教师 V，其余层保持学生 K/V
测量 Δacc（相对 Self 基线）+ 逐层 V 有效秩。
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
    load_model, capture_kv, build_cache, answer_loglik,
    greedy_answer, exact_match, layer_map_proportional, KV,
    MODEL_PATHS, TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
)
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope


def effective_rank(matrix: np.ndarray) -> float:
    """有效秩 = exp(entropy of normalised singular values)。"""
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    s = s / s.sum()
    return float(np.exp(-np.sum(s * np.log(s + 1e-30))))


def run_phase1(seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train.json"))
    test = json.load(open(f"{DATA_DIR}/test.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]

    layer_map = layer_map_proportional()
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)

    # ── 教师 KV ──
    print("[P1] load teacher 8B ...")
    teacher, tok_t = load_model(MODEL_PATHS["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]
    del teacher
    torch.cuda.empty_cache()

    # ── 学生 KV ──
    print("[P1] load student 0.6B ...")
    student, tok_s = load_model(MODEL_PATHS["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]

    # ── 校准堆叠 + 拟合 AffineMapper ──
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
    mk.fit(calib_t_all.k, calib_s_all.k, layer_map, kv_kind="K",
           positions=pos_all, de_rope_fn=de_rope_k)
    mv.fit(calib_t_all.v, calib_s_all.v, layer_map, kv_kind="V",
           positions=pos_all, de_rope_fn=None)
    print(f"  [Affine] fit K+V {time.time()-t0:.1f}s")

    def map_teacher_kv(kv_t, pos):
        km = mk.transform(kv_t.k, layer_map, kv_kind="K",
                          positions=pos, de_rope_fn=de_rope_k)
        vm = mv.transform(kv_t.v, layer_map, kv_kind="V",
                          positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    # ── Self 基线 ──
    print("[P1] computing Self baseline ...")
    self_lls, self_ems = [], []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        c = build_cache(eval_s_kv[i])
        self_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
        self_ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))
    self_mean_ll = float(np.mean(self_lls))
    self_mean_em = float(np.mean(self_ems))
    print(f"  Self: LL={self_mean_ll:.3f}, EM={self_mean_em:.3f}")

    # ── 逐层替换扫描 ──
    # 映射全部教师 KV
    mapped_t_kv = [map_teacher_kv(eval_t_kv[i],
                   np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64))
                   for i in range(len(eval_set))]

    layer_results = []
    for sl in range(STUDENT_LAYERS):
        tl = layer_map[sl][0]  # 对应的教师层
        k_lls, k_ems = [], []
        v_lls, v_ems = [], []
        for i, s in enumerate(eval_set):
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
            mk_l, mv_l = mapped_t_kv[i].k, mapped_t_kv[i].v

            # K-only: 替换学生层 sl 的 K 为映射教师 K（mapper 输出已按学生层排列）
            k_mod = ek.copy()
            k_mod[sl] = mk_l[sl]
            c = build_cache(KV(k=k_mod, v=ev))
            k_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
            k_ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))

            # V-only: 替换学生层 sl 的 V 为映射教师 V
            v_mod = ev.copy()
            v_mod[sl] = mv_l[sl]
            c = build_cache(KV(k=ek, v=v_mod))
            v_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
            v_ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))

        k_mean_ll = float(np.mean(k_lls))
        k_mean_em = float(np.mean(k_ems))
        v_mean_ll = float(np.mean(v_lls))
        v_mean_em = float(np.mean(v_ems))
        layer_results.append({
            "student_layer": sl,
            "teacher_layer": tl,
            "K_only_LL": k_mean_ll,
            "K_only_LL_delta": k_mean_ll - self_mean_ll,
            "K_only_EM": k_mean_em,
            "V_only_LL": v_mean_ll,
            "V_only_LL_delta": v_mean_ll - self_mean_ll,
            "V_only_EM": v_mean_em,
        })
        if (sl + 1) % 7 == 0:
            print(f"  [{sl+1}/{STUDENT_LAYERS}] K_ΔLL={k_mean_ll - self_mean_ll:+.2f}, "
                  f"V_ΔLL={v_mean_ll - self_mean_ll:+.2f}")

    # ── 有效秩：逐层 V 子空间 ──
    print("[P1] computing effective rank per layer ...")
    eff_ranks = []
    for sl in range(STUDENT_LAYERS):
        # 校准集上该层 V 的有效秩（学生 vs 映射教师）
        all_v_s = np.concatenate([eval_s_kv[i].v[sl] for i in range(len(eval_set))], axis=0)
        tl = layer_map[sl][0]
        all_v_t = np.concatenate([mapped_t_kv[i].v[sl] for i in range(len(eval_set))], axis=0)
        # 差异子空间
        diff = all_v_s - all_v_t
        rank_s = effective_rank(all_v_s)
        rank_t = effective_rank(all_v_t)
        rank_d = effective_rank(diff)
        eff_ranks.append({
            "student_layer": sl,
            "teacher_layer": tl,
            "rank_student_V": rank_s,
            "rank_teacher_V_mapped": rank_t,
            "rank_diff": rank_d,
        })

    # ── 汇总 ──
    best_v_layer = max(layer_results, key=lambda r: r["V_only_LL_delta"])
    best_k_layer = max(layer_results, key=lambda r: r["K_only_LL_delta"])

    report = {
        "task": "phase1_causal_substrate",
        "seed": seed,
        "pair": "Qwen3-8B -> Qwen3-0.6B",
        "n_calib": n_calib,
        "n_eval": n_eval,
        "self_LL": self_mean_ll,
        "self_EM": self_mean_em,
        "layer_results": layer_results,
        "effective_ranks": eff_ranks,
        "summary": {
            "best_V_only_layer": best_v_layer["student_layer"],
            "best_V_only_delta_LL": best_v_layer["V_only_LL_delta"],
            "best_V_only_EM": best_v_layer["V_only_EM"],
            "best_K_only_layer": best_k_layer["student_layer"],
            "best_K_only_delta_LL": best_k_layer["K_only_LL_delta"],
            "best_K_only_EM": best_k_layer["K_only_EM"],
            "mean_V_only_delta": float(np.mean([r["V_only_LL_delta"] for r in layer_results])),
            "mean_K_only_delta": float(np.mean([r["K_only_LL_delta"] for r in layer_results])),
        },
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 打印摘要 ──
    print("\n=== Phase 1 Summary ===")
    print(f"Self:  LL={self_mean_ll:.3f}, EM={self_mean_em:.3f}")
    print(f"Best K-only:  layer {best_k_layer['student_layer']}, "
          f"ΔLL={best_k_layer['K_only_LL_delta']:+.3f}, EM={best_k_layer['K_only_EM']:.3f}")
    print(f"Best V-only:  layer {best_v_layer['student_layer']}, "
          f"ΔLL={best_v_layer['V_only_LL_delta']:+.3f}, EM={best_v_layer['V_only_EM']:.3f}")
    print(f"Mean K-only ΔLL: {report['summary']['mean_K_only_delta']:+.3f}")
    print(f"Mean V-only ΔLL: {report['summary']['mean_V_only_delta']:+.3f}")
    print(f"saved: {output}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase1_causal_seed{a.seed}.json"
    run_phase1(a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
