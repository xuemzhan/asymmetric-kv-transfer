"""V3 Phase 1 v2：因果基板定位 — 逐层 K/V 替换扫描 + 统计检验 + selective V + K prefix-cumulative

v2 变更（相对 phase1_causal.py）：
1. 数据：train_v2_seed{seed}.json / test_v2_seed{seed}.json（n_calib=70, n_eval=56）
2. 逐层扫描：每层 K-only / V-only ΔLL 附 bootstrap CI95 + Wilcoxon vs Self
3. Selective V injection（预注册固定层）: {L8}, {L12}, {L8,L12}, {ALL} — 只替换 V，测 LL+EM
4. K prefix-cumulative: 前缀 [0..l] 累计替换 K（l ∈ {4,8,12,16,20,24,27}），测 LL+EM
5. 3 seeds 独立运行，报告含逐层统计
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/workspace/apcs")
sys.path.insert(0, "/workspace/v3")
from phase0_g0 import (
    load_model, capture_kv, build_cache, answer_loglik,
    greedy_answer, exact_match, layer_map_proportional, KV,
    MODEL_PATHS, TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
)
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope
from stats_utils import bootstrap_ci95, paired_wilcoxon_test

DATA_DIR = "/workspace/v3/data"
REPORT_DIR = "/workspace/v3/reports"

# 预注册的 selective V injection 配置（基于 paper-1 发现 L8/L12 为 V hotspot；固定不事后调整）
SELECTIVE_V_LAYERS = [8, 12]
PREFIX_LS = [4, 8, 12, 16, 20, 24, 27]


def effective_rank(matrix: np.ndarray) -> float:
    """有效秩 = exp(entropy of normalised singular values)。"""
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    s = s / s.sum()
    return float(np.exp(-np.sum(s * np.log(s + 1e-30))))


def run_phase1_v2(seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train_v2_seed{seed}.json"))
    test = json.load(open(f"{DATA_DIR}/test_v2_seed{seed}.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]

    layer_map = layer_map_proportional()
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)

    # ── 教师 KV ──
    print(f"[P1v2 seed{seed}] load teacher 8B ...")
    teacher, tok_t = load_model(MODEL_PATHS["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]
    del teacher
    torch.cuda.empty_cache()

    # ── 学生 KV ──
    print("[P1v2] load student 0.6B ...")
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

    # ── Self 基线（含 per-sample 用于统计） ──
    print("[P1v2] computing Self baseline ...")
    self_lls, self_ems = [], []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        c = build_cache(eval_s_kv[i])
        self_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
        self_ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))
    self_mean_ll = float(np.mean(self_lls))
    self_mean_em = float(np.mean(self_ems))
    print(f"  Self: LL={self_mean_ll:.3f}, EM={self_mean_em:.3f}")

    # 映射全部教师 KV
    mapped_t_kv = [map_teacher_kv(eval_t_kv[i],
                   np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64))
                   for i in range(len(eval_set))]

    # ── 逐层替换扫描（loglik only，快） ──
    print("[P1v2] per-layer scan (loglik) ...")
    layer_results = []
    for sl in range(STUDENT_LAYERS):
        tl = layer_map[sl][0]
        k_lls, v_lls = [], []
        for i, s in enumerate(eval_set):
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
            mk_l, mv_l = mapped_t_kv[i].k, mapped_t_kv[i].v

            k_mod = ek.copy(); k_mod[sl] = mk_l[sl]
            c = build_cache(KV(k=k_mod, v=ev))
            k_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))

            v_mod = ev.copy(); v_mod[sl] = mv_l[sl]
            c = build_cache(KV(k=ek, v=v_mod))
            v_lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))

        k_lls = np.array(k_lls); v_lls = np.array(v_lls); sll = np.array(self_lls)
        k_mean, k_lo, k_hi = bootstrap_ci95(k_lls - sll, seed=seed)
        v_mean, v_lo, v_hi = bootstrap_ci95(v_lls - sll, seed=seed)
        _, k_p, k_d = paired_wilcoxon_test(k_lls.tolist(), sll.tolist())
        _, v_p, v_d = paired_wilcoxon_test(v_lls.tolist(), sll.tolist())
        layer_results.append({
            "student_layer": sl, "teacher_layer": tl,
            "K_only_LL_delta": float(k_mean),
            "K_delta_ci95": [float(k_lo), float(k_hi)],
            "K_wilcoxon_p": float(k_p), "K_cohens_d": float(k_d),
            "V_only_LL_delta": float(v_mean),
            "V_delta_ci95": [float(v_lo), float(v_hi)],
            "V_wilcoxon_p": float(v_p), "V_cohens_d": float(v_d),
        })
        if (sl + 1) % 7 == 0:
            print(f"  [{sl+1}/{STUDENT_LAYERS}] K_ΔLL={k_mean:+.2f} V_ΔLL={v_mean:+.2f}")

    # ── Selective V injection（LL + EM） ──
    print("[P1v2] selective V injection ...")
    sel_cfgs = {
        f"V_{l}": [l] for l in SELECTIVE_V_LAYERS
    }
    sel_cfgs["V_L8_L12"] = list(SELECTIVE_V_LAYERS)
    sel_cfgs["V_ALL"] = list(range(STUDENT_LAYERS))
    sel_results = {}
    for cname, layers in sel_cfgs.items():
        lls, ems = [], []
        for i, s in enumerate(eval_set):
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
            v_mod = ev.copy()
            for sl in layers:
                v_mod[sl] = mapped_t_kv[i].v[sl]
            c = build_cache(KV(k=ek, v=v_mod))
            lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
            ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))
        lls = np.array(lls)
        mean, lo, hi = bootstrap_ci95(lls - np.array(self_lls), seed=seed)
        _, p, d = paired_wilcoxon_test(lls.tolist(), self_lls)
        sel_results[cname] = {
            "layers": layers,
            "LL_delta_vs_Self": float(mean),
            "LL_delta_ci95": [float(lo), float(hi)],
            "LL_mean": float(lls.mean()),
            "EM": float(np.mean(ems)),
            "wilcoxon_p_vs_Self": float(p),
            "cohens_d": float(d),
        }
        print(f"  {cname}: ΔLL={mean:+.3f} EM={np.mean(ems):.3f} p={p:.3e}")

    # ── K prefix-cumulative（LL + EM） ──
    print("[P1v2] K prefix-cumulative ...")
    prefix_results = {}
    for pl in PREFIX_LS:
        layers = list(range(pl + 1))
        lls, ems = [], []
        for i, s in enumerate(eval_set):
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
            k_mod = ek.copy()
            for sl in layers:
                k_mod[sl] = mapped_t_kv[i].k[sl]
            c = build_cache(KV(k=k_mod, v=ev))
            lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
            ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))
        lls = np.array(lls)
        mean, lo, hi = bootstrap_ci95(lls - np.array(self_lls), seed=seed)
        _, p, d = paired_wilcoxon_test(lls.tolist(), self_lls)
        prefix_results[str(pl)] = {
            "layers_in_prefix": pl + 1,
            "LL_delta_vs_Self": float(mean),
            "LL_delta_ci95": [float(lo), float(hi)],
            "LL_mean": float(lls.mean()),
            "EM": float(np.mean(ems)),
            "wilcoxon_p_vs_Self": float(p),
            "cohens_d": float(d),
        }
        print(f"  prefix[0..{pl}]: ΔLL={mean:+.3f} EM={np.mean(ems):.3f} p={p:.3e}")

    # ── 有效秩：逐层 V 子空间（差异秩 = 不可传输内容量） ──
    print("[P1v2] computing effective rank per layer ...")
    eff_ranks = []
    for sl in range(STUDENT_LAYERS):
        all_v_s = np.concatenate([eval_s_kv[i].v[sl] for i in range(len(eval_set))], axis=0)
        tl = layer_map[sl][0]
        all_v_t = np.concatenate([mapped_t_kv[i].v[sl] for i in range(len(eval_set))], axis=0)
        diff = all_v_s - all_v_t
        eff_ranks.append({
            "student_layer": sl, "teacher_layer": tl,
            "rank_student_V": float(effective_rank(all_v_s)),
            "rank_teacher_V_mapped": float(effective_rank(all_v_t)),
            "rank_diff": float(effective_rank(diff)),
        })

    # ── 汇总 ──
    best_v = max(layer_results, key=lambda r: r["V_only_LL_delta"])
    best_k = max(layer_results, key=lambda r: r["K_only_LL_delta"])
    report = {
        "task": "phase1_causal_substrate_v2",
        "seed": seed,
        "pair": "Qwen3-8B -> Qwen3-0.6B",
        "n_calib": n_calib, "n_eval": n_eval,
        "self_LL": self_mean_ll, "self_EM": self_mean_em,
        "layer_results": layer_results,
        "effective_ranks": eff_ranks,
        "selective_V_injection": sel_results,
        "K_prefix_cumulative": prefix_results,
        "summary": {
            "best_V_only_layer": best_v["student_layer"],
            "best_V_only_delta_LL": best_v["V_only_LL_delta"],
            "best_V_only_p": best_v["V_wilcoxon_p"],
            "best_K_only_layer": best_k["student_layer"],
            "best_K_only_delta_LL": best_k["K_only_LL_delta"],
            "best_K_only_p": best_k["K_wilcoxon_p"],
            "best_selective_V": max(sel_results, key=lambda k: sel_results[k]["LL_delta_vs_Self"]),
            "best_prefix": max(prefix_results, key=lambda k: prefix_results[k]["LL_delta_vs_Self"]),
        },
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Phase 1 v2 Summary ===")
    print(f"Self: LL={self_mean_ll:.3f} EM={self_mean_em:.3f}")
    print(f"Best K-only: L{best_k['student_layer']} ΔLL={best_k['K_only_LL_delta']:+.3f} p={best_k['K_wilcoxon_p']:.2e}")
    print(f"Best V-only: L{best_v['student_layer']} ΔLL={best_v['V_only_LL_delta']:+.3f} p={best_v['V_wilcoxon_p']:.2e}")
    for cname, v in sel_results.items():
        print(f"  {cname}: ΔLL={v['LL_delta_vs_Self']:+.3f} EM={v['EM']:.3f}")
    print(f"saved: {output}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase1_causal_v2_seed{a.seed}.json"
    run_phase1_v2(a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
