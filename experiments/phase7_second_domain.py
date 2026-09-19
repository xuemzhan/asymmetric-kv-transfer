"""V3 W7：第二域验证 — SQuAD 上 4-way ablation（跨域校准）

设计（预注册）：
- 校准：OOD 域 train_v2_seed{seed}（70 样本）拟合 AffineMapper K/V
- 评估：SQuAD validation 30 样本（squad_test_seed0.json），4-way ablation
  （Self / K-only / V-only / Joint），LL + EM + bootstrap CI95 + Wilcoxon
- novelty probe：nq_open 30 问题（无 context），测学生/教师先验
- 跨域校准 → 若 K-only 显著优于 Self 而 V-only 不显著，则支持结论普适性

用法：python3 experiments/phase7_second_domain.py --seed 0
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
sys.path.insert(0, _ROOT)
from phase0_g0 import (
    load_model, capture_kv, build_cache, answer_loglik,
    greedy_answer, exact_match, layer_map_proportional, KV,
    MODEL_PATHS, TEACHER_LAYERS, STUDENT_LAYERS, KV_HEADS, HEAD_DIM, ROPE_THETA,
)
from apcs.mapper.math import AffineMapper
from apcs.rope.runner import _rope_pairs, de_rope
from stats_utils import bootstrap_ci95, paired_wilcoxon_test


def run_second_domain(seed: int, n_calib: int, n_eval: int, output: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train_v2_seed{seed}.json"))
    eval_set = json.load(open(f"{DATA_DIR}/squad_test_seed0.json"))[:n_eval]
    probe_set = json.load(open(f"{DATA_DIR}/nq_open_probe_seed0.json"))
    calib = train[:n_calib]
    print(f"[W7 seed{seed}] calib={len(calib)} (OOD), eval={len(eval_set)} (SQuAD), probe={len(probe_set)}")

    layer_map = layer_map_proportional()
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)

    # ── 教师 KV ──
    print("[W7] load teacher 8B ...")
    teacher, tok_t = load_model(MODEL_PATHS["teacher"])
    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]

    # 教师 novelty probe（无 context）
    probe_t = {"n": len(probe_set), "em": 0.0, "mean_ll": 0.0}
    p_lls, p_ems = [], []
    for s in probe_set:
        q = "Question: " + s["q"] + "\nAnswer:"
        p_ems.append(float(exact_match(greedy_answer(teacher, tok_t, None, q), s["answer"])))
        p_lls.append(answer_loglik(teacher, tok_t, None, q, s["answer"]))
    probe_t = {"n": len(probe_set), "em": float(np.mean(p_ems)), "mean_ll": float(np.mean(p_lls))}
    print(f"  teacher probe: em={probe_t['em']:.3f}")
    del teacher
    torch.cuda.empty_cache()

    # ── 学生 KV ──
    print("[W7] load student 0.6B ...")
    student, tok_s = load_model(MODEL_PATHS["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]

    # 学生 novelty probe
    p_lls, p_ems = [], []
    for s in probe_set:
        q = "Question: " + s["q"] + "\nAnswer:"
        p_ems.append(float(exact_match(greedy_answer(student, tok_s, None, q), s["answer"])))
        p_lls.append(answer_loglik(student, tok_s, None, q, s["answer"]))
    probe_s = {"n": len(probe_set), "em": float(np.mean(p_ems)), "mean_ll": float(np.mean(p_lls))}
    print(f"  student probe: em={probe_s['em']:.3f}")

    # ── 校准 + 拟合 ──
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

    mapped_t_kv = [map_teacher_kv(eval_t_kv[i],
                   np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64))
                   for i in range(len(eval_set))]

    # ── 4-way ablation ──
    arms = {"Self": None, "K-only": "K", "V-only": "V", "Joint": "KV"}
    results = {}
    for arm_name, mode in arms.items():
        lls, ems = [], []
        for i, s in enumerate(eval_set):
            q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
            ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
            mk_i, mv_i = mapped_t_kv[i].k, mapped_t_kv[i].v
            if mode == "K":
                c = build_cache(KV(k=mk_i, v=ev))
            elif mode == "V":
                c = build_cache(KV(k=ek, v=mv_i))
            elif mode == "KV":
                c = build_cache(KV(k=mk_i, v=mv_i))
            else:
                c = build_cache(KV(k=ek, v=ev))
            lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
            ems.append(exact_match(greedy_answer(student, tok_s, c, q), s["answer"]))
        lls = np.array(lls)
        mean, lo, hi = bootstrap_ci95(lls, seed=seed)
        results[arm_name] = {
            "LL_mean": float(mean),
            "LL_ci95": [float(lo), float(hi)],
            "EM": float(np.mean(ems)),
            "n": len(lls),
        }
        print(f"  {arm_name}: LL={mean:.3f} CI95=[{lo:.3f},{hi:.3f}] EM={np.mean(ems):.3f}")

    # ── 统计检验：K-only/V-only/Joint vs Self ──
    self_lls = np.array(_collect_lls(eval_set, eval_s_kv, mapped_t_kv, student, tok_s, None))
    stats = {}
    for arm_name in ["K-only", "V-only", "Joint"]:
        arm_lls = _collect_lls(eval_set, eval_s_kv, mapped_t_kv, student, tok_s, arms[arm_name])
        arm_lls = np.array(arm_lls)
        delta_mean, dlo, dhi = bootstrap_ci95(arm_lls - self_lls, seed=seed)
        _, p, d = paired_wilcoxon_test(arm_lls.tolist(), self_lls.tolist())
        stats[arm_name] = {
            "delta_mean": float(delta_mean),
            "delta_ci95": [float(dlo), float(dhi)],
            "wilcoxon_p": float(p),
            "cohens_d": float(d),
        }
        print(f"  {arm_name} vs Self: ΔLL={delta_mean:+.3f} p={p:.3e} d={d:+.2f}")

    report = {
        "task": "phase7_second_domain_squad",
        "seed": seed,
        "pair": "Qwen3-8B -> Qwen3-0.6B",
        "calib_domain": "OOD (train_v2)", "eval_domain": "SQuAD",
        "n_calib": n_calib, "n_eval": n_eval,
        "results_4way": results,
        "stats_vs_self": stats,
        "novelty_probe": {"teacher": probe_t, "student": probe_s},
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved: {output}")
    return report


def _collect_lls(eval_set, eval_s_kv, mapped_t_kv, student, tok_s, mode):
    """重算指定 arm 的 per-sample LL（用于配对检验）。"""
    lls = []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        ek, ev = eval_s_kv[i].k, eval_s_kv[i].v
        mk_i, mv_i = mapped_t_kv[i].k, mapped_t_kv[i].v
        if mode == "K":
            c = build_cache(KV(k=mk_i, v=ev))
        elif mode == "V":
            c = build_cache(KV(k=ek, v=mv_i))
        elif mode == "KV":
            c = build_cache(KV(k=mk_i, v=mv_i))
        else:
            c = build_cache(KV(k=ek, v=ev))
        lls.append(answer_loglik(student, tok_s, c, q, s["answer"]))
    return lls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/phase7_second_domain_seed{a.seed}.json"
    run_second_domain(a.seed, a.n_calib, a.n_eval, out)


if __name__ == "__main__":
    main()
