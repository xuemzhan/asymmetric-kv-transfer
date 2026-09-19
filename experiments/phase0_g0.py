"""V3 Phase 0：生死门 G0 —— 强映射器下 V 是否仍不可传输？

对应 V3 方案 §3 Phase 0 / §4 Gate G0。

科学问题（生死门）：
    论文一的 V 失败（cosine 0.577）用朴素 Ridge + KV-cosine 指标得到。
    反对意见："两个坐标系即使正交也能通过线性旋转对齐，V 原始 cosine 低
    不代表无高质量映射。" G0 必须回答：更强的映射器（affine/procrustes/
    CCA/whitened）能否把教师 V 内容迁移到学生可消费的水平。

协议（继承论文一纪律：真实任务分为主指标，禁止用 KV cosine 冒充 retention）：
  对每个 held-out 样本，学生只看到 query，文档 KV 全部来自注入：
    - Self    : 学生自 prefill 文档的 KV（gold 状态参照）
    - K-only  : 学生 K + 映射教师 V
    - V-only  : 映射教师 K + 学生 V      ← 隔离 V 侧是否可迁移
    - Joint   : 映射教师 K + 映射教师 V   ← 全量 handoff
  参照：
    - student_full : 学生 prefill 文档+问题（学生自己能力的上界）
    - teacher_full : 教师 prefill 文档+问题（教师上界，gap 来源）
  主指标：答案 token 的 teacher-forced log-likelihood（连续、灵敏）+ 贪心精确匹配。

G0 判定：
  - V-only/Joint 的 LL 显著不低于 Self/student_full → V 可被强映射器恢复
    → G0 FAIL（机理死亡 → 软着陆相图方向）。
  - 强映射器下 V-only/Joint 仍显著低于 Self → V 失败是本质 → G0 PASS
    （机理存活，进入 Phase 1 因果基板 + 标量 g）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")
sys.path.insert(0, APCS_DIR)
import torch

from apcs.mapper.math import (
    AffineMapper,
    CCAMapper,
    ProcrustesMapper,
    RidgePerHeadMapper,
    WhitenedMapper,
)
from apcs.rope.runner import _rope_pairs, de_rope

MODEL_PATHS = {
    "teacher": os.path.join(MODELS_DIR, "Qwen--Qwen3-8B", "snapshots", "master"),
    "student": os.path.join(MODELS_DIR, "Qwen--Qwen3-0.6B", "snapshots", "master"),
}

TEACHER_LAYERS = 36
STUDENT_LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0


def layer_map_proportional() -> list[list[int]]:
    return [[min(TEACHER_LAYERS - 1, round(s * TEACHER_LAYERS / STUDENT_LAYERS))]
            for s in range(STUDENT_LAYERS)]


@dataclass
class KV:
    k: np.ndarray  # (L, S, H, D)
    v: np.ndarray


def load_model(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tok = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tok


def capture_kv(model, tok, text: str) -> KV:
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    pkv = out.past_key_values
    k_layers, v_layers = [], []
    for k, v in zip(pkv.key_cache, pkv.value_cache):
        k_layers.append(k[0].permute(1, 0, 2))
        v_layers.append(v[0].permute(1, 0, 2))
    return KV(
        k=torch.stack(k_layers).float().cpu().numpy().astype(np.float32),
        v=torch.stack(v_layers).float().cpu().numpy().astype(np.float32),
    )


def build_cache(kv: KV):
    from transformers.cache_utils import DynamicCache

    L, S, H, D = kv.k.shape
    kt = torch.from_numpy(kv.k).permute(0, 2, 1, 3).to(torch.bfloat16).to("cuda:0")
    vt = torch.from_numpy(kv.v).permute(0, 2, 1, 3).to(torch.bfloat16).to("cuda:0")
    cache = DynamicCache()
    for i in range(L):
        cache.update(kt[i].unsqueeze(0), vt[i].unsqueeze(0), i)
    return cache


def answer_loglik(model, tok, cache, query: str, answer: str) -> float:
    """注入 cache 后，teacher-force query+answer，返回 answer token 的平均 logprob。"""
    a_ids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    past = cache
    if q_ids.shape[1] > 0:
        with torch.no_grad():
            out = model(q_ids, past_key_values=cache, use_cache=True)
        past = out.past_key_values
    total = 0.0
    for j in range(a_ids.shape[1]):
        with torch.no_grad():
            out = model(
                a_ids[:, j:j + 1], past_key_values=past, use_cache=True
            )
        lg = out.logits[0, 0].float()
        logp = torch.log_softmax(lg, dim=-1)[a_ids[0, j]].item()
        total += logp
        past = out.past_key_values
    return total / max(a_ids.shape[1], 1)


def greedy_answer(model, tok, cache, query: str, max_new: int = 16) -> str:
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    with torch.no_grad():
        out = model(q_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
    gen = []
    for _ in range(max_new):
        with torch.no_grad():
            out = model(torch.tensor([[gen[-1]]], device="cuda:0") if gen else q_ids[:, -1:], past_key_values=past, use_cache=True)
        tok_id = int(torch.argmax(out.logits[0, -1]).item())
        gen.append(tok_id)
        past = out.past_key_values
        if tok_id == tok.eos_token_id:
            break
    return tok.decode(gen, skip_special_tokens=True)


def exact_match(generated: str, answer: str) -> bool:
    return answer.strip().lower() in generated.lower()


def novelty_probe(model, tok, samples, max_new=24):
    """Novelty probe: no-context OOD entity questions should be near random."""
    em_list, ll_list = [], []
    for s in samples:
        q = "Question: " + s["q"] + "\nAnswer:"
        gen = greedy_answer(model, tok, None, q, max_new)
        em_list.append(1.0 if exact_match(gen, s["answer"]) else 0.0)
        ll_list.append(answer_loglik(model, tok, None, q, s["answer"]))
    return {
        "n": len(samples),
        "em": float(np.mean(em_list)),
        "mean_ll": float(np.mean(ll_list)),
    }


def run_g0(seed: int, n_calib: int, n_eval: int, output: str, eval_override=None, run_probe=False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train = json.load(open(f"{DATA_DIR}/train.json"))
    test = json.load(open(f"{DATA_DIR}/test.json"))
    calib = train[:n_calib]
    eval_set = test[:n_eval]
    if eval_override is not None:
        eval_set = eval_override

    layer_map = layer_map_proportional()
    inv_freq = _rope_pairs(HEAD_DIM, ROPE_THETA)
    de_rope_k = lambda k, pos: de_rope(k, pos, inv_freq)  # noqa: E731

    # ---------- 教师 KV + 教师 full 基线 ----------
    print("[G0] load teacher 8B ...")
    teacher, tok_t = load_model(MODEL_PATHS["teacher"])

    calib_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in calib]
    eval_t_kv = [capture_kv(teacher, tok_t, s["doc"]) for s in eval_set]

    # teacher_full：教师 prefill doc+query 后的 answer loglik（gap 上界）
    # 注意：KV 只捕获 doc+query，不含 "\nAnswer:" 前缀，避免双重计数
    teacher_ll, teacher_em = [], []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        c = build_cache(eval_t_kv[i])
        teacher_ll.append(answer_loglik(teacher, tok_t, c, q, s["answer"]))
        teacher_em.append(float(exact_match(greedy_answer(teacher, tok_t, c, q), s["answer"])))
    probe_t = novelty_probe(teacher, tok_t, eval_set) if run_probe else None
    del teacher
    torch.cuda.empty_cache()

    # ---------- 学生 KV + 学生 full 基线 ----------
    print("[G0] load student 0.6B ...")
    student, tok_s = load_model(MODEL_PATHS["student"])
    calib_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in calib]
    eval_s_kv = [capture_kv(student, tok_s, s["doc"]) for s in eval_set]
    student_ll, student_em = [], []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        c = build_cache(eval_s_kv[i])
        student_ll.append(answer_loglik(student, tok_s, c, q, s["answer"]))
        student_em.append(float(exact_match(greedy_answer(student, tok_s, c, q), s["answer"])))
    probe_s = novelty_probe(student, tok_s, eval_set) if run_probe else None

    # ---------- 校准堆叠 ----------
    def stack_kv(kvs):
        return KV(
            k=np.concatenate([kv.k for kv in kvs], axis=1),
            v=np.concatenate([kv.v for kv in kvs], axis=1),
        )

    calib_t_all = stack_kv(calib_t_kv)
    calib_s_all = stack_kv(calib_s_kv)
    pos_all = np.arange(calib_t_all.k.shape[1], dtype=np.float64)

    # ---------- 拟合强映射器 ----------
    mappers = {
        "Affine": (AffineMapper(lam=1e-3), AffineMapper(lam=1e-3)),
        "Whitened": (WhitenedMapper(lam=1e-3), WhitenedMapper(lam=1e-3)),
        "Procrustes": (ProcrustesMapper(), ProcrustesMapper()),
        "CCA-r128": (CCAMapper(r=HEAD_DIM, lam=1e-3), CCAMapper(r=HEAD_DIM, lam=1e-3)),
        "RidgePerHead": (RidgePerHeadMapper(lam=1e-3), RidgePerHeadMapper(lam=1e-3)),
    }
    for name, (mk, mv) in mappers.items():
        t0 = time.time()
        mk.fit(calib_t_all.k, calib_s_all.k, layer_map, kv_kind="K",
               positions=pos_all, de_rope_fn=de_rope_k)
        mv.fit(calib_t_all.v, calib_s_all.v, layer_map, kv_kind="V",
               positions=pos_all, de_rope_fn=None)
        print(f"  [{name}] fit {time.time()-t0:.1f}s")

    def map_kv(name, kv_t, pos):
        mk, mv = mappers[name]
        km = mk.transform(kv_t.k, layer_map, kv_kind="K", positions=pos, de_rope_fn=de_rope_k)
        vm = mv.transform(kv_t.v, layer_map, kv_kind="V", positions=pos, de_rope_fn=None)
        return KV(k=km.astype(np.float32), v=vm.astype(np.float32))

    # ---------- 四路消融 ----------
    rows = []
    for i, s in enumerate(eval_set):
        q = "\n\nQuestion: " + s["q"] + "\nAnswer:"
        pos_i = np.arange(eval_t_kv[i].k.shape[1], dtype=np.float64)
        row = {"id": s["id"], "hop": s["hop"], "answer": s["answer"]}

        # Self
        c = build_cache(eval_s_kv[i])
        row["Self"] = answer_loglik(student, tok_s, c, q, s["answer"])
        row["Self_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])

        # student_full / teacher_full
        row["student_full"] = student_ll[i]
        row["student_full_em"] = student_em[i]
        row["teacher_full"] = teacher_ll[i]
        row["teacher_full_em"] = teacher_em[i]

        for name in mappers:
            m = map_kv(name, eval_t_kv[i], pos_i)
            # V-only: 学生 K + 映射教师 V
            c = build_cache(KV(k=eval_s_kv[i].k, v=m.v))
            row[f"{name}_V-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
            row[f"{name}_V-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])
            # K-only: 映射教师 K + 学生 V
            c = build_cache(KV(k=m.k, v=eval_s_kv[i].v))
            row[f"{name}_K-only"] = answer_loglik(student, tok_s, c, q, s["answer"])
            row[f"{name}_K-only_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])
            # Joint
            c = build_cache(KV(k=m.k, v=m.v))
            row[f"{name}_Joint"] = answer_loglik(student, tok_s, c, q, s["answer"])
            row[f"{name}_Joint_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])
        rows.append(row)
        if (i + 1) % 4 == 0:
            print(f"  [{i+1}/{len(eval_set)}] done")

    # Hop decomposition
    hops = sorted({r["hop"] for r in rows})
    summary_hop = {}
    for h in hops:
        sub = [r for r in rows if r["hop"] == h]
        entry = {}
        for key in ["Self", "student_full", "teacher_full"]:
            xs = np.array([r[key] for r in sub], dtype=float)
            entry[key] = {"mean": float(xs.mean()), "n": len(xs)}
        for name in mappers:
            for arm in ["V-only", "K-only", "Joint"]:
                k = f"{name}_{arm}"
                xs = np.array([r[k] for r in sub], dtype=float)
                entry[k] = {"mean": float(xs.mean()), "n": len(xs)}
        summary_hop[str(h)] = entry

    # ---------- 汇总 ----------
    def agg(key):
        xs = np.array([r[key] for r in rows], dtype=float)
        n = len(xs)
        return {
            "mean": float(xs.mean()),
            "ci95": float(1.96 * xs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        }

    # 只汇总"每映射器 × 臂"的 LL（主指标）
    summary = {
        "Self": agg("Self"),
        "student_full": agg("student_full"),
        "teacher_full": agg("teacher_full"),
    }
    em = {
        "Self": float(np.mean([r["Self_em"] for r in rows])),
        "student_full": float(np.mean([r["student_full_em"] for r in rows])),
        "teacher_full": float(np.mean([r["teacher_full_em"] for r in rows])),
    }
    for name in mappers:
        for arm in ["V-only", "K-only", "Joint"]:
            summary[f"{name}_{arm}"] = agg(f"{name}_{arm}")
            em[f"{name}_{arm}"] = float(np.mean([r[f"{name}_{arm}_em"] for r in rows]))

    report = {
        "task": "G0_linchpin",
        "seed": seed,
        "pair": "Qwen3-8B -> Qwen3-0.6B",
        "n_calib": n_calib,
        "n_eval": n_eval,
        "model_paths": MODEL_PATHS,
        "summary_ll": summary,
        "exact_match": em,
        "summary_hop": summary_hop,
        "rows": rows,
    }
    if run_probe:
        report["novelty_probe"] = {"teacher": probe_t, "student": probe_s}
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--output", type=str, default="")
    a = ap.parse_args()
    out = a.output or f"{REPORT_DIR}/g0_seed{a.seed}.json"
    report = run_g0(a.seed, a.n_calib, a.n_eval, out)
    print(json.dumps(report["summary_ll"], ensure_ascii=False, indent=2))
    print("exact_match:", json.dumps(report["exact_match"], indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
