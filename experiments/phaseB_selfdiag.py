"""I-6 / C2: diagnose the non-monotonic Self exact match (0.6B 0.90, 1.7B 0.44, 4B 0.82).

For each student size, generate the corrected greedy Self answer for every test
sample and report raw EM, normalized EM, token-level F1, extractive-answer
containment, generation length, and an error breakdown (format vs content).

Usage:
  python3 experiments/phaseB_selfdiag.py --seed 0 --n-eval 56 --students 0.6B 1.7B 4B
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string

import numpy as np
import torch

from phaseB_common import (
    MODELS,
    build_cache,
    capture_all,
    exact_match,
    greedy_answer_fixed,
    load_data,
    load_model_gpu,
    query_of,
)

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    hit = 0
    for t in g:
        if common.get(t, 0) > 0:
            common[t] -= 1
            hit += 1
    if hit == 0:
        return 0.0
    prec, rec = hit / len(p), hit / len(g)
    return 2 * prec * rec / (prec + rec)


def run_student(name, seed, n_eval):
    model, tok = load_model_gpu(MODELS[name])
    test = load_data(seed, "test")[:n_eval]
    kvs = capture_all(model, tok, test)
    rows = []
    for i, s in enumerate(test):
        gen = greedy_answer_fixed(model, tok, build_cache(kvs[i]), query_of(s))
        raw = exact_match(gen, s["answer"])
        nrm = normalize(s["answer"]) in normalize(gen)
        rows.append({
            "id": s["id"], "answer": s["answer"], "gen": gen,
            "raw_em": bool(raw), "norm_em": bool(nrm),
            "f1": token_f1(gen, s["answer"]), "gen_tokens": len(gen.split()),
        })
    del model
    torch.cuda.empty_cache()
    n = len(rows)
    fmt_fixes = sum(1 for r in rows if (not r["raw_em"]) and r["norm_em"])
    return {
        "student": name, "n": n,
        "raw_em": float(np.mean([r["raw_em"] for r in rows])),
        "norm_em": float(np.mean([r["norm_em"] for r in rows])),
        "token_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_gen_tokens": float(np.mean([r["gen_tokens"] for r in rows])),
        "raw_fail_norm_pass": int(fmt_fixes),
        "raw_fail_norm_fail": int(sum(1 for r in rows if not r["raw_em"] and not r["norm_em"])),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--students", nargs="+", default=["0.6B", "1.7B", "4B"])
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    res = [run_student(s, a.seed, a.n_eval) for s in a.students]
    out = a.output or f"{REPORT_DIR}/phaseB_selfdiag_seed{a.seed}.json"
    with open(out, "w") as f:
        json.dump({"task": "phaseB_selfdiag", "seed": a.seed, "students": res},
                  f, ensure_ascii=False, indent=2)
    print(f"[selfdiag] saved: {out}")
    for r in res:
        print(f"  {r['student']:5s} rawEM={r['raw_em']:.3f} normEM={r['norm_em']:.3f} "
              f"F1={r['token_f1']:.3f} len={r['mean_gen_tokens']:.1f} "
              f"rawFail/normPass={r['raw_fail_norm_pass']}")


if __name__ == "__main__":
    main()
