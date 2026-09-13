"""I-1 / B1: gold-standard validation of the corrected evaluator.

For each student size, compares two decoding paths on the full Self (student
document cache) test set, sample by sample:

  (a) cache injection: build_cache(student_doc_kv) + query prefix -> greedy;
  (b) full-prefill reference: prefill concat(doc_ids, query_ids) -> greedy.

Reports (per student, per seed):
  - token-level greedy generation agreement (identical strings);
  - first-generated-token logits max abs error (injected vs full prefill);
  - normalized exact-match agreement.

Pre-registered gate: agreement ~= 100% => the paper's premise holds; a clear
shortfall => the evaluator must be fixed before any other conclusion.

Usage:
  python3 phaseB_evalcheck.py --seed 0 --n-eval 56 --students 0.6B 1.7B 4B
"""
from __future__ import annotations

import argparse
import json
import re
import string

import numpy as np
import torch

from phaseB_common import (
    MODELS,
    capture_all,
    build_cache,
    greedy_answer_fixed,
    load_data,
    load_model_gpu,
    query_of,
)

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def norm_in(gen: str, answer: str) -> bool:
    return normalize(answer) in normalize(gen)


def greedy_full(model, tok, doc: str, query: str, max_new: int = 16):
    """Reference greedy from a full doc+query prefill. Returns (text, first_logits).

    Doc and query are tokenized separately and concatenated so the reference
    matches the injected path exactly (joint tokenization would merge BPE across
    the boundary).
    """
    doc_ids = tok(doc, return_tensors="pt").input_ids
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([doc_ids, q_ids], dim=1).to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    first_logits = out.logits[0, -1].float().cpu().numpy()
    gen = []
    next_in = None
    for _ in range(max_new):
        if next_in is None:
            logits = out.logits[0, -1]
        else:
            with torch.no_grad():
                out = model(next_in, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1]
            past = out.past_key_values
        t = int(torch.argmax(logits).item())
        gen.append(t)
        next_in = torch.tensor([[t]], device=model.device)
        if t == tok.eos_token_id:
            break
    return tok.decode(gen, skip_special_tokens=True), first_logits


def first_logits_injected(model, tok, kv, query: str):
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    with torch.no_grad():
        out = model(q_ids, past_key_values=build_cache(kv), use_cache=True)
    return out.logits[0, -1].float().cpu().numpy()


def run_student(student_name: str, seed: int, n_eval: int):
    model, tok = load_model_gpu(MODELS[student_name])
    test = load_data(seed, "test")[:n_eval]
    kvs = capture_all(model, tok, test)
    rows = []
    for i, s in enumerate(test):
        q = query_of(s)
        inj = greedy_answer_fixed(model, tok, build_cache(kvs[i]), q)
        ref, ref_logits = greedy_full(model, tok, s["doc"], q)
        inj_logits = first_logits_injected(model, tok, kvs[i], q)
        max_err = float(np.max(np.abs(inj_logits - ref_logits)))
        rows.append({
            "id": s["id"],
            "answer": s["answer"],
            "agree_tokens": bool(set(inj.split()) == set(ref.split())),
            "gen_injected": inj,
            "gen_full": ref,
            "first_logits_max_abs_err": max_err,
            "em_injected": bool(norm_in(inj, s["answer"])),
            "em_full": bool(norm_in(ref, s["answer"])),
        })
        if (i + 1) % 14 == 0:
            print(f"  [{student_name} {i+1}/{len(test)}]", flush=True)
    del model
    torch.cuda.empty_cache()
    n = len(rows)
    return {
        "student": student_name,
        "n": n,
        "token_agreement": float(np.mean([r["agree_tokens"] for r in rows])),
        "normalized_em_agreement": float(np.mean(
            [r["em_injected"] == r["em_full"] for r in rows])),
        "first_logits_max_abs_err_mean": float(np.mean(
            [r["first_logits_max_abs_err"] for r in rows])),
        "first_logits_max_abs_err_max": float(np.max(
            [r["first_logits_max_abs_err"] for r in rows])),
        "self_em_injected": float(np.mean([r["em_injected"] for r in rows])),
        "self_em_full": float(np.mean([r["em_full"] for r in rows])),
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
    report = {"task": "phaseB_evalcheck", "seed": a.seed,
              "n_eval": a.n_eval, "students": res}
    out = a.output or f"/workspace/v3/reports/phaseB_evalcheck_seed{a.seed}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[evalcheck] saved: {out}")
    for r in res:
        print(f"  {r['student']:5s} token_agree={r['token_agreement']:.3f} "
              f"normEM_agree={r['normalized_em_agreement']:.3f} "
              f"logit_err(mean/max)={r['first_logits_max_abs_err_mean']:.3f}/"
              f"{r['first_logits_max_abs_err_max']:.3f} "
              f"SelfEM={r['self_em_injected']:.3f}/{r['self_em_full']:.3f}")


if __name__ == "__main__":
    main()
