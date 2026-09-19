"""I-1 / B1 + A1: gold-standard validation of the corrected evaluator.

For each student size, compares two decoding paths on the full Self (student
document cache) test set, sample by sample:

  (a) cache injection: build_cache(student_doc_kv) + query prefix -> greedy;
  (b) full-prefill reference: prefill concat(doc_ids, query_ids) -> greedy.

Reports (per student, per seed):
  - token-level greedy generation agreement (identical strings);
  - first-generated-token logit error distribution: mean |d|, p95 |d|, max |d|,
    the share of vocabulary entries with |d| > 0.5, and
    KL(p_full || p_cache) in nats (A1, REVISION_PLAN3 II.A1);
  - first-generated-token top-1 agreement;
  - normalized exact-match agreement.

`--domain squad` repeats the same validation on the second domain
(`data/squad_test_seed0.json`, student-only Self path).

`--attrib` runs the attribution probe: it separates the residual between the
two paths into (1) the bf16 numpy round-trip of the captured cache, (2) the
SDPA cache path vs the full-sequence path (repeated under eager attention),
and (3) implicit vs explicit position_ids.

Pre-registered gate (REVISION_PLAN3 II.A1):
  top-1 agreement 1.000 and median KL <= 0.01 nats  => the residual is
      numerical, the evaluator premise holds;
  top-1 disagreement or median KL > 0.1 nats        => stop and fix the
      evaluator before any other conclusion is used.

Usage:
  python3 experiments/phaseB_evalcheck.py --seed 0 --n-eval 56 --students 0.6B 1.7B 4B
  python3 experiments/phaseB_evalcheck.py --domain squad --n-eval 30 --students 0.6B
  python3 experiments/phaseB_evalcheck.py --attrib --students 0.6B --n-eval 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string

import numpy as np
import torch

from phase0_g0 import capture_kv
from phaseB_common import (
    MODELS,
    capture_all,
    build_cache,
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
SQUAD_PATH = os.path.join(DATA_DIR, "squad_test_seed0.json")


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return " ".join(s.split())


def norm_in(gen: str, answer: str) -> bool:
    return normalize(answer) in normalize(gen)


# ---------------------------------------------------------------------------
# Logit-space diagnostics (A1)
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(p || q) in nats, computed in float64 after a float32 softmax."""
    p = np.clip(softmax(p_logits), 1e-12, None)
    q = np.clip(softmax(q_logits), 1e-12, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def logit_diff_stats(injected: np.ndarray, reference: np.ndarray) -> dict:
    """Distribution of |delta logit| plus the two shape-level summaries."""
    d = np.abs(injected.astype(np.float64) - reference.astype(np.float64))
    return {
        "mean_abs_logit_err": float(d.mean()),
        "p95_abs_logit_err": float(np.percentile(d, 95)),
        "max_abs_logit_err": float(d.max()),
        "frac_abs_err_gt_0.5": float((d > 0.5).mean()),
        "kl_full_to_cache": kl_divergence(reference, injected),
        "first_top1_agree": bool(int(np.argmax(injected)) == int(np.argmax(reference))),
    }


# ---------------------------------------------------------------------------
# Decoding paths
# ---------------------------------------------------------------------------
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


def first_logits_injected(model, tok, kv, query: str, cache=None,
                          position_ids=None):
    """First-token logits after injecting a document cache and the query."""
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    kwargs = {}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    with torch.no_grad():
        out = model(q_ids, past_key_values=cache or build_cache(kv),
                    use_cache=True, **kwargs)
    return out.logits[0, -1].float().cpu().numpy()


def cache_from_past(past):
    """Rebuild a DynamicCache from in-memory past_key_values (no numpy round-trip)."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(past.key_cache, past.value_cache)):
        cache.update(k, v, i)
    return cache


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_samples(domain: str, seed: int, n_eval: int) -> list:
    if domain == "squad":
        with open(SQUAD_PATH) as f:
            return json.load(f)[:n_eval]
    return load_data(seed, "test")[:n_eval]


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------
def run_student(student_name: str, seed: int, n_eval: int,
                domain: str = "synthetic"):
    model, tok = load_model_gpu(MODELS[student_name])
    test = load_samples(domain, seed, n_eval)
    kvs = capture_all(model, tok, test)
    rows = []
    for i, s in enumerate(test):
        q = query_of(s)
        inj = greedy_answer_fixed(model, tok, build_cache(kvs[i]), q)
        ref, ref_logits = greedy_full(model, tok, s["doc"], q)
        inj_logits = first_logits_injected(model, tok, kvs[i], q)
        row = {
            "id": s["id"],
            "answer": s["answer"],
            "agree_tokens": bool(set(inj.split()) == set(ref.split())),
            "gen_injected": inj,
            "gen_full": ref,
            "em_injected": bool(norm_in(inj, s["answer"])),
            "em_full": bool(norm_in(ref, s["answer"])),
        }
        row.update(logit_diff_stats(inj_logits, ref_logits))
        rows.append(row)
        if (i + 1) % 14 == 0:
            print(f"  [{student_name} {i + 1}/{len(test)}]", flush=True)
    del model
    torch.cuda.empty_cache()
    return summarize_student(student_name, rows, domain)


def summarize_student(student_name: str, rows: list, domain: str) -> dict:
    max_errs = np.array([r["max_abs_logit_err"] for r in rows])
    mean_errs = np.array([r["mean_abs_logit_err"] for r in rows])
    p95_errs = np.array([r["p95_abs_logit_err"] for r in rows])
    kls = np.array([r["kl_full_to_cache"] for r in rows])
    top1 = np.array([r["first_top1_agree"] for r in rows], dtype=bool)
    diverged = np.array([not r["agree_tokens"] for r in rows], dtype=bool)
    worst = int(np.argmax(max_errs))
    return {
        "student": student_name,
        "domain": domain,
        "n": len(rows),
        # legacy fields (kept so the existing paper numbers stay reproducible)
        "token_agreement": float(np.mean([r["agree_tokens"] for r in rows])),
        "normalized_em_agreement": float(np.mean(
            [r["em_injected"] == r["em_full"] for r in rows])),
        "first_logits_max_abs_err_mean": float(max_errs.mean()),
        "first_logits_max_abs_err_max": float(max_errs.max()),
        "self_em_injected": float(np.mean([r["em_injected"] for r in rows])),
        "self_em_full": float(np.mean([r["em_full"] for r in rows])),
        # A1 additions
        "first_top1_agreement": float(top1.mean()),
        "mean_abs_logit_err_mean": float(mean_errs.mean()),
        "p95_abs_logit_err_mean": float(p95_errs.mean()),
        "p95_abs_logit_err_max": float(p95_errs.max()),
        "kl_median": float(np.median(kls)),
        "kl_mean": float(kls.mean()),
        "kl_max": float(kls.max()),
        "frac_items_max_err_gt_1": float((max_errs > 1.0).mean()),
        "max_err_median_on_top1_disagree": (
            float(np.median(max_errs[~top1])) if (~top1).any() else None),
        "max_err_median_on_top1_agree": (
            float(np.median(max_errs[top1])) if top1.any() else None),
        "worst_item_diverged": bool(diverged[worst]),
        "token_divergence_corr_with_max_err": (
            float(np.corrcoef(max_errs, diverged.astype(float))[0, 1])
            if diverged.any() and not diverged.all() else None),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Attribution probe (A1c)
# ---------------------------------------------------------------------------
def probe_item(model, tok, sample: dict, position_ids_mode: bool = True) -> dict:
    """Separate round-trip, kernel-path and position-id effects for one sample."""
    q = query_of(sample)
    doc_ids = tok(sample["doc"], return_tensors="pt").input_ids.to(model.device)
    q_ids = tok(q, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    ids = torch.cat([doc_ids, q_ids], dim=1)
    with torch.no_grad():
        full_out = model(ids, use_cache=True)
        doc_out = model(doc_ids, use_cache=True)
    logits_full = full_out.logits[0, -1].float().cpu().numpy()
    n_doc = doc_ids.shape[1]

    cache_direct = cache_from_past(doc_out.past_key_values)
    kv = capture_kv(model, tok, sample["doc"])
    cache_round = build_cache(kv)

    logits_direct = first_logits_injected(model, tok, kv, q, cache=cache_direct)
    logits_round = first_logits_injected(model, tok, kv, q, cache=cache_round)

    out = {
        "id": sample["id"],
        "n_doc": int(n_doc),
        "roundtrip": logit_diff_stats(logits_round, logits_direct),
        "kernel_path": logit_diff_stats(logits_round, logits_full),
    }
    if position_ids_mode:
        pos = torch.arange(n_doc, n_doc + q_ids.shape[1],
                           device=model.device).unsqueeze(0)
        logits_pos = first_logits_injected(model, tok, kv, q,
                                           cache=build_cache(kv),
                                           position_ids=pos)
        out["position_ids"] = logit_diff_stats(logits_pos, logits_round)
    return out


def aggregate_probe(rows: list, keys: tuple) -> dict:
    agg = {}
    for key in keys:
        parts = [r[key] for r in rows if key in r]
        if not parts:
            continue
        agg[key] = {
            "max_abs_logit_err_mean": float(np.mean(
                [p["max_abs_logit_err"] for p in parts])),
            "max_abs_logit_err_max": float(np.max(
                [p["max_abs_logit_err"] for p in parts])),
            "mean_abs_logit_err_mean": float(np.mean(
                [p["mean_abs_logit_err"] for p in parts])),
            "kl_median": float(np.median([p["kl_full_to_cache"] for p in parts])),
            "kl_mean": float(np.mean([p["kl_full_to_cache"] for p in parts])),
            "first_top1_agreement": float(np.mean(
                [p["first_top1_agree"] for p in parts])),
        }
    return agg


def run_attrib(student_name: str, seed: int, n_items: int, output: str):
    model, tok = load_model_gpu(MODELS[student_name])
    test = load_samples("synthetic", seed, n_items)

    kv1 = capture_kv(model, tok, test[0]["doc"])
    kv2 = capture_kv(model, tok, test[0]["doc"])
    determinism = float(max(np.abs(kv1.k - kv2.k).max(),
                            np.abs(kv1.v - kv2.v).max()))

    rows = [probe_item(model, tok, s) for s in test]
    del model
    torch.cuda.empty_cache()

    eager_model, eager_tok = load_model_gpu(MODELS[student_name],
                                            attn_implementation="eager")
    eager_rows = [probe_item(eager_model, eager_tok, s, position_ids_mode=False)
                  for s in test]
    del eager_model
    torch.cuda.empty_cache()

    report = {
        "task": "phaseB_evalcheck_attrib",
        "student": student_name,
        "seed": seed,
        "n": len(test),
        "cache_determinism_max_abs_diff": determinism,
        "default_attn": aggregate_probe(rows, ("roundtrip", "kernel_path",
                                               "position_ids")),
        "eager_attn": aggregate_probe(eager_rows, ("kernel_path",)),
        "rows": rows,
        "eager_rows": eager_rows,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[evalcheck-attrib] saved: {output}")
    print(f"  cache determinism (max abs diff): {determinism:.3e}")
    for name, agg in report["default_attn"].items():
        print(f"  {name:14s} top1={agg['first_top1_agreement']:.3f} "
              f"kl_med={agg['kl_median']:.4f} "
              f"max_err={agg['max_abs_logit_err_max']:.3f}")
    if report["eager_attn"].get("kernel_path"):
        agg = report["eager_attn"]["kernel_path"]
        print(f"  {'eager kernel':14s} top1={agg['first_top1_agreement']:.3f} "
              f"kl_med={agg['kl_median']:.4f} "
              f"max_err={agg['max_abs_logit_err_max']:.3f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--students", nargs="+", default=["0.6B", "1.7B", "4B"])
    ap.add_argument("--domain", default="synthetic", choices=["synthetic", "squad"])
    ap.add_argument("--attrib", action="store_true",
                    help="run the cache/kernel/position attribution probe instead")
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    if a.attrib:
        out = a.output or os.path.join(REPORT_DIR, "phaseB_evalcheck_attrib.json")
        run_attrib(a.students[0], a.seed, a.n_eval, out)
        return

    res = [run_student(s, a.seed, a.n_eval, a.domain) for s in a.students]
    report = {"task": "phaseB_evalcheck", "seed": a.seed, "domain": a.domain,
              "n_eval": a.n_eval, "students": res}
    out = a.output or f"{REPORT_DIR}/phaseB_evalcheck_seed{a.seed}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[evalcheck] saved: {out}")
    for r in res:
        print(f"  {r['student']:5s} token_agree={r['token_agreement']:.3f} "
              f"normEM_agree={r['normalized_em_agreement']:.3f} "
              f"top1={r['first_top1_agreement']:.3f} "
              f"kl_med={r['kl_median']:.4f} "
              f"logit_err(mean/max)={r['first_logits_max_abs_err_mean']:.3f}/"
              f"{r['first_logits_max_abs_err_max']:.3f} "
              f"SelfEM={r['self_em_injected']:.3f}/{r['self_em_full']:.3f}")


if __name__ == "__main__":
    main()
