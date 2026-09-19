"""Shared utilities for review-response (B-class) experiments.

Generalizes the single-pair helpers in phase0_g0.py to arbitrary Qwen3 pairs and
arbitrary layer maps. All new B-class scripts import from here so the KV capture,
mapper fitting, and injection protocol remain identical to the published pipeline.

Design notes:
- Model loading / KV capture / cache building / scoring are imported unchanged
  from phase0_g0.py so numbers stay comparable with prior reports.
- A layer map is a list (length = student layers) of teacher-layer indices.
  AffineMapper/transform average over the teacher indices given per student layer,
  and the fit is per-(layer, head) with an intercept.
"""
from __future__ import annotations

import json
import os
import sys
import time

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

from phase0_g0 import (  # noqa: E402
    KV,
    HEAD_DIM,
    ROPE_THETA,
    answer_loglik,
    build_cache,
    capture_kv,
    exact_match,
    greedy_answer,
    load_model,
)
from apcs.mapper.math import AffineMapper  # noqa: E402
from apcs.rope.runner import _rope_pairs, de_rope  # noqa: E402
from stats_utils import bootstrap_ci95, paired_wilcoxon_test  # noqa: E402

MSC = MODELS_DIR
MODELS = {
    "8B": f"{MSC}/Qwen--Qwen3-8B/snapshots/master",
    "4B": f"{MSC}/Qwen--Qwen3-4B/snapshots/master",
    "1.7B": f"{MSC}/Qwen--Qwen3-1.7B/snapshots/master",
    "0.6B": f"{MSC}/Qwen--Qwen3-0.6B/snapshots/master",
}

# teacher, student, teacher_layers, student_layers
PAIRS = {
    "8B_0.6B": ("8B", "0.6B", 36, 28),
    "4B_1.7B": ("4B", "1.7B", 36, 28),
    "4B_0.6B": ("4B", "0.6B", 36, 28),
    "8B_1.7B": ("8B", "1.7B", 36, 28),
    "1.7B_0.6B": ("1.7B", "0.6B", 28, 28),
    "8B_4B": ("8B", "4B", 36, 36),
}

INV_FREQ = _rope_pairs(HEAD_DIM, ROPE_THETA)


def de_rope_k(k, pos):
    return de_rope(k, pos, INV_FREQ)


def load_data(seed: int, split: str):
    """Load a per-seed v2 split.

    `V3_DATA_DIR` overrides the default data directory so the optional
    wide-document split (REVISION_PLAN3 II.A5) can be evaluated without
    overwriting the published split.
    """
    data_dir = os.environ.get("V3_DATA_DIR", DATA_DIR)
    return json.load(open(f"{data_dir}/{split}_v2_seed{seed}.json"))


def load_pair(pair: str):
    t_name, s_name, t_layers, s_layers = PAIRS[pair]
    print(f"[common] load teacher {t_name} ({t_layers}L) ...", flush=True)
    teacher, tok_t = load_model_gpu(MODELS[t_name])
    print(f"[common] load student {s_name} ({s_layers}L) ...", flush=True)
    student, tok_s = load_model_gpu(MODELS[s_name])
    return teacher, tok_t, student, tok_s, t_layers, s_layers


def load_teacher(pair: str):
    t_name, _, t_layers, _ = PAIRS[pair]
    print(f"[common] load teacher {t_name} ({t_layers}L) ...", flush=True)
    model, tok = load_model_gpu(MODELS[t_name])
    return model, tok, t_layers


def load_student(pair: str):
    _, s_name, _, s_layers = PAIRS[pair]
    print(f"[common] load student {s_name} ({s_layers}L) ...", flush=True)
    model, tok = load_model_gpu(MODELS[s_name])
    return model, tok, s_layers


def load_model_gpu(path: str, attn_implementation: str | None = None):
    """Deterministic full-GPU load, streamed to the device.

    The container enforces a cgroup memory cap and a supervisor watchdog that
    SIGTERMs the top-RSS process once the cap is approached. A plain
    `from_pretrained` then `.to('cuda:0')` materialises a full second copy of
    the weights in host RAM (the 8B teacher reaches ~34GB RSS), which trips
    that watchdog. `low_cpu_mem_usage` plus an explicit `device_map` streams
    the shards straight to `cuda:0`, so host RSS stays near the KV arrays.

    An explicit `{"": 0}` map (not `"auto"`) pins every module to the GPU, so
    the load stays deterministic; `auto` could CPU-offload an 8B teacher
    depending on transient device memory and be ~10x slower.

    `attn_implementation` is only passed when given, so every existing caller
    keeps the library default; the evaluator attribution probe passes
    "eager" to separate SDPA cache-path numerics from an implementation bug.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True,
              "device_map": {"": 0}}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tok = AutoTokenizer.from_pretrained(path)
    model.eval()
    return model, tok


def capture_all(model, tok, samples) -> list:
    return [capture_kv(model, tok, s["doc"]) for s in samples]


def capture_pair_kv(pair: str, train_docs, eval_docs):
    """Capture teacher and student KV with separated model lifetimes (OOM-safe).

    Returns (calib_t, eval_t, calib_s, eval_s, student, tok_s, t_layers, s_layers).
    The teacher is freed before the student is loaded, so pairs like 8B->4B or
    runs on a partly-occupied GPU do not exceed device memory.
    """
    teacher, tok_t, t_layers = load_teacher(pair)
    calib_t = capture_all(teacher, tok_t, train_docs)
    eval_t = capture_all(teacher, tok_t, eval_docs)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(pair)
    calib_s = capture_all(student, tok_s, train_docs)
    eval_s = capture_all(student, tok_s, eval_docs)
    return calib_t, eval_t, calib_s, eval_s, student, tok_s, t_layers, s_layers


def stack_kv(kvs) -> KV:
    return KV(
        k=np.concatenate([kv.k for kv in kvs], axis=1),
        v=np.concatenate([kv.v for kv in kvs], axis=1),
    )


def pos_of(kv: KV) -> np.ndarray:
    return np.arange(kv.k.shape[1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Layer maps
# ---------------------------------------------------------------------------
def layer_map_proportional(t_layers: int, s_layers: int) -> list[list[int]]:
    return [[min(t_layers - 1, round(s * t_layers / s_layers))]
            for s in range(s_layers)]


def layer_map_offset(t_layers: int, s_layers: int, offset: int) -> list[list[int]]:
    """Shift the proportional map by `offset` teacher layers (clamped)."""
    base = [min(t_layers - 1, round(s * t_layers / s_layers))
            for s in range(s_layers)]
    return [[int(np.clip(b + offset, 0, t_layers - 1))] for b in base]


def layer_map_permuted(t_layers: int, s_layers: int, seed: int) -> list[list[int]]:
    """Random permutation of the proportional teacher indices (destroys alignment)."""
    base = [min(t_layers - 1, round(s * t_layers / s_layers))
            for s in range(s_layers)]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(s_layers)
    return [[base[perm[s]]] for s in range(s_layers)]


def layer_map_identity(t_layers: int, s_layers: int) -> list[list[int]]:
    return [[s] for s in range(s_layers)]


def layer_map_shift(t_layers: int, s_layers: int, k: int) -> list[list[int]]:
    """Cyclic shift by k within the proportional teacher index sequence."""
    prop = layer_map_proportional(t_layers, s_layers)
    n = s_layers
    return [[prop[(s + k) % n][0]] for s in range(n)]


def layer_map_learned_topk(
    t_layers: int,
    s_layers: int,
    calib_t: KV,
    calib_s: KV,
    k: int = 1,
    lam: float = 1e-3,
    kind: str = "K",
    max_tokens: int = 1000,
    subsample_seed: int = 0,
) -> tuple[list[list[int]], dict]:
    """Learn source teacher layer(s) per student layer by calibration fit quality.

    For each student layer s we fit a per-head affine map from every teacher layer
    t to s (on calibration tokens) and score held-out reconstruction R^2 via a
    2-fold split of the calibration tokens. The top-k teacher layers are kept.
    This tests the "is V failure caused by a poor proportional layer map?" question.

    Calibration tokens are subsampled to `max_tokens` (default 2000) for speed;
    the per-(layer, head) solves are otherwise the dominant cost.
    """
    L_s = calib_s.k.shape[0]
    n_total = calib_t.k.shape[1]
    if max_tokens is not None and n_total > max_tokens:
        sel = np.sort(np.random.RandomState(subsample_seed).choice(
            n_total, size=max_tokens, replace=False))
    else:
        sel = np.arange(n_total)
    scores = np.zeros((L_s, t_layers), dtype=np.float64)
    positions = sel.astype(np.float64)
    # precompute (de-rope for K is done once per teacher layer, on subsampled tokens)
    x_ts = []
    for t in range(t_layers):
        if kind == "K":
            x_ts.append(de_rope_k(calib_t.k[t][sel], positions))
        else:
            x_ts.append(calib_t.v[t][sel])
    y_ts = [(calib_s.k[s] if kind == "K" else calib_s.v[s])[sel]
            for s in range(L_s)]  # list of (S,H,D)
    H = x_ts[0].shape[1]
    eps = np.eye(HEAD_DIM)
    tr = np.arange(0, len(sel), 2)
    te = np.arange(1, len(sel), 2)
    # Factor once per (teacher layer, head); batch the L_s solves via stacked RHS.
    for t in range(t_layers):
        x_t = x_ts[t]
        for h in range(H):
            a = x_t[:, h, :].astype(np.float64)
            xa = a[tr]
            xm = xa.mean(0, keepdims=True)
            Xc = xa - xm
            G = Xc.T @ Xc + lam * eps
            rhs, yte, ymu = [], [], []
            for s in range(L_s):
                y = y_ts[s][:, h, :].astype(np.float64)
                ym = y[tr].mean(0, keepdims=True)
                rhs.append(Xc.T @ (y[tr] - ym))
                yte.append(y[te])
                ymu.append(ym)
            RHS = np.concatenate(rhs, axis=1)          # (D, D*L_s)
            B = np.linalg.solve(G, RHS)                # (D, D*L_s)
            a_te = a[te] - xm
            for s in range(L_s):
                pred = a_te @ B[:, s * HEAD_DIM:(s + 1) * HEAD_DIM] + ymu[s]
                y_s = yte[s]
                ss_res = float(((y_s - pred) ** 2).sum())
                ss_tot = float(((y_s - y_s.mean(0, keepdims=True)) ** 2).sum()) + 1e-12
                scores[s, t] += 1.0 - ss_res / ss_tot
    scores /= H
    lmap: list[list[int]] = []
    for s in range(L_s):
        top = np.argsort(-scores[s])[:k]
        lmap.append([int(t) for t in sorted(top.tolist())])
    return lmap, {"score_matrix": scores.tolist()}


# ---------------------------------------------------------------------------
# Mapper helpers
# ---------------------------------------------------------------------------
def fit_mapper(kind: str, calib_t: KV, calib_s: KV, layer_map, lam: float = 1e-3):
    m = AffineMapper(lam=lam)
    pos_all = pos_of(calib_t)
    if kind == "K":
        m.fit(calib_t.k, calib_s.k, layer_map, kv_kind="K",
              positions=pos_all, de_rope_fn=de_rope_k)
    else:
        m.fit(calib_t.v, calib_s.v, layer_map, kv_kind="V",
              positions=pos_all, de_rope_fn=None)
    return m


def map_teacher(mk, mv, kv_t: KV, layer_map) -> KV:
    pos = pos_of(kv_t)
    km = mk.transform(kv_t.k, layer_map, kv_kind="K",
                      positions=pos, de_rope_fn=de_rope_k)
    vm = mv.transform(kv_t.v, layer_map, kv_kind="V",
                      positions=pos, de_rope_fn=None)
    return KV(k=km.astype(np.float32), v=vm.astype(np.float32))


def raw_map_teacher(kv_t: KV, layer_map, s_layers: int) -> KV:
    """No learned affine: select teacher layer(s) and average. Identity mapper control."""
    L_t, S, H, D = kv_t.k.shape
    k = np.zeros((s_layers, S, H, D), dtype=np.float32)
    v = np.zeros((s_layers, S, H, D), dtype=np.float32)
    for s in range(s_layers):
        ts = layer_map[s]
        k[s] = np.mean([kv_t.k[t] for t in ts], axis=0)
        v[s] = np.mean([kv_t.v[t] for t in ts], axis=0)
    return KV(k=k, v=v)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def query_of(sample: dict) -> str:
    return "\n\nQuestion: " + sample["q"] + "\nAnswer:"


def greedy_answer_fixed(model, tok, cache, query: str, max_new: int = 16) -> str:
    """Correct greedy decoding.

    Two legacy bugs are fixed relative to phase0_g0.greedy_answer:
      1. the cache is NOT shared with answer_loglik, so the prompt does not
         already contain the gold answer when generation starts;
      2. the first generated token comes from the prefill logits instead of
         re-feeding the last query token (double-feed).
    """
    q_ids = tok(query, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    with torch.no_grad():
        out = model(q_ids, past_key_values=cache, use_cache=True)
    past = out.past_key_values
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
        tok_id = int(torch.argmax(logits).item())
        gen.append(tok_id)
        next_in = torch.tensor([[tok_id]], device=model.device)
        if tok_id == tok.eos_token_id:
            break
    return tok.decode(gen, skip_special_tokens=True)


def score_arm(student, tok_s, kv: KV, sample: dict, want_em: bool = True,
              return_gen: bool = False):
    """Fresh cache for LL and for generation; corrected greedy.

    If return_gen, returns (ll, em, gen) with the generated text (I-4).
    """
    q = query_of(sample)
    ll = answer_loglik(student, tok_s, build_cache(kv), q, sample["answer"])
    em = None
    gen = None
    if want_em or return_gen:
        gen = greedy_answer_fixed(student, tok_s, build_cache(kv), q)
        em = exact_match(gen, sample["answer"])
    if return_gen:
        return ll, em, gen
    return ll, em


def bootstrap_ci95_clustered(values, groups, n_boot=10000, seed=0):
    """Cluster (document-level) bootstrap CI95 for the mean.

    The synthetic set repeats each document across ~7 questions, so per-sample
    resampling understates uncertainty. This resamples whole documents.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    buckets = [values[groups == g] for g in uniq]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(buckets), len(buckets))
        boots[b] = np.concatenate([buckets[j] for j in pick]).mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def doc_groups(samples: list) -> np.ndarray:
    docs = [s["doc"] for s in samples]
    uniq = {d: i for i, d in enumerate(dict.fromkeys(docs))}
    return np.array([uniq[d] for d in docs])


def summarize_rows(rows: list, keys: list, base: str, seed: int,
                   groups: np.ndarray | None = None) -> dict:
    """Mean LL, bootstrap CI95 of delta vs base, Wilcoxon, EM for each key."""
    base_lls = np.array([r[base] for r in rows], dtype=float)
    out = {}
    for key in keys:
        xs = np.array([r[key] for r in rows], dtype=float)
        mean, lo, hi = bootstrap_ci95(xs, seed=seed)
        dmean, dlo, dhi = bootstrap_ci95(xs - base_lls, seed=seed)
        _, p, d = paired_wilcoxon_test(xs.tolist(), base_lls.tolist())
        entry = {
            "LL_mean": float(mean), "LL_ci95": [float(lo), float(hi)],
            "delta_vs_self": float(dmean), "delta_ci95": [float(dlo), float(dhi)],
            "wilcoxon_p": float(p), "cohens_d": float(d),
            "EM": float(np.mean([r[key + "_em"] for r in rows]))
            if (key + "_em") in rows[0] else None,
        }
        if (key + "_em") in rows[0]:
            ems = np.array([float(r[key + "_em"]) for r in rows])
            emean, elo, ehi = bootstrap_ci95(ems, seed=seed)
            entry["EM_ci95"] = [float(elo), float(ehi)]
            if groups is not None:
                cm2, clo2, chi2 = bootstrap_ci95_clustered(ems, groups, seed=seed)
                entry["EM_ci95_clustered"] = [float(clo2), float(chi2)]
        if groups is not None:
            cm, clo, chi = bootstrap_ci95_clustered(xs, groups, seed=seed)
            dm, dlo2, dhi2 = bootstrap_ci95_clustered(xs - base_lls, groups, seed=seed)
            entry["LL_mean_clustered"] = cm
            entry["LL_ci95_clustered"] = [clo, chi]
            entry["delta_ci95_clustered"] = [dlo2, dhi2]
        out[key] = entry
    return out


def save(report: dict, output: str):
    import os
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[common] saved: {output}", flush=True)
