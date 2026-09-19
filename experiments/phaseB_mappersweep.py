"""A2: mapper objective sweep (alpha / lambda) (audit4 section 7, REVISION_PLAN4 II.A2).

`phaseB_errorbudget` ranks five mapper variants by three error metrics and finds
that consumption-space error (e_attn, e_WO) tracks V-only EM while raw error
does not. That is only five points. This script turns it into a sweep: for each
alpha the value mapper minimises

    (1-alpha) || M V_T - V_S ||^2  +  alpha || A_S M V_T - A_S V_S ||^2

implemented by stacking rows X = [sqrt(1-a) V_T ; sqrt(a) A_S V_T],
Y = [sqrt(1-a) V_S ; sqrt(a) A_S V_S] and solving the usual per-(layer, head)
centred ridge. alpha=0 is the affine map and alpha=1 the attention-output-aware
map, so both endpoints act as built-in self-consistency checks.

For every (alpha, lambda) configuration the script reports e_raw, e_attn, e_WO
and corrected V-only EM on the same run, then relates the three errors to EM by
Spearman correlation with a configuration-level bootstrap interval.

Pre-registered gate (REVISION_PLAN4 II.A2):
  * rho_attn, rho_WO <= -0.7 and rho_raw >= -0.3 with non-overlapping
    config-level bootstrap intervals -> keep the strong "consumption-space
    distance predicts transfer" statement
  * same direction but intervals overlap -> keep direction, phrase as "consistent with"
  * sign flips -> report the segment honestly and restrict the claim to alpha ends

Usage (single run):
  python3 experiments/phaseB_mappersweep.py --pair 1.7B_0.6B --seed 0 \
    --n-calib 70 --n-eval 56 --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
    --lams 1e-4,1e-3,1e-2

Pool the per-run reports afterwards:
  python3 experiments/phaseB_mappersweep.py --aggregate \
    'reports/phaseB_mappersweep_*.json'
"""
from __future__ import annotations

import argparse
import glob
import gc
import json
import os

import numpy as np
import torch

from phaseB_common import (
    PAIRS,
    HEAD_DIM,
    KV,
    build_cache,
    capture_all,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    query_of,
    save,
    score_arm,
    stack_kv,
)
from phaseB_errorbudget import apply_oa, error_budget, spearman
from phaseB_mechanism import get_attn_map
from stats_utils import bootstrap_spearman_ci

_ROOT = (os.environ.get("V3_ROOT")
         or ("/workspace/v3"
             if os.path.isdir(os.path.join("/workspace/v3", "experiments"))
             else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("V3_DATA_DIR", os.path.join(_ROOT, "data"))
REPORT_DIR = os.environ.get("V3_REPORT_DIR", os.path.join(_ROOT, "reports"))
MODELS_DIR = os.environ.get("V3_MODELS_DIR", "/root/.cache/modelscope/models")
APCS_DIR = os.environ.get("V3_APCS_DIR", "/workspace/apcs")


def fit_blended_mapper(calib_t, calib_s, attn_by_sample, layer_map,
                       alpha: float, lam: float = 1e-3):
    """Per-(layer, head) ridge on [sqrt(1-a) V_T ; sqrt(a) A_S V_T] -> [..V_S..].

    Row blocks are truncated to the common document length, exactly as
    `fit_output_aware_mapper`, so alpha=1 reproduces that function and alpha=0
    the affine V mapper.
    """
    W, b = {}, {}
    L_s = calib_s[0].v.shape[0]
    H = calib_s[0].v.shape[2]
    ra, ro = np.sqrt(1.0 - alpha), np.sqrt(alpha)
    for s in range(L_s):
        src = layer_map[s][0]
        Xs = {h: [] for h in range(H)}
        Ys = {h: [] for h in range(H)}
        for i in range(len(calib_t)):
            A = attn_by_sample[i][s].mean(axis=0)      # (S_q, n_doc)
            Vt = calib_t[i].v[src]
            Vs = calib_s[i].v[s]
            m = min(A.shape[1], Vt.shape[0], Vs.shape[0])
            A, Vt, Vs = A[:, :m], Vt[:m], Vs[:m]
            for h in range(H):
                vt, vs = Vt[:, h, :], Vs[:, h, :]
                if ra > 0.0 and ro > 0.0:
                    Xs[h].append(np.concatenate(
                        [ra * vt, ro * (A @ vt)], axis=0))
                    Ys[h].append(np.concatenate(
                        [ra * vs, ro * (A @ vs)], axis=0))
                elif ra > 0.0:
                    Xs[h].append(ra * vt)
                    Ys[h].append(ra * vs)
                else:
                    Xs[h].append(ro * (A @ vt))
                    Ys[h].append(ro * (A @ vs))
        for h in range(H):
            X = np.concatenate(Xs[h], axis=0).astype(np.float64)
            Y = np.concatenate(Ys[h], axis=0).astype(np.float64)
            Xm, Ym = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
            Xc, Yc = X - Xm, Y - Ym
            W[(s, h)] = np.linalg.solve(
                Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
            b[(s, h)] = (Ym - Xm @ W[(s, h)]).ravel()
    return W, b


def config_id(alpha: float, lam: float) -> str:
    return f"a{alpha:.2f}_l{lam:g}"


def bootstrap_config_rho(err: np.ndarray, em: np.ndarray, n_boot=10000,
                         seed=0) -> list:
    """Configuration-level bootstrap CI95 for Spearman(error, EM).

    Thin wrapper over :func:`stats_utils.bootstrap_spearman_ci` (average ranks);
    the name and call signature are unchanged so the A2 reports keep their
    provenance. See `paper/audit/METRIC_CORRECTION.md` section 14 for the W31
    change from ordinal to average ranks.
    """
    return bootstrap_spearman_ci(err, em, n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="1.7B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2")
    ap.add_argument("--output", default="")
    ap.add_argument("--aggregate", default="", help="glob of run reports to pool")
    a = ap.parse_args()

    if a.aggregate:
        aggregate(a.aggregate)
        return

    alphas = [round(float(x), 4) for x in a.alphas.split(",") if x.strip()]
    lams = [float(x) for x in a.lams.split(",") if x.strip()]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train = load_data(a.seed, "train")[: a.n_calib]
    test = load_data(a.seed, "test")[: a.n_eval]

    print("[A2] capture teacher then student ...", flush=True)
    teacher, tok_t, t_layers = load_teacher(a.pair)
    calib_t = capture_all(teacher, tok_t, train)
    eval_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(a.pair)
    calib_s = capture_all(student, tok_s, train)
    eval_s = capture_all(student, tok_s, test)

    lmap = layer_map_proportional(t_layers, s_layers)

    print("[A2] calibration attention maps ...", flush=True)
    attn_calib = [get_attn_map(student, tok_s,
                               build_cache(KV(k=calib_s[i].k, v=calib_s[i].v)),
                               query_of(s), calib_s[i].k.shape[1])
                  for i, s in enumerate(train)]
    print("[A2] evaluation attention maps (Self cache) ...", flush=True)
    attn_eval = [get_attn_map(student, tok_s,
                              build_cache(KV(k=eval_s[i].k, v=eval_s[i].v)),
                              query_of(s), eval_s[i].k.shape[1])
                 for i, s in enumerate(test)]

    D = int(getattr(student.config, "head_dim", None) or HEAD_DIM)
    n_kv = student.config.num_key_value_heads
    n_q = student.config.num_attention_heads
    group_of_head = {h: [h * (n_q // n_kv) + g for g in range(n_q // n_kv)]
                     for h in range(n_kv)}
    wo_slices = {}
    for l in range(s_layers):
        Wl = student.model.layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy()
        for q in range(n_q):
            wo_slices[(l, q)] = Wl[:, q * D:(q + 1) * D].T.astype(np.float64)

    self_ll = []
    for i, s in enumerate(test):
        ll, _ = score_arm(student, tok_s, KV(k=eval_s[i].k, v=eval_s[i].v), s,
                          want_em=False)
        self_ll.append(ll)
    self_ll = np.array(self_ll)

    configs = []
    for lam in lams:
        for alpha in alphas:
            key = config_id(alpha, lam)
            print(f"[A2] fit {key} ...", flush=True)
            W, b = fit_blended_mapper(calib_t, calib_s, attn_calib, lmap,
                                      alpha, lam)
            e_raw, e_attn, e_wo, ems = [], [], [], []
            for i, s in enumerate(test):
                v_hat = apply_oa(eval_t[i], lmap, W, b)
                stats = error_budget(v_hat, eval_s[i].v.astype(np.float32),
                                     attn_eval[i], wo_slices, group_of_head)
                _, em = score_arm(student, tok_s,
                                  KV(k=eval_s[i].k, v=v_hat), s)
                e_raw.append(stats["e_raw"])
                e_attn.append(stats["e_attn"])
                e_wo.append(stats["e_wo"])
                ems.append(float(em))
            configs.append({
                "alpha": alpha, "lam": lam, "key": key,
                "e_raw": float(np.mean(e_raw)), "e_attn": float(np.mean(e_attn)),
                "e_wo": float(np.mean(e_wo)), "EM": float(np.mean(ems)),
                "EM_per_sample": [float(x) for x in ems],
            })
            print(f"     e_raw={configs[-1]['e_raw']:.3f} "
                  f"e_attn={configs[-1]['e_attn']:.3f} "
                  f"e_wo={configs[-1]['e_wo']:.3f} EM={configs[-1]['EM']:.3f}",
                  flush=True)
            del W, b
            gc.collect()
            torch.cuda.empty_cache()

    em_vec = np.array([c["EM"] for c in configs])
    rho = {}
    ci = {}
    for name in ("e_raw", "e_attn", "e_wo"):
        err = np.array([c[name] for c in configs])
        rho[name] = spearman(err, em_vec)
        ci[name] = bootstrap_config_rho(err, em_vec, seed=a.seed)

    gate = {
        "rho_raw": rho["e_raw"], "rho_attn": rho["e_attn"], "rho_wo": rho["e_wo"],
        "rho_ci95": ci,
        "attn_wo_strong": bool(rho["e_attn"] is not None
                               and rho["e_attn"] <= -0.7
                               and rho["e_wo"] is not None
                               and rho["e_wo"] <= -0.7
                               and rho["e_raw"] is not None
                               and rho["e_raw"] >= -0.3),
        "rho_raw_upper": ci["e_raw"][1],
        "n_configs": len(configs),
    }
    if gate["attn_wo_strong"]:
        ci_raw, ci_attn = ci["e_raw"], ci["e_attn"]
        disjoint = (ci_raw[0] is not None and ci_attn[0] is not None
                    and (ci_raw[1] < ci_attn[0] or ci_raw[0] > ci_attn[1]))
    else:
        disjoint = False
    if gate["attn_wo_strong"] and disjoint:
        gate["branch"] = "strong statement preserved"
    elif (rho["e_attn"] is not None and rho["e_attn"] <= -0.7
          and rho["e_wo"] is not None and rho["e_wo"] <= -0.7):
        gate["branch"] = "direction preserved, phrase as consistent with"
    else:
        gate["branch"] = "sign flip / weak: restrict claim to alpha ends"
    gate["bootstrap_intervals_disjoint"] = bool(disjoint)

    report = {
        "task": "phaseB_mappersweep",
        "pair": a.pair, "seed": a.seed,
        "n_calib": a.n_calib, "n_eval": a.n_eval,
        "alphas": alphas, "lams": lams,
        "self_LL_mean": float(self_ll.mean()),
        "configs": configs,
        "rho": rho, "rho_ci95": ci,
        "gate": gate,
    }
    out = a.output or (f"{REPORT_DIR}/phaseB_mappersweep_{a.pair}"
                       f"_seed{a.seed}.json")
    save(report, out)
    print("\n=== A2 mapper sweep (pair=%s seed=%d) ===" % (a.pair, a.seed))
    print(f"  rho_raw={rho['e_raw']} CI={ci['e_raw']}")
    print(f"  rho_attn={rho['e_attn']} CI={ci['e_attn']}")
    print(f"  rho_wo={rho['e_wo']} CI={ci['e_wo']}")
    print(f"  gate: {gate}")


def aggregate(pattern: str):
    """Pool per-run config points and recompute correlations across all runs."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[A2] no reports match {pattern}")
        return
    all_err = {k: [] for k in ("e_raw", "e_attn", "e_wo")}
    all_em, labels = [], []
    for f in files:
        d = json.load(open(f))
        run_id = f"{d['pair']}_s{d['seed']}"
        for c in d["configs"]:
            for k in all_err:
                all_err[k].append(c[k])
            all_em.append(c["EM"])
            labels.append((run_id, c["key"]))
    em_vec = np.array(all_em)
    rho, ci = {}, {}
    for k in all_err:
        err = np.array(all_err[k])
        rho[k] = spearman(err, em_vec)
        ci[k] = bootstrap_config_rho(err, em_vec, seed=0)
    out = {"task": "phaseB_mappersweep_aggregate", "n_points": len(all_em),
           "n_runs": len(files), "rho_pooled": rho, "rho_ci95": ci,
           "running": [f"{p}/{k}" for p, k in labels]}
    path = (f"{REPORT_DIR}/phaseB_mappersweep_aggregate.json")
    save(out, path)
    print(json.dumps({"rho_pooled": rho, "rho_ci95": ci}, indent=2))


if __name__ == "__main__":
    main()
