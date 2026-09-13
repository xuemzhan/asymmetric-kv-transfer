"""Innovation: consumption adapter for cross-model KV transfer.

Motivation
----------
The corrected-evaluation results show that mapped teacher K/V are not consumed
by the student: greedy exact match collapses even though teacher-forced LL rises,
and a shuffled-target mapper does not help. B3/outaware show the bottleneck is
the *consumer* (attention output / o_proj), not the state alone.

This script learns a tiny low-rank correction on the student's `o_proj`
    o_proj'(x) = W_O x + B (A x)
at every layer (rank r), with the student and the state mapper frozen, trained
only on the synthetic calibration split with injected mapped teacher KV. The
adapter is then evaluated on held-out test samples.

Conditions (all use the corrected evaluator):
  - no-adapter Self / K-only / V-only / Joint
  - adapter trained on Joint teacher KV      -> test Self / Joint
  - adapter trained on Self (student) KV     -> test Joint   (task-adapter control)
  - adapter trained on shuffled-doc Joint KV -> test Joint   (content control)

If the Joint-trained adapter recovers test EM for teacher KV but the controls do
not, cross-model transfer is recoverable by adapting the consumer, not the state.

Usage:
  python3 phaseB_adapter.py --pair 1.7B_0.6B --seed 0 --rank 8 --epochs 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from phaseB_common import (
    PAIRS,
    KV,
    build_cache,
    capture_all,
    doc_groups,
    fit_mapper,
    layer_map_proportional,
    load_data,
    load_student,
    load_teacher,
    map_teacher,
    score_arm,
    stack_kv,
)


def _causal_extra_arms(test, eval_s, mapped_eval, rng):
    """Wrong-document teacher KV, moment-matched random KV, and zero KV."""
    from phaseB_controls import wrong_doc_partner, rand_like_from_stats
    wrong_idx = wrong_doc_partner(test, rng)
    wrong = [KV(k=mapped_eval[wrong_idx[i]].k, v=mapped_eval[wrong_idx[i]].v)
             for i in range(len(test))]
    rand = [rand_like_from_stats(eval_s[i], rng) for i in range(len(test))]
    zero = [KV(k=np.zeros_like(eval_s[i].k), v=np.zeros_like(eval_s[i].v))
            for i in range(len(test))]
    return {"WrongJoint": wrong, "RandKV": rand, "ZeroKV": zero}


class AdaptedLinear(nn.Module):
    """base is frozen; adds a zero-initialised low-rank correction."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.normal_(self.A, std=1e-3)

    def forward(self, x):
        out = self.base(x)
        corr = (x.float() @ self.A.T) @ self.B.T
        return out + corr.to(out.dtype)


def install_adapters(model, rank: int, targets=("o_proj",)):
    mods = []
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in targets:
            base = getattr(attn, name)
            ad = AdaptedLinear(base, rank).to(base.weight.device)
            setattr(attn, name, ad)
            mods.append((name, ad))
    return mods


def remove_adapters(model, targets=("o_proj",)):
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in targets:
            setattr(attn, name, getattr(attn, name).base)


def train_adapter(model, tok, samples, doc_states, epochs, lr, seed, adapters,
                  tag=""):
    params = [p for _, a in adapters for p in (a.A, a.B)]
    opt = torch.optim.Adam(params, lr=lr)
    n = len(samples)
    for ep in range(epochs):
        order = np.random.RandomState(seed + ep).permutation(n)
        tot = 0.0
        for j in order:
            s = samples[j]
            q_ids = tok("\n\nQuestion: " + s["q"] + "\nAnswer:",
                        return_tensors="pt", add_special_tokens=False).input_ids
            a_ids = tok(s["answer"], return_tensors="pt",
                        add_special_tokens=False).input_ids
            q_ids = q_ids.to(model.device)
            a_ids = a_ids.to(model.device)
            ids = torch.cat([q_ids, a_ids], dim=1)
            # fresh cache each step: DynamicCache is mutated by the forward
            cache = build_cache(doc_states[j])
            out = model(ids, past_key_values=cache, use_cache=True)
            logits = out.logits[0]
            Lq, La = q_ids.shape[1], a_ids.shape[1]
            pred = logits[Lq - 1: Lq + La - 1]
            loss = torch.nn.functional.cross_entropy(pred, a_ids[0])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss)
        print(f"  [{tag}] epoch {ep+1}/{epochs} loss={tot/n:.3f}", flush=True)


def eval_arms(model, tok, eval_s, mapped, samples, extra_arms=None):
    """extra_arms: {name: list[KV]} evaluated alongside the four standard arms."""
    extra_arms = extra_arms or {}
    rows = []
    for i, s in enumerate(samples):
        row = {"id": s["id"]}
        arms = {
            "Self": KV(k=eval_s[i].k, v=eval_s[i].v),
            "K-only": KV(k=mapped[i].k, v=eval_s[i].v),
            "V-only": KV(k=eval_s[i].k, v=mapped[i].v),
            "Joint": KV(k=mapped[i].k, v=mapped[i].v),
        }
        for name, kvs in extra_arms.items():
            arms[name] = kvs[i]
        for name, kv in arms.items():
            ll, em = score_arm(model, tok, kv, s)
            row[name] = ll
            row[name + "_em"] = float(em)
        rows.append(row)
    return rows


def summarize(rows, tag):
    out = {}
    names = [k for k in rows[0] if k != "id" and not k.endswith("_em")]
    for name in names:
        out[name] = {
            "EM": float(np.mean([r[name + "_em"] for r in rows])),
            "LL": float(np.mean([r[name] for r in rows])),
        }
    print(f"  {tag:28s} " + " ".join(
        f"{k}:EM={v['EM']:.3f}" for k, v in out.items()))
    return out


def run(pair, seed, rank, epochs, lr, n_calib, n_eval, output, targets,
        causal=False, conditions=("joint", "self", "shuffled")):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = load_data(seed, "train")[:n_calib]
    test = load_data(seed, "test")[:n_eval]
    teacher, tok_t, t_layers = load_teacher(pair)
    calib_t = capture_all(teacher, tok_t, train)
    eval_t = capture_all(teacher, tok_t, test)
    del teacher
    torch.cuda.empty_cache()
    student, tok_s, s_layers = load_student(pair)
    calib_s = capture_all(student, tok_s, train)
    eval_s = capture_all(student, tok_s, test)

    lmap = layer_map_proportional(t_layers, s_layers)
    ct, cs = stack_kv(calib_t), stack_kv(calib_s)
    mk = fit_mapper("K", ct, cs, lmap)
    mv = fit_mapper("V", ct, cs, lmap)
    mapped_calib = [map_teacher(mk, mv, calib_t[i], lmap) for i in range(len(train))]
    mapped_eval = [map_teacher(mk, mv, eval_t[i], lmap) for i in range(len(test))]

    def states_for(kind):
        if kind == "Self":
            return [KV(k=calib_s[i].k, v=calib_s[i].v) for i in range(len(train))]
        if kind == "Joint":
            return [KV(k=mapped_calib[i].k, v=mapped_calib[i].v)
                    for i in range(len(train))]
        if kind == "ShuffledJoint":
            groups = doc_groups(train)
            uniq = np.unique(groups)
            gmap = {g: uniq[(j + 1) % len(uniq)] for j, g in enumerate(uniq)}
            targets = []
            for i, g in enumerate(groups):
                members = np.where(groups == gmap[g])[0]
                targets.append(int(members[i % len(members)]))
            return [KV(k=mapped_calib[j].k, v=mapped_calib[j].v) for j in targets]
        raise ValueError(kind)

    results = {}
    # baseline (no adapter)
    base_rows = eval_arms(student, tok_s, eval_s, mapped_eval, test)
    results["no_adapter"] = summarize(base_rows, "no-adapter")

    causal_extra = None
    if causal:
        rng = np.random.RandomState(seed + 999)
        causal_extra = _causal_extra_arms(test, eval_s, mapped_eval, rng)

    all_conditions = [("Joint", "adapter(joint)", "joint"),
                      ("Self", "adapter(self)", "self"),
                      ("ShuffledJoint", "adapter(shuffled)", "shuffled")]
    for kind, tag, cname in all_conditions:
        if cname not in conditions:
            continue
        adapters = install_adapters(student, rank, targets)
        train_adapter(student, tok_s, train, states_for(kind), epochs, lr, seed,
                      adapters, tag=tag)
        extra = causal_extra if (causal and kind == "Joint") else None
        rows = eval_arms(student, tok_s, eval_s, mapped_eval, test,
                         extra_arms=extra)
        results[tag] = summarize(rows, tag)
        if extra is not None:
            results["causal"] = summarize(rows, "causal(joint)")
        remove_adapters(student, targets)

    report = {"task": "phaseB_adapter", "pair": pair, "seed": seed,
              "rank": rank, "epochs": epochs, "lr": lr, "targets": list(targets),
              "n_calib": n_calib, "n_eval": n_eval, "results": results}
    with open(output, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[adapter] saved: {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="1.7B_0.6B", choices=list(PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-calib", type=int, default=70)
    ap.add_argument("--n-eval", type=int, default=56)
    ap.add_argument("--targets", default="o_proj",
                    help="comma-separated modules: q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--causal", action="store_true",
                    help="also evaluate the fixed adapter on wrong/random/zero KV")
    ap.add_argument("--conditions", default="joint,self,shuffled",
                    help="which adapter training conditions to run")
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    targets = tuple(t.strip() for t in a.targets.split(",") if t.strip())
    tag = "_".join(targets)
    out = a.output or f"/workspace/v3/reports/phaseB_adapter_{a.pair}_{tag}_seed{a.seed}.json"
    conds = tuple(c.strip() for c in a.conditions.split(",") if c.strip())
    run(a.pair, a.seed, a.rank, a.epochs, a.lr, a.n_calib, a.n_eval, out, targets,
        causal=a.causal, conditions=conds)


if __name__ == "__main__":
    main()
