"""Regenerate paper figures from the corrected/innovation reports.

Outputs:
  fig_corrected_main.pdf : corrected 4-arm EM (6 pairs, 3 seeds) + adapter EM
  fig_controls.pdf       : content controls: LL gain vs corrected EM
"""
from __future__ import annotations
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(os.path.dirname(HERE))
REP = os.path.join(V3, "reports")

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

PAIRS = ["8B_0.6B", "4B_1.7B", "4B_0.6B", "8B_1.7B", "1.7B_0.6B", "8B_4B"]


def frange(a, b=None, n=6):
    return np.linspace(a, b, n) if n > 1 else np.array([a])


def load_fourarm():
    out = {}
    for s in (0, 1, 2):
        p = os.path.join(REP, f"phaseB_fourarm_seed{s}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for r in d["pair_results"]:
            out.setdefault(r["pair"], {"Self": [], "K-only": [], "V-only": [], "Joint": []})
            out[r["pair"]]["Self"].append(r["self_EM"])
            for a in ("K-only", "V-only", "Joint"):
                out[r["pair"]][a].append(r["summary"][a]["EM"])
    return out


def fig_main():
    fa = load_fourarm()
    pairs = [p for p in PAIRS if p in fa]
    arms = ["Self", "K-only", "V-only", "Joint"]
    colors = ["#444444", "#1f77b4", "#d62728", "#2ca02c"]
    x = np.arange(len(pairs))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), width_ratios=[1.5, 1])

    ax = axes[0]
    for i, (a, c) in enumerate(zip(arms, colors)):
        m = [np.mean(fa[p][a]) for p in pairs]
        e = [np.std(fa[p][a], ddof=1) for p in pairs]
        ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=2, color=c, label=a)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "→") for p in pairs], rotation=20)
    ax.set_ylabel("corrected exact match")
    ax.set_title("(a) Standard mapping: teacher states do not transfer")
    ax.legend(ncol=4, fontsize=7.5, frameon=False)
    ax.set_ylim(0, 1.05)

    # panel b: adapter (1.7B 3-seed mean, 8B seed0)
    ax = axes[1]
    conds = ["no_adapter", "adapter(self)", "adapter(joint)"]
    labels = ["no adapter", "adapter\n(student KV)", "adapter\n(teacher KV)"]
    x2 = np.arange(len(conds))
    w2 = 0.26
    for i, (a, c) in enumerate(zip(("K-only", "V-only", "Joint"),
                                   ("#1f77b4", "#d62728", "#2ca02c"))):
        b8 = [json.load(open(os.path.join(
            REP, "phaseB_adapter_8B_0.6B_o_proj_seed0.json")))["results"][cond][a]["EM"]
            for cond in conds]
        ax.bar(x2 + (i - 1) * w2, b8, w2, color=c, label=a)
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Consumption adapter (8B→0.6B)")
    ax.legend(fontsize=7.5, frameon=False)

    fig.savefig(os.path.join(HERE, "fig_corrected_main.pdf"))
    plt.close(fig)


def fig_controls():
    d = json.load(open(os.path.join(REP, "phaseB_controls_8B_0.6B_seed0_fixed.json")))
    s = d["summary"]
    order = ["K-only", "Wrong_K-only", "Rand_K", "Rand_KV", "Zero_KV", "Wrong_Joint", "Joint"]
    labels = ["K-only", "wrong doc", "random K", "random KV", "zero KV", "wrong-doc Joint", "Joint"]
    dll = [s[k]["delta_vs_self"] for k in order]
    em = [s[k]["EM"] for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    y = np.arange(len(order))
    axes[0].barh(y, dll, color="#1f77b4")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ΔLL vs Self")
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_title("(a) LL gain is content-agnostic")
    axes[1].barh(y, em, color="#d62728")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("corrected exact match")
    axes[1].set_title("(b) No arm transfers task performance")
    axes[1].set_xlim(0, 1.0)
    fig.savefig(os.path.join(HERE, "fig_controls.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_main()
    fig_controls()
    print("figures written")
