"""Generate publication-quality figures for V3 paper.

Produces three vector PDFs:
  1. fig_cost.pdf      -- state/weight transfer cost crossover vs sequence length
  2. fig_layers.pdf    -- per-layer K-only vs V-only Delta-LL profile (3-seed mean)
  3. fig_cca.pdf       -- per-head CCA rho_K vs rho_V scatter (correlation != transferability)

Data sources (all in ../reports/):
  - phase1_causal_v2_seed{0,1,2}.json  -> layer_results (28 layers)
  - w4_cca_perhead.json                 -> per-head rho1_K / rho1_V per pair
  - mapper/cache cost constants         -> from v3_data_baseline.md sec 6
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
FIG = os.path.dirname(os.path.abspath(__file__))

# Publication style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colorblind-safe palette
BLUE = "#377eb8"
ORANGE = "#ff7f00"
GREEN = "#4daf4a"
RED = "#e41a1c"
PURPLE = "#984ea3"
BROWN = "#a65628"


def load_causal_layers():
    """Return dict: layer -> {K: mean dLL, V: mean dLL} averaged over 3 seeds."""
    acc = {}
    for s in (0, 1, 2):
        p = os.path.join(REPORTS, f"phase1_causal_v2_seed{s}.json")
        with open(p) as f:
            d = json.load(f)
        for row in d["layer_results"]:
            l = row["student_layer"]
            acc.setdefault(l, {"K": [], "V": []})
            acc[l]["K"].append(row["K_only_LL_delta"])
            acc[l]["V"].append(row["V_only_LL_delta"])
    layers = sorted(acc)
    K = [np.mean(acc[l]["K"]) for l in layers]
    V = [np.mean(acc[l]["V"]) for l in layers]
    return np.array(layers), np.array(K), np.array(V)


def load_cca_heads():
    """Return flattened per-head (rho_K, rho_V) for each pair."""
    out = {}
    with open(os.path.join(REPORTS, "w4_cca_perhead.json")) as f:
        d = json.load(f)
    for pair, r in d["results"].items():
        rK, rV = [], []
        for layer in r["layers"]:
            rK.extend(layer["rho1_K_per_head"])
            rV.extend(layer["rho1_V_per_head"])
        out[pair] = (np.array(rK), np.array(rV))
    return out


# ---------------------------------------------------------------------------
# Figure 1: Cost crossover
# ---------------------------------------------------------------------------
def fig_cost():
    mapper_bytes = 58.78e6 * 4          # 235.12 MB (fp32)
    cache_per_token = 112 * 1024        # 112 KiB per token (28-layer student)
    crossover = mapper_bytes / cache_per_token  # ~2050

    seq = np.linspace(0, 6000, 300)
    cache_bytes = cache_per_token * seq
    mapper_line = np.full_like(seq, mapper_bytes)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(seq, cache_bytes / 1e6, color=BLUE, label="Cache transfer (state)")
    ax.plot(seq, mapper_line / 1e6, color=ORANGE, label="Mapper transfer (weights)")

    ax.axvline(crossover, color=RED, linestyle="--", linewidth=1.0)
    ax.annotate(
        f"crossover\n$\\approx${int(round(crossover))} tokens",
        xy=(crossover, 0), xytext=(crossover + 500, 40),
        fontsize=8, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
    )

    ax.fill_between(seq, 0, cache_bytes / 1e6, where=(seq <= crossover),
                    color=BLUE, alpha=0.08)
    ax.fill_between(seq, 0, mapper_line / 1e6, where=(seq >= crossover),
                    color=ORANGE, alpha=0.08)

    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Transfer cost (MB)")
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 750)
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(FIG, "fig_cost.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Layer localization profile
# ---------------------------------------------------------------------------
def fig_layers():
    layers, K, V = load_causal_layers()

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.38
    x = np.arange(len(layers))
    ax.bar(x - w / 2, K, width=w, color=BLUE, label="K-only $\\Delta$LL")
    ax.bar(x + w / 2, V, width=w, color=ORANGE, label="V-only $\\Delta$LL")

    # Highlight V hotspots L8 / L12
    for hl in (8, 12):
        ax.bar(hl - w / 2, K[hl], width=w, color=BLUE, edgecolor=RED, linewidth=1.0)
        ax.bar(hl + w / 2, V[hl], width=w, color=ORANGE, edgecolor=RED, linewidth=1.0)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Student layer")
    ax.set_ylabel("$\\Delta$LL vs. Self")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], rotation=90, fontsize=6)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(FIG, "fig_layers.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: CCA scatter (correlation != transferability)
# ---------------------------------------------------------------------------
def fig_cca():
    cca = load_cca_heads()
    pair_colors = {"8B_0.6B": BLUE, "4B_1.7B": GREEN, "4B_0.6B": ORANGE,
                   "8B_1.7B": RED, "1.7B_0.6B": PURPLE, "8B_4B": BROWN}
    pair_labels = {"8B_0.6B": "8B$\\to$0.6B", "4B_1.7B": "4B$\\to$1.7B",
                   "4B_0.6B": "4B$\\to$0.6B", "8B_1.7B": "8B$\\to$1.7B",
                   "1.7B_0.6B": "1.7B$\\to$0.6B", "8B_4B": "8B$\\to$4B"}

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for pair, (rK, rV) in cca.items():
        ax.scatter(rK, rV, s=6, alpha=0.5, color=pair_colors[pair],
                   label=pair_labels[pair], edgecolors="none", rasterized=True)

    ax.plot([0.97, 1.0], [0.97, 1.0], color="black", linestyle="--", linewidth=0.8,
            label="$\\rho_K = \\rho_V$")

    ax.set_xlabel("CCA $\\rho_1$ (K)")
    ax.set_ylabel("CCA $\\rho_1$ (V)")
    ax.set_xlim(0.975, 1.0005)
    ax.set_ylim(0.975, 1.0005)
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(FIG, "fig_cca.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_cost()
    fig_layers()
    fig_cca()
    print("Generated:")
    for f in ("fig_cost.pdf", "fig_layers.pdf", "fig_cca.pdf"):
        p = os.path.join(FIG, f)
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")
