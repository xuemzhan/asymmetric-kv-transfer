"""Generate publication figures for the V3 paper.

Outputs five vector PDFs in figures4papers style:
  1. fig_main.pdf  -- six-pair K/V asymmetry and heterogeneous reassembly
  2. fig_cca.pdf   -- CCA geometric alignment vs. functional transfer
  3. fig_layers.pdf-- per-layer K/V profile and selective V injection
  4. fig_squad.pdf -- SQuAD cross-domain replication
  5. fig_cost.pdf  -- state/weight transfer cost crossover

Every figure is designed to be readable without the main text: it carries its
own pair names, metric labels, uncertainty, and significance markers.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIG = ROOT / "paper" / "figures"


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


PAIR_ORDER = [
    "8B_0.6B",
    "4B_0.6B",
    "1.7B_0.6B",
    "8B_4B",
    "4B_1.7B",
    "8B_1.7B",
]
PAIR_LABELS = {
    "8B_0.6B": "8B\u21920.6B",
    "4B_0.6B": "4B\u21920.6B",
    "1.7B_0.6B": "1.7B\u21920.6B",
    "8B_4B": "8B\u21924B",
    "4B_1.7B": "4B\u21921.7B",
    "8B_1.7B": "8B\u21921.7B",
}


def finalize(fig, name):
    path = FIG / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  {path} ({path.stat().st_size:,} bytes)")


def mean_std(values):
    vals = np.asarray(values, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def load_scaling():
    out = {}
    for pair in PAIR_ORDER:
        k, v, em_self, em_k, em_v, k_sig, v_sig = [], [], [], [], [], [], []
        for seed in ("0", "1", "2"):
            with open(REPORTS / f"phase4_scaling_law_v2_seed{seed}.json") as f:
                data = json.load(f)
            row = next(r for r in data["pair_results"] if r["pair"] == pair)
            k.append(row["wilcoxon_vs_self"]["K-only"]["delta_mean"])
            v.append(row["wilcoxon_vs_self"]["V-only"]["delta_mean"])
            em_self.append(row["exact_match"]["Self"])
            em_k.append(row["exact_match"]["K-only"])
            em_v.append(row["exact_match"]["V-only"])
            k_sig.append(row["wilcoxon_vs_self"]["K-only"]["p_value"] < 0.05)
            v_sig.append(row["wilcoxon_vs_self"]["V-only"]["p_value"] < 0.05)
        out[pair] = {
            "k": mean_std(k),
            "v": mean_std(v),
            "em_self": float(np.mean(em_self)),
            "em_k": float(np.mean(em_k)),
            "em_v": float(np.mean(em_v)),
            "k_sig": all(k_sig),
            "v_sig": all(v_sig),
        }
    return out


def load_reassembly():
    with open(REPORTS / "g0_v2_summary.json") as f:
        data = json.load(f)
    seeds = ("0", "1", "2")
    teacher = np.mean([data["summary_ll"]["teacher_full"][s]["mean"] for s in seeds])
    student = np.mean([data["summary_ll"]["student_full"][s]["mean"] for s in seeds])
    konly = np.mean([data["summary_ll"]["Affine_K-only"][s]["mean"] for s in seeds])
    em_k = float(np.mean([data["exact_match"]["Affine_K-only"][s] for s in seeds]))
    em_student = float(np.mean([data["exact_match"]["student_full"][s] for s in seeds]))
    em_teacher = float(np.mean([data["exact_match"]["teacher_full"][s] for s in seeds]))
    return teacher, student, konly, em_teacher, em_student, em_k


def load_layers():
    k, v = {}, {}
    for seed in ("0", "1", "2"):
        with open(REPORTS / f"phase1_causal_v2_seed{seed}.json") as f:
            data = json.load(f)
        for row in data["layer_results"]:
            layer = row["student_layer"]
            k.setdefault(layer, []).append(row["K_only_LL_delta"])
            v.setdefault(layer, []).append(row["V_only_LL_delta"])
    layers = sorted(k)
    k_mean = [np.mean(k[l]) for l in layers]
    v_mean = [np.mean(v[l]) for l in layers]

    sel = {"L8": [], "L12": [], "L8+L12": [], "ALL": []}
    for seed in ("0", "1", "2"):
        with open(REPORTS / f"phase1_causal_v2_seed{seed}.json") as f:
            data = json.load(f)
        sel["L8"].append(data["selective_V_injection"]["V_8"]["LL_delta_vs_Self"])
        sel["L12"].append(data["selective_V_injection"]["V_12"]["LL_delta_vs_Self"])
        sel["L8+L12"].append(data["selective_V_injection"]["V_L8_L12"]["LL_delta_vs_Self"])
        sel["ALL"].append(data["selective_V_injection"]["V_ALL"]["LL_delta_vs_Self"])
    return layers, k_mean, v_mean, {key: np.mean(val) for key, val in sel.items()}


def load_cca():
    with open(REPORTS / "w4_cca_perhead.json") as f:
        data = json.load(f)
    heads = {}
    means = {}
    for pair in PAIR_ORDER:
        rows = data["results"][pair]["layers"]
        rk = [x for row in rows for x in row["rho1_K_per_head"]]
        rv = [x for row in rows for x in row["rho1_V_per_head"]]
        heads[pair] = (np.array(rk), np.array(rv))
        means[pair] = (
            float(np.mean(rk)),
            float(np.mean(rv)),
            float(data["results"][pair]["fraction_heads_K_gt_V"]),
        )
    return heads, means


def load_squad():
    arms = ["K-only", "V-only", "Joint"]
    deltas, em = {a: [] for a in arms}, {a: [] for a in arms}
    self_em = []
    for seed in ("0", "1", "2"):
        with open(REPORTS / f"phase7_second_domain_seed{seed}.json") as f:
            data = json.load(f)
        self_em.append(data["results_4way"]["Self"]["EM"])
        for a in arms:
            deltas[a].append(data["stats_vs_self"][a]["delta_mean"])
            em[a].append(data["results_4way"][a]["EM"])
    delta_stats = {a: mean_std(deltas[a]) for a in arms}
    em_stats = {a: float(np.mean(em[a])) for a in arms}
    return delta_stats, em_stats, float(np.mean(self_em))


def add_significance(ax, x, y, yerr, sig, ytop):
    for xi, yi, ei, si in zip(x, y, yerr, sig):
        ax.text(xi, yi + ei + 0.16, "*" if si else "n.s.", ha="center",
                va="bottom", fontsize=13, color="#222222")
    ax.set_ylim(top=ytop)


# ---------------------------------------------------------------------------
# Figure 1: main asymmetry
# ---------------------------------------------------------------------------
def fig_main():
    scaling = load_scaling()
    teacher, student, konly, em_teacher, em_student, em_k = load_reassembly()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6),
                             gridspec_kw={"width_ratios": [1.55, 1.45, 0.82]})
    labels = [PAIR_LABELS[p] for p in PAIR_ORDER]
    x = np.arange(len(labels))
    w = 0.38

    ax = axes[0]
    k_mean = np.array([scaling[p]["k"][0] for p in PAIR_ORDER])
    k_err = np.array([scaling[p]["k"][1] for p in PAIR_ORDER])
    v_mean = np.array([scaling[p]["v"][0] for p in PAIR_ORDER])
    v_err = np.array([scaling[p]["v"][1] for p in PAIR_ORDER])
    ax.bar(x - w / 2, k_mean, w, yerr=k_err, capsize=3,
           color=PALETTE["blue_main"], label="K-only",
           edgecolor="black", linewidth=1.0)
    ax.bar(x + w / 2, v_mean, w, yerr=v_err, capsize=3,
           color=PALETTE["red_strong"], label="V-only",
           edgecolor="black", linewidth=1.0)
    add_significance(ax, x - w / 2, k_mean, k_err,
                     [scaling[p]["k_sig"] for p in PAIR_ORDER], 9.8)
    add_significance(ax, x + w / 2, v_mean, v_err,
                     [scaling[p]["v_sig"] for p in PAIR_ORDER], 9.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("\u0394 answer log-likelihood")
    ax.set_title("(a) Six-pair K/V asymmetry")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    s = [scaling[p]["em_self"] for p in PAIR_ORDER]
    k = [scaling[p]["em_k"] for p in PAIR_ORDER]
    v = [scaling[p]["em_v"] for p in PAIR_ORDER]
    ax.bar(x - 0.27, s, 0.24, color=PALETTE["neutral"], label="Self",
           edgecolor="black", linewidth=0.9)
    ax.bar(x, k, 0.24, color=PALETTE["blue_main"], label="K-only",
           edgecolor="black", linewidth=0.9)
    ax.bar(x + 0.27, v, 0.24, color=PALETTE["red_strong"], label="V-only",
           edgecolor="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Exact match")
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) Greedy exact match")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    configs = ["Teacher\nfull", "Student\nfull", "K-only\n(8B K + 0.6B V)"]
    values = [teacher, student, konly]
    colors = [PALETTE["neutral"], PALETTE["neutral"], PALETTE["blue_main"]]
    bars = ax.bar(configs, values, color=colors, width=0.56,
                  edgecolor="black", linewidth=1.0)
    ems = [f"EM {em_teacher:.2f}", f"EM {em_student:.2f}", f"EM {em_k:.2f}"]
    for b, val, em in zip(bars, values, ems):
        ax.text(b.get_x() + b.get_width() / 2, val - 1.4,
                f"{val:.2f}\n{em}", ha="center", va="center", fontsize=12,
                color="white", fontweight="bold")
    ax.set_ylabel("Answer log-likelihood")
    ax.set_title("(c) Reassembly\n8B\u21920.6B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)
    finalize(fig, "fig_main")


# ---------------------------------------------------------------------------
# Figure 2: CCA
# ---------------------------------------------------------------------------
def fig_cca():
    heads, means = load_cca()
    colors = {
        "8B_0.6B": PALETTE["blue_main"],
        "4B_0.6B": PALETTE["red_strong"],
        "1.7B_0.6B": PALETTE["green_3"],
        "8B_4B": PALETTE["violet"],
        "4B_1.7B": PALETTE["teal"],
        "8B_1.7B": "#E8A33D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    for pair in PAIR_ORDER:
        rk, rv = heads[pair]
        ax.scatter(rk, rv, s=7, alpha=0.45, color=colors[pair],
                   label=PAIR_LABELS[pair], edgecolors="none", rasterized=True)
    ax.plot([0.96, 1.0], [0.96, 1.0], "--", color="black", linewidth=1.0)
    ax.set_xlabel("CCA \u03c1\u2081 (K)")
    ax.set_ylabel("CCA \u03c1\u2081 (V)")
    ax.set_xlim(0.96, 1.001)
    ax.set_ylim(0.96, 1.001)
    ax.set_title("(a) Per-head alignment")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = np.arange(len(PAIR_ORDER))
    rk = np.array([means[p][0] for p in PAIR_ORDER])
    rv = np.array([means[p][1] for p in PAIR_ORDER])
    frac = [means[p][2] * 100 for p in PAIR_ORDER]
    ax.bar(x - 0.19, rk, 0.34, color=PALETTE["blue_main"], label="\u03c1(K)",
           edgecolor="black", linewidth=0.8)
    ax.bar(x + 0.19, rv, 0.34, color=PALETTE["red_strong"], label="\u03c1(V)",
           edgecolor="black", linewidth=0.8)
    for xi, f in zip(x, frac):
        ax.text(xi, 0.9855, f"{f:.1f}%", ha="center", va="bottom",
                fontsize=10, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABELS[p] for p in PAIR_ORDER], rotation=30, ha="right")
    ax.set_ylim(0.98, 1.0)
    ax.set_ylabel("Mean CCA \u03c1\u2081")
    ax.set_title("(b) Heads with \u03c1(K)>\u03c1(V)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)
    finalize(fig, "fig_cca")


# ---------------------------------------------------------------------------
# Figure 3: layer localization
# ---------------------------------------------------------------------------
def fig_layers():
    layers, k, v, sel = load_layers()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    x = np.arange(len(layers))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, k, w, color=PALETTE["blue_main"], label="K-only",
           edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, v, w, color=PALETTE["red_strong"], label="V-only",
           edgecolor="black", linewidth=0.8)
    for hl in (8, 12):
        ax.bar(hl - w / 2, k[hl], w, color=PALETTE["blue_main"],
               edgecolor=PALETTE["highlight"], linewidth=2.0)
        ax.bar(hl + w / 2, v[hl], w, color=PALETTE["red_strong"],
               edgecolor=PALETTE["highlight"], linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=9, rotation=90)
    ax.set_xlabel("Student layer")
    ax.set_ylabel("\u0394 answer log-likelihood")
    ax.set_title("(a) Per-layer injection, 8B\u21920.6B")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    labels = ["V at L8", "V at L12", "V at L8+L12", "V at all 28 layers"]
    vals = [sel["L8"], sel["L12"], sel["L8+L12"], sel["ALL"]]
    colors = [PALETTE["red_strong"], PALETTE["red_strong"],
              PALETTE["violet"], PALETTE["neutral"]]
    bars = ax.bar(labels, vals, color=colors, width=0.62,
                  edgecolor="black", linewidth=1.0)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.04,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("\u0394 answer log-likelihood")
    ax.set_title("(b) Selective V injection")
    ax.set_ylim(-0.55, 2.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)
    finalize(fig, "fig_layers")


# ---------------------------------------------------------------------------
# Figure 4: SQuAD cross-domain replication
# ---------------------------------------------------------------------------
def fig_squad():
    delta, em, self_em = load_squad()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    arms = ["K-only", "V-only", "Joint"]
    labels = arms
    x = np.arange(len(arms))

    ax = axes[0]
    means = [delta[a][0] for a in arms]
    errs = [delta[a][1] for a in arms]
    colors = [PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"]]
    bars = ax.bar(x, means, yerr=errs, capsize=4, color=colors, width=0.56,
                  edgecolor="black", linewidth=1.0)
    ps = {"K-only": "p<3\u00d710\u207b\u2076", "V-only": "n.s.", "Joint": "p<4\u00d710\u207b\u2074"}
    for b, a in zip(bars, arms):
        ax.text(b.get_x() + b.get_width() / 2, delta[a][0] + delta[a][1] + 0.22,
                ps[a], ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("\u0394 answer log-likelihood")
    ax.set_title("(a) OOD\u2192SQuAD transfer")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    self = [self_em]
    k = [em["K-only"]]
    v = [em["V-only"]]
    j = [em["Joint"]]
    ax.bar([0], self, 0.45, color=PALETTE["neutral"], label="Self")
    ax.bar([1], k, 0.45, color=PALETTE["blue_main"], label="K-only")
    ax.bar([2], v, 0.45, color=PALETTE["red_strong"], label="V-only")
    ax.bar([3], j, 0.45, color=PALETTE["teal"], label="Joint")
    for xi, val in zip(range(4), [self[0], k[0], v[0], j[0]]):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                fontsize=12)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Self", "K-only", "V-only", "Joint"])
    ax.set_ylabel("Exact match")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Greedy exact match")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.7)
    finalize(fig, "fig_squad")


# ---------------------------------------------------------------------------
# Figure 5: cost crossover
# ---------------------------------------------------------------------------
def fig_cost():
    mapper_bytes = 58.78e6 * 4
    cache_per_token = 112 * 1024
    crossover = mapper_bytes / cache_per_token

    seq = np.linspace(0, 6000, 400)
    cache = cache_per_token * seq / 1e6
    mapper = np.full_like(seq, mapper_bytes / 1e6)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(seq, cache, color=PALETTE["blue_main"], label="KV cache transfer")
    ax.plot(seq, mapper, color=PALETTE["red_strong"], label="Mapper transfer (fp32)")
    ax.axvline(crossover, color="black", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"crossover\n\u2248 {int(round(crossover))} tokens",
        xy=(crossover, mapper[0]),
        xytext=(crossover + 720, 80),
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )
    ax.fill_between(seq, 0, cache, where=(seq <= crossover),
                    color=PALETTE["blue_main"], alpha=0.08)
    ax.fill_between(seq, 0, mapper, where=(seq >= crossover),
                    color=PALETTE["red_strong"], alpha=0.08)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Transfer cost (MB)")
    ax.set_xlim(0, 6000)
    ax.set_ylim(0, 620)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    finalize(fig, "fig_cost")


def draw_box(ax, x, y, w, h, text, fc, tc="black", fs=12, lw=1.1, ec="black",
             weight="normal"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.05",
            fc=fc, ec=ec, lw=lw, mutation_aspect=1,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=tc, fontweight=weight,
        linespacing=1.35,
    )


def draw_arrow(ax, p0, p1, color="black", lw=1.4, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(
        mpatches.FancyArrowPatch(
            p0, p1, arrowstyle=style, color=color, lw=lw, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}", mutation_scale=14,
        )
    )


def fig_overview():
    """Conceptual overview of the problem, protocol, design space, and readouts."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
    for ax in axes.ravel():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7.4)
        ax.axis("off")

    # (a) Problem: which components cross models?
    ax = axes[0, 0]
    ax.text(5, 7.05, "(a) Problem: what transfers across models?",
            ha="center", va="center", fontsize=17, fontweight="bold")

    draw_box(ax, 4.15, 5.55, 1.7, 0.75, "document $d$",
             fc=PALETTE["neutral"], fs=13)
    draw_arrow(ax, (4.25, 5.55), (2.05, 5.35), rad=-0.15)
    draw_arrow(ax, (5.75, 5.55), (7.95, 5.35), rad=0.15)

    # Teacher group
    draw_box(ax, 0.55, 2.55, 2.75, 2.75, "", fc="white", ec="black", lw=1.2)
    ax.text(1.92, 4.95, "Teacher\n(Qwen3-8B)", ha="center", va="center",
            fontsize=12, fontweight="bold")
    draw_box(ax, 0.85, 3.55, 2.15, 0.72, "$K_T$ (addressing)",
             fc=PALETTE["blue_main"], tc="white", fs=12)
    draw_box(ax, 0.85, 2.75, 2.15, 0.72, "$V_T$ (content)",
             fc=PALETTE["red_strong"], tc="white", fs=12)

    # Student group
    draw_box(ax, 6.70, 2.55, 2.75, 2.75, "", fc="white", ec="black", lw=1.2)
    ax.text(8.08, 4.95, "Student\n(Qwen3-0.6B)", ha="center", va="center",
            fontsize=12, fontweight="bold")
    draw_box(ax, 7.00, 3.55, 2.15, 0.72, "$K_S$",
             fc=PALETTE["blue_secondary"], tc="white", fs=12)
    draw_box(ax, 7.00, 2.75, 2.15, 0.72, "$V_S$",
             fc=PALETTE["red_2"], tc="black", fs=12)

    draw_arrow(ax, (3.30, 3.10), (6.70, 3.10), color="black", lw=1.4)
    ax.text(5.0, 3.32, "map $M:T\\to S$\n(de-RoPE for K)",
            ha="center", va="bottom", fontsize=11)
    draw_arrow(ax, (3.30, 2.45), (6.70, 2.45), color=PALETTE["red_strong"], lw=1.4,
               ls="--")
    ax.text(5.0, 1.85, "Only K transfers freely;\nV is gated by capability match",
            ha="center", va="top", fontsize=11)

    # (b) Four injection arms
    ax = axes[0, 1]
    ax.text(5, 7.05, "(b) Four-arm injection protocol",
            ha="center", va="center", fontsize=17, fontweight="bold")
    arms = [
        ("Self", "Self", [("K", PALETTE["neutral"]), ("V", PALETTE["neutral"])]),
        ("K-only", "Teacher $K$\n+ student $V$",
         [("K", PALETTE["blue_main"]), ("V", PALETTE["neutral"])]),
        ("V-only", "Student $K$\n+ teacher $V$",
         [("K", PALETTE["neutral"]), ("V", PALETTE["red_strong"])]),
        ("Joint", "Teacher $K$\n+ teacher $V$",
         [("K", PALETTE["blue_main"]), ("V", PALETTE["red_2"])]),
    ]
    xs = [1.15, 3.45, 5.75, 8.05]
    for cx, (title, subtitle, slots) in zip(xs, arms):
        ax.text(cx, 6.15, title, ha="center", va="center", fontsize=14,
                fontweight="bold")
        draw_box(ax, cx - 0.55, 4.85, 1.1, 0.62, slots[0][0],
                 fc=slots[0][1], tc="white" if slots[0][1] != PALETTE["neutral"] else "black",
                 fs=13)
        draw_box(ax, cx - 0.55, 4.05, 1.1, 0.62, slots[1][0],
                 fc=slots[1][1], tc="white" if slots[1][1] != PALETTE["neutral"] else "black",
                 fs=13)
        ax.text(cx, 3.35, subtitle, ha="center", va="center", fontsize=11)
        draw_box(ax, cx - 0.85, 1.65, 1.7, 0.8, "score:\nLL / EM",
                 fc="white", fs=11)
        draw_arrow(ax, (cx, 3.95), (cx, 2.50), color="black", lw=1.0)

    # (c) Exploration space
    ax = axes[1, 0]
    ax.text(5, 7.05, "(c) Exploration space",
            ha="center", va="center", fontsize=17, fontweight="bold")
    draw_box(ax, 0.55, 4.70, 4.15, 1.55,
             "6 Qwen3 pairs\n0.6B--8B, equal/unequal layers",
             fc=PALETTE["blue_secondary"], tc="white", fs=12)
    draw_box(ax, 5.30, 4.70, 4.15, 1.55,
             "3 seeds\nentity-cluster held-out splits",
             fc=PALETTE["teal"], tc="white", fs=12)
    draw_box(ax, 0.55, 2.35, 4.15, 1.55,
             "Synthetic OOD (n=56)\n+ SQuAD cross-domain",
             fc=PALETTE["green_3"], tc="black", fs=12)
    draw_box(ax, 5.30, 2.35, 4.15, 1.55,
             "Linear mappers\nAffine/Whitened/Procrustes/CCA/Ridge",
             fc=PALETTE["violet"], tc="white", fs=11)

    # (d) Readouts
    ax = axes[1, 1]
    ax.text(5, 7.05, "(d) What the paper measures",
            ha="center", va="center", fontsize=17, fontweight="bold")
    rows = [
        ("Functional transfer", "\u0394LL and exact match across four arms"),
        ("Geometric alignment", "CCA $\\rho_1$ for K and V (correlation \u2260 transfer)"),
        ("Layer localization", "V hotspots at layers 8 and 12"),
        ("Cost crossover", "cache state vs. mapper weights (\u2248 2050 tokens)"),
    ]
    for i, (head, sub) in enumerate(rows):
        y = 5.65 - i * 1.28
        draw_box(ax, 0.55, y, 8.90, 1.02, f"{head}\n{sub}",
                 fc="white", fs=11)

    fig.tight_layout(pad=0.7)
    finalize(fig, "fig_overview")


def fig_problem():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(5, 7.05, "Problem: what transfers across models?",
            ha="center", va="center", fontsize=16, fontweight="bold")

    draw_box(ax, 4.15, 5.55, 1.7, 0.75, "document $d$",
             fc=PALETTE["neutral"], fs=13)
    draw_arrow(ax, (4.25, 5.55), (2.05, 5.35), rad=-0.15)
    draw_arrow(ax, (5.75, 5.55), (7.95, 5.35), rad=0.15)

    draw_box(ax, 0.55, 2.55, 2.75, 2.75, "", fc="white", ec="black", lw=1.2)
    ax.text(1.92, 4.95, "Teacher\n(Qwen3-8B)", ha="center", va="center",
            fontsize=12, fontweight="bold")
    draw_box(ax, 0.85, 3.55, 2.15, 0.72, "$K_T$ (addressing)",
             fc=PALETTE["blue_main"], tc="white", fs=12)
    draw_box(ax, 0.85, 2.75, 2.15, 0.72, "$V_T$ (content)",
             fc=PALETTE["red_strong"], tc="white", fs=12)

    draw_box(ax, 6.70, 2.55, 2.75, 2.75, "", fc="white", ec="black", lw=1.2)
    ax.text(8.08, 4.95, "Student\n(Qwen3-0.6B)", ha="center", va="center",
            fontsize=12, fontweight="bold")
    draw_box(ax, 7.00, 3.55, 2.15, 0.72, "$K_S$",
             fc=PALETTE["blue_secondary"], tc="white", fs=12)
    draw_box(ax, 7.00, 2.75, 2.15, 0.72, "$V_S$",
             fc=PALETTE["red_2"], tc="black", fs=12)

    draw_arrow(ax, (3.30, 3.10), (6.70, 3.10), color="black", lw=1.4)
    ax.text(5.0, 3.32, "map $M:T\\to S$\n(de-RoPE for K)",
            ha="center", va="bottom", fontsize=11)
    draw_arrow(ax, (3.30, 2.45), (6.70, 2.45), color=PALETTE["red_strong"], lw=1.4,
               ls="--")
    ax.text(5.0, 1.85, "Only K transfers freely;\nV is gated by capability match",
            ha="center", va="top", fontsize=11)

    fig.tight_layout(pad=0.5)
    finalize(fig, "fig_problem")


def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(5, 7.05, "Four-arm injection protocol",
            ha="center", va="center", fontsize=16, fontweight="bold")

    arms = [
        ("Self", "Self", [("K", PALETTE["neutral"]), ("V", PALETTE["neutral"])]),
        ("K-only", "Teacher $K$\n+ student $V$",
         [("K", PALETTE["blue_main"]), ("V", PALETTE["neutral"])]),
        ("V-only", "Student $K$\n+ teacher $V$",
         [("K", PALETTE["neutral"]), ("V", PALETTE["red_strong"])]),
        ("Joint", "Teacher $K$\n+ teacher $V$",
         [("K", PALETTE["blue_main"]), ("V", PALETTE["red_2"])]),
    ]
    xs = [1.15, 3.45, 5.75, 8.05]
    for cx, (title, subtitle, slots) in zip(xs, arms):
        ax.text(cx, 6.15, title, ha="center", va="center", fontsize=13,
                fontweight="bold")
        draw_box(ax, cx - 0.55, 4.85, 1.1, 0.62, slots[0][0],
                 fc=slots[0][1], tc="white" if slots[0][1] != PALETTE["neutral"] else "black",
                 fs=13)
        draw_box(ax, cx - 0.55, 4.05, 1.1, 0.62, slots[1][0],
                 fc=slots[1][1], tc="white" if slots[1][1] != PALETTE["neutral"] else "black",
                 fs=13)
        ax.text(cx, 3.35, subtitle, ha="center", va="center", fontsize=11)
        draw_box(ax, cx - 0.85, 1.65, 1.7, 0.8, "score:\nLL / EM",
                 fc="white", fs=11)
        draw_arrow(ax, (cx, 3.95), (cx, 2.50), color="black", lw=1.0)

    fig.tight_layout(pad=0.5)
    finalize(fig, "fig_protocol")


def fig_exploration():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(5, 7.05, "Exploration space",
            ha="center", va="center", fontsize=16, fontweight="bold")

    draw_box(ax, 0.55, 4.70, 4.15, 1.55,
             "6 Qwen3 pairs\n0.6B--8B, equal/unequal layers",
             fc=PALETTE["blue_secondary"], tc="white", fs=12)
    draw_box(ax, 5.30, 4.70, 4.15, 1.55,
             "3 seeds\nentity-cluster held-out splits",
             fc=PALETTE["teal"], tc="white", fs=12)
    draw_box(ax, 0.55, 2.35, 4.15, 1.55,
             "Synthetic OOD (n=56)\n+ SQuAD cross-domain",
             fc=PALETTE["green_3"], tc="black", fs=12)
    draw_box(ax, 5.30, 2.35, 4.15, 1.55,
             "Linear mappers\nAffine/Whitened/Procrustes/CCA/Ridge",
             fc=PALETTE["violet"], tc="white", fs=11)

    fig.tight_layout(pad=0.5)
    finalize(fig, "fig_exploration")


def fig_readouts():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(5, 7.05, "What the paper measures",
            ha="center", va="center", fontsize=16, fontweight="bold")
    rows = [
        ("Functional transfer", "\u0394LL and exact match across four arms"),
        ("Geometric alignment", "CCA $\\rho_1$ for K and V (correlation \u2260 transfer)"),
        ("Layer localization", "V hotspots at layers 8 and 12"),
        ("Cost crossover", "cache state vs. mapper weights (\u2248 2050 tokens)"),
    ]
    for i, (head, sub) in enumerate(rows):
        y = 5.65 - i * 1.28
        draw_box(ax, 0.55, y, 8.90, 1.02, f"{head}\n{sub}",
                 fc="white", fs=11)

    fig.tight_layout(pad=0.5)
    finalize(fig, "fig_readouts")


if __name__ == "__main__":
    fig_problem()
    fig_protocol()
    fig_exploration()
    fig_readouts()
    fig_main()
    fig_cca()
    fig_layers()
    fig_squad()
    fig_cost()
    print("Done.")
