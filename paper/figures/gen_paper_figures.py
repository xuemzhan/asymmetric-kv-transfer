#!/usr/bin/env python3
"""Publication figures for the corrected cross-model KV-transfer paper.

House style: figures4papers conventions, as encoded by the `scientific-figure-making`
skill (references/design-theory.md, api.md, common-patterns.md):
semantic palette, top/right spines off, frameless legends, black bar edges,
alpha-based ablation encoding, in-place value annotation, dynamic y-limits,
tight_layout(pad=2), 300 dpi PDF + PNG vector export.

Sizing note: the skill's nominal sizes (font.size 24, axes.linewidth 3) target very
large canvases (e.g. 45x12 in). Our figures are inserted at \textwidth or column
width (3.4-7.2 in), so absolute sizes are scaled down to preserve the *relative*
type-to-canvas ratio: font.size 9, axes.linewidth 1.5.

DATA PROVENANCE
Every number below is transcribed from this project's own result summaries:
  FOURARM  <- paper/audit/METRIC_CORRECTION.md 2b ; ITERATION_LOG.md W16/W18
  CONTROLS <- paper/audit/METRIC_CORRECTION.md 3  ; ITERATION_LOG.md W18
  MAPPERS  <- paper/audit/METRIC_CORRECTION.md 3c/3d ; ITERATION_LOG.md W18
  ROUTING  <- paper/audit/METRIC_CORRECTION.md 3c ; ITERATION_LOG.md W18
  ADAPTER  <- paper/audit/METRIC_CORRECTION.md 3e ; ITERATION_LOG.md W18
  LAYERS   <- paper/audit/METRIC_CORRECTION.md 3b/3c ; ITERATION_LOG.md W18
Where a value has no reported dispersion, no error bar is drawn (noted in captions).
No number is invented, and none is rounded beyond the precision of those sources.
"""
from __future__ import annotations

import logging
import os

# House style names Helvetica/Arial first; on machines without them matplotlib
# falls back through the stack. Keep the stack, silence the findfont log spam.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# --- house style ------------------------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "dark": "#4D4D4D", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BAR_LW = 1.0


def fmt_em(v):
    """Compact EM label: floor values print as a single 0."""
    return "0" if abs(v) < 0.005 else f"{v:.2f}"


def finalize(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def _rects(bars):
    """Flatten BarContainer / list / single Rectangle into a list of Rectangles."""
    if hasattr(bars, "get_height"):
        return [bars]
    out = []
    for b in bars:
        out.extend(_rects(b))
    return out


def annotate(ax, bars, fmt="{:.2f}", fs=7.0, pad=0.012):
    """In-place value labels above bars (house convention)."""
    for r in _rects(bars):
        h = r.get_height()
        if not np.isfinite(h):
            continue
        lab = fmt(h) if callable(fmt) else fmt.format(h)
        if h < 0:
            ax.text(r.get_x() + r.get_width() / 2, h - pad, lab,
                    ha="center", va="top", fontsize=fs, color="#272727")
        else:
            ax.text(r.get_x() + r.get_width() / 2, h + pad, lab,
                    ha="center", va="bottom", fontsize=fs, color="#272727")


def box(ax, x, y, w, h, text, fc, fs=8.0, ec="#272727", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p0, p1, color="#272727", lw=1.2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))


# --- transcribed data -------------------------------------------------------
PAIRS = ["8B$\\to$0.6B", "4B$\\to$1.7B", "4B$\\to$0.6B",
         "8B$\\to$1.7B", "1.7B$\\to$0.6B", "8B$\\to$4B"]

# three-seed mean (std). std 0.0 == not reported -> no error bar drawn.
EM = {
    "Self":    ([0.899, 0.440, 0.899, 0.440, 0.899, 0.815],
                [0.010, 0.010, 0.010, 0.010, 0.010, 0.021]),
    "K-only":  ([0.000, 0.054, 0.000, 0.000, 0.030, 0.149],
                [0.0, 0.018, 0.0, 0.0, 0.010, 0.010]),
    "V-only":  ([0.000, 0.000, 0.000, 0.000, 0.113, 0.440],
                [0.0, 0.0, 0.0, 0.0, 0.052, 0.010]),
    "Joint":   ([0.000, 0.024, 0.006, 0.012, 0.000, 0.065],
                [0.0, 0.010, 0.010, 0.021, 0.0, 0.055]),
}
DLL = {
    "K-only": ([2.62, 2.90, 1.58, 1.04, -0.82, 8.52],
               [0.29, 0.09, 0.02, 1.53, 0.02, 0.28]),
    "V-only": ([-0.23, 2.91, -1.22, 5.63, 0.63, 2.52],
               [0.19, 0.21, 0.06, 0.04, 0.17, 0.07]),
    "Joint":  ([4.52, 2.31, 2.50, 4.69, -2.25, 8.37],
               [0.15, 0.25, 0.52, 0.27, 0.14, 0.28]),
}

CONTROLS = [  # 8B->0.6B; three-seed means (std)
    # label, dLL, dLL std, EM
    ("K-only", 2.62, 0.29, 0.000),
    ("wrong-doc K", 2.61, 0.31, 0.000),
    ("random K", 2.33, 0.27, 0.012),
    ("zero KV", 2.75, 0.01, 0.000),
    ("Joint", 4.52, 0.15, 0.000),
    ("wrong-doc Joint", 4.50, 0.18, 0.000),
]
# Correspondence-breaking shuffles of the STUDENT's own cache. Built from the
# student state, so they are pair-independent: the same values hold on 1.7B->0.6B.
SHUFFLES = [
    # label, dLL, dLL std, EM, EM std
    ("shuf. K\n(corr.\nbroken)", -1.19, 0.10, 0.000, 0.000),
    ("shuf. V\n(corr.\nbroken)", -1.28, 0.04, 0.000, 0.000),
    ("same perm.\nK and V", -0.01, 0.00, 0.887, 0.037),
]
SELF_EM, SELF_EM_SD = 0.899, 0.010

MAPPERS = [  # V-only corrected EM, three-seed means
    ("Affine", 0.113, None, 0.000, None),
    ("OutAware", 0.935, None, 0.250, 0.018),
    ("shuffled", 0.077, None, 0.065, 0.027),
    ("$W_O$-aware", 0.899, None, 0.268, 0.018),
]
ROUTING = {"1.7B$\\to$0.6B": (0.70, 0.698, 0.701, 0.165, 0.971),
           "8B$\\to$0.6B": (0.59, 0.577, 0.590, 0.245, 0.932)}

# "Self" is the student's own cache under the SAME adapted model, so the adapted
# Self bars are the reference frame (0.940 on both pairs, against 0.899 unadapted).
ADAPTER_17 = {  # 1.7B->0.6B, three-seed mean (std)
    "Self":   {"none": (0.899, 0.010), "teacher": (0.940, 0.041),
               "student": (0.899, 0.045), "shuffled": (0.482, 0.064)},
    "K-only": {"none": (0.030, 0.010), "teacher": (0.940, 0.041),
               "student": (0.203, 0.203), "shuffled": (0.559, 0.020)},
    "V-only": {"none": (0.113, 0.051), "teacher": (0.940, 0.041),
               "student": (0.345, 0.536), "shuffled": (0.345, 0.054)},
    "Joint":  {"none": (0.000, 0.0), "teacher": (0.827, 0.237),
               "student": (0.179, 0.247), "shuffled": (0.458, 0.152)},
}
ADAPTER_8B = {  # 8B->0.6B, three-seed mean (std)
    "Self":   {"none": (0.899, 0.010), "teacher": (0.940, 0.041),
               "student": (0.976, 0.021), "shuffled": (0.411, 0.099)},
    "K-only": {"none": (0.000, 0.0), "teacher": (0.792, 0.162),
               "student": (0.113, 0.010), "shuffled": (0.369, 0.109)},
    "V-only": {"none": (0.000, 0.0), "teacher": (0.357, 0.047),
               "student": (0.107, None), "shuffled": (0.113, 0.052)},
    "Joint":  {"none": (0.000, 0.0), "teacher": (0.470, 0.119),
               "student": (0.119, 0.021), "shuffled": (0.131, 0.021)},
}

# Layer-map ablation, 1.7B->0.6B, seed 0, V-only corrected EM under two mappers.
# The affine mapper pins the value arm at the floor, so the layer map looks
# irrelevant; the consumption-space mapper lifts the proportional map to 0.911 and
# the same scrambling then costs most of it. Bars carry the seed-0
# document-clustered CI95 half-widths (affine: no interval archived).
LAYER_MAPS = [
    # label, affine V EM, outaware V EM, outaware CI half-width
    ("proportional", 0.054, 0.911, 0.0715),
    ("offset +3", 0.000, 0.214, 0.0895),
    ("offset $-$3", 0.000, 0.232, 0.0890),
    ("random perm.", 0.000, 0.107, 0.0625),
]
# Proportional V-only EM under the consumption-space mapper, three seeds
OUTAWARE_PROP_SEEDS = [0.911, 0.929, 0.964]

HELDOUT = [("val-selected\n(12,16)", 0.351), ("post-hoc\n(12,20)", 0.131),
           ("post-hoc\n(8,12)", 0.220), ("all layers", 0.000)]


# --- figures ----------------------------------------------------------------
def fig_setting():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.05, 3.6, 2.45, 1.0, "document $d$", PALETTE["green_1"], fs=7.6)
    box(ax, 3.0, 3.6, 2.2, 1.0, "teacher $T$", PALETTE["blue_secondary"], fs=8.5)
    box(ax, 3.0, 1.9, 2.2, 1.0, "student $S$", PALETTE["blue_main"], fs=8.5)
    box(ax, 6.15, 3.05, 3.65, 1.25, "mapper $M_K, M_V$\n(de-RoPE, layer map)",
        PALETTE["red_1"], fs=7.0)
    box(ax, 6.15, 1.0, 3.65, 1.2, "inject into $S$;\nanswer query",
        PALETTE["green_2"], fs=7.6)
    arrow(ax, (2.5, 4.1), (3.0, 4.1))
    arrow(ax, (2.5, 4.0), (3.0, 2.5), rad=-0.22)
    arrow(ax, (4.1, 3.6), (4.1, 2.9))
    arrow(ax, (5.2, 4.1), (6.15, 3.95), rad=0.12)
    arrow(ax, (5.2, 2.3), (6.15, 1.75), rad=-0.2)
    arrow(ax, (7.98, 3.05), (7.98, 2.2))
    ax.text(0.15, 1.05, "$C_T(d)=\\{(K^\\ell_T,V^\\ell_T)\\}$", fontsize=7.6, ha="left")
    ax.text(0.15, 0.55, "$C_S(d)=\\{(K^\\ell_S,V^\\ell_S)\\}$", fontsize=7.6, ha="left")
    ax.set_title("(a) Setting: is the mapped state consumed?")

    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    cells = [(0.2, 2.75, "K: student\nV: student", "Self", PALETTE["neutral"]),
             (5.1, 2.75, "K: student\nV: teacher", "V-only", PALETTE["red_2"]),
             (0.2, 0.35, "K: teacher\nV: student", "K-only", PALETTE["blue_secondary"]),
             (5.1, 0.35, "K: teacher\nV: teacher", "Joint", PALETTE["blue_main"])]
    for x, y, sub, name, fc in cells:
        box(ax, x, y, 4.7, 2.0, f"{name}\n{sub}", fc, fs=8.0)
    ax.set_title("(b) Four-arm injection protocol")
    finalize(fig, "fig_setting")


def fig_fourarm():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), width_ratios=[1.3, 1])
    arms = ["Self", "K-only", "V-only", "Joint"]
    cols = [PALETTE["neutral"], PALETTE["blue_secondary"], PALETTE["red_strong"], PALETTE["blue_main"]]
    x = np.arange(len(PAIRS)); w = 0.2

    ax = axes[0]
    for i, (a, c) in enumerate(zip(arms, cols)):
        m, e = EM[a]
        b = ax.bar(x + (i - 1.5) * w, m, w, yerr=[0 if s == 0 else s for s in e],
                   capsize=1.5, color=c, edgecolor="#272727", linewidth=BAR_LW, label=a)
        annotate(ax, b, fmt=fmt_em, fs=5.4)
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, fontsize=7.4, rotation=30, ha="right")
    ax.set_ylabel("corrected exact match"); ax.set_ylim(0, 1.30)
    ax.set_title("(a) Teacher arms do not answer questions")
    ax.legend(ncol=4, fontsize=6.8, loc="upper center", handlelength=1.0,
              columnspacing=1.0, handletextpad=0.5)

    ax = axes[1]
    for i, (a, c) in enumerate(zip(["K-only", "V-only", "Joint"],
                                   [PALETTE["blue_secondary"], PALETTE["red_strong"], PALETTE["blue_main"]])):
        m, e = DLL[a]
        ax.bar(x + (i - 1) * (w + 0.03), m, w + 0.03, yerr=e, capsize=1.5, color=c,
               edgecolor="#272727", linewidth=BAR_LW, label=a)
    ax.axhline(0, color="#272727", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(PAIRS, fontsize=7.4, rotation=30, ha="right")
    ax.set_ylim(-3.4, 10.4)
    ax.set_ylabel("$\\Delta$ log-likelihood vs Self")
    ax.set_title("(b) The same injection raises likelihood")
    ax.legend(ncol=3, fontsize=6.8, loc="upper center", handlelength=1.0,
              columnspacing=1.2, handletextpad=0.5)
    finalize(fig, "fig_fourarm")


def fig_controls():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6),
                             width_ratios=[1.26, 1.0, 0.98])
    labels = [c[0] for c in CONTROLS]
    y = np.arange(len(CONTROLS))[::-1]

    ax = axes[0]
    dll = [c[1] for c in CONTROLS]
    err = [0 if c[2] is None else c[2] for c in CONTROLS]
    bars = ax.barh(y, dll, xerr=err, capsize=1.5, color=PALETTE["blue_secondary"],
                   edgecolor="#272727", linewidth=BAR_LW, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#272727", lw=1.0)
    ax.set_xlabel("$\\Delta$ log-likelihood vs Self"); ax.set_xlim(0, 5.6)
    for b, c in zip(bars, CONTROLS):
        off = 0.0 if c[2] is None else c[2]
        ax.text(b.get_width() + off + 0.07, b.get_y() + b.get_height() / 2,
                f"{c[1]:+.2f}", va="center", fontsize=7.2)
    ax.set_title("(a) Content-free caches match K-only", fontsize=8.0)

    ax = axes[1]
    em = [c[3] for c in CONTROLS] + [SELF_EM]
    lab2 = labels + ["Self\n(own cache)"]
    cols = [PALETTE["red_2"]] * len(CONTROLS) + [PALETTE["neutral"]]
    e2 = [0] * len(CONTROLS) + [SELF_EM_SD]
    yy = np.arange(len(em))[::-1]
    b = ax.barh(yy, em, xerr=e2, capsize=1.5, color=cols, edgecolor="#272727",
                linewidth=BAR_LW, height=0.62)
    ax.set_yticks(yy); ax.set_yticklabels(lab2, fontsize=8)
    ax.set_xlabel("corrected exact match"); ax.set_xlim(0, 1.20)
    for r, sd in zip(b, e2):
        w = r.get_width()
        ax.text(w + sd + 0.02, r.get_y() + r.get_height() / 2,
                f"{w:.3f}" if w < 0.1 else f"{w:.2f}", va="center", fontsize=7.2)
    ax.set_title("(b) Only the correct cache answers", fontsize=8.0)

    ax = axes[2]
    sl = [s[0] for s in SHUFFLES]
    sem = [s[3] for s in SHUFFLES]
    esd = [s[4] for s in SHUFFLES]
    xs = np.arange(len(SHUFFLES))
    cols = [PALETTE["red_2"], PALETTE["red_2"], PALETTE["neutral"]]
    b = ax.bar(xs, sem, 0.62, yerr=esd, capsize=1.5, color=cols,
               edgecolor="#272727", linewidth=BAR_LW)
    annotate(ax, [b], fmt=fmt_em, fs=7.4)
    ax.axhline(SELF_EM, color=PALETTE["dark"], lw=1.2, ls="--")
    ax.text(0.06, SELF_EM + 0.035, f"Self {SELF_EM:.2f}", fontsize=6.8,
            ha="left", color="#272727")
    ax.set_xticks(xs); ax.set_xticklabels(sl, fontsize=6.4)
    ax.set_xlim(-0.6, len(sem) - 0.4)
    ax.set_ylim(0, 1.12); ax.set_ylabel("corrected EM")
    ax.set_title("(c) K--V correspondence registers", fontsize=8.0)
    finalize(fig, "fig_controls")


def fig_mappers():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), width_ratios=[1.15, 1])
    ax = axes[0]
    names = [m[0] for m in MAPPERS]
    x = np.arange(len(names)); w = 0.34
    for i, (pi, c, lab) in enumerate([(1, PALETTE["blue_main"], "1.7B$\\to$0.6B"),
                                      (3, PALETTE["red_strong"], "8B$\\to$0.6B")]):
        m = [MAPPERS[j][pi] for j in range(len(names))]
        e = [0 if MAPPERS[j][pi + 1] is None else MAPPERS[j][pi + 1] for j in range(len(names))]
        b = ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=1.5, color=c,
                   edgecolor="#272727", linewidth=BAR_LW, label=lab)
        annotate(ax, b, fmt=fmt_em, fs=6.6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("corrected V-only exact match"); ax.set_ylim(0, 1.30)
    ax.set_title("(a) Fit the value mapper in the consumer's space")
    ax.legend(ncol=2, fontsize=7.5)

    ax = axes[1]
    pairs = list(ROUTING)
    x = np.arange(len(pairs)); w = 0.26
    series = [("top-1 agreement", 0, PALETTE["blue_main"]),
              ("routing TV", 3, PALETTE["violet"]),
              ("attention cosine", 4, PALETTE["teal"])]
    for i, (lab, idx, c) in enumerate(series):
        v = [ROUTING[p][idx] for p in pairs]
        b = ax.bar(x + (i - 1) * w, v, w, color=c, edgecolor="#272727",
                   linewidth=BAR_LW, label=lab)
        annotate(ax, b, fmt="{:.2f}", fs=6.6)
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylim(0, 1.52)
    ax.set_ylabel("diagnostic (0-1)")
    ax.set_title("(b) Routing diverges under mapped keys")
    ax.legend(ncol=2, fontsize=6.8, loc="upper center", handlelength=1.0,
              columnspacing=1.2, handletextpad=0.5)
    finalize(fig, "fig_mappers")


def _adapter_panel(ax, data, title):
    arms = ["Self", "K-only", "V-only", "Joint"]
    conds = [("none", PALETTE["neutral"]), ("teacher", PALETTE["blue_main"]),
             ("student", PALETTE["dark"]), ("shuffled", PALETTE["red_2"])]
    alpha = {"none": 0.95, "teacher": 1.0, "student": 0.75, "shuffled": 0.85}
    hatch = {"none": "", "teacher": "", "student": "//", "shuffled": ".."}
    x = np.arange(len(arms)); w = 0.2
    for i, (cname, c) in enumerate(conds):
        m, e = [], []
        for a in arms:
            val, sd = data[a][cname]
            m.append(val); e.append(0 if sd is None else sd)
        b = ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=1.5, color=c,
                   edgecolor="#272727", linewidth=BAR_LW, alpha=alpha[cname],
                   hatch=hatch[cname], label=cname)
        annotate(ax, b, fmt=fmt_em, fs=5.4)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=7.6)
    ax.set_ylim(0, 1.34); ax.set_ylabel("corrected exact match")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=6.8, title="adapter trained on", title_fontsize=6.8,
              loc="upper center", handlelength=1.0, columnspacing=1.0,
              handletextpad=0.5)


def fig_adapter():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    _adapter_panel(axes[0], ADAPTER_17, "(a) 1.7B$\\to$0.6B (three seeds)")
    _adapter_panel(axes[1], ADAPTER_8B, "(b) 8B$\\to$0.6B (three seeds)")
    finalize(fig, "fig_adapter")


def fig_layers():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    names = [m[0] for m in LAYER_MAPS]
    x = np.arange(len(names)); w = 0.34
    series = [(1, PALETTE["neutral"], "affine mapper", [0] * len(LAYER_MAPS)),
              (2, PALETTE["blue_main"], "consumption-space mapper",
               [m[3] for m in LAYER_MAPS])]
    for i, c, lab, err in series:
        v = [m[i] for m in LAYER_MAPS]
        b = ax.bar(x + (i - 1.5) * w, v, w, yerr=err, capsize=1.5, color=c,
                   edgecolor="#272727", linewidth=BAR_LW, label=lab)
        for r, val, e in zip(b, v, err):
            ax.text(r.get_x() + r.get_width() / 2, val + e + 0.022, fmt_em(val),
                    ha="center", va="bottom", fontsize=6.4, color="#272727")
    ax.axhline(0, color="#272727", lw=1.0)
    ax.set_ylim(0, 1.22)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel("corrected V-only EM (seed 0)")
    ax.set_title("(a) Alignment matters once the consumer reads value")
    ax.legend(ncol=1, fontsize=6.8, loc="upper right", handlelength=1.0,
              columnspacing=1.0, handletextpad=0.5)

    ax = axes[1]
    lab = [h[0] for h in HELDOUT]
    v = [h[1] for h in HELDOUT]
    x = np.arange(len(v))
    b = ax.bar(x, v, 0.55,
               color=[PALETTE["blue_main"], PALETTE["red_2"],
                      PALETTE["red_2"], PALETTE["neutral"]],
               edgecolor="#272727", linewidth=BAR_LW)
    annotate(ax, [b], fmt="{:.3f}", fs=6.8)
    ax.axhline(SELF_EM, color=PALETTE["dark"], lw=1.2, ls="--")
    ax.text(len(v) - 0.45, SELF_EM + 0.02, f"Self {SELF_EM:.2f}", fontsize=7.4,
            ha="right", color="#272727")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("test corrected exact match")
    ax.set_title("(b) Held-out value-layer selection")
    finalize(fig, "fig_layers")


EVALCHECK = [  # student, token agreement, normalized-EM agreement, logit max err,
               # self EM under injection, self EM under full prefill
    ("$0.6$B", 0.786, 0.964, 1.344, 0.911, 0.875),
    ("$1.7$B", 0.804, 0.929, 0.875, 0.429, 0.393),
    ("$4$B", 0.929, 0.982, 0.969, 0.804, 0.786),
]


def fig_evalcheck():
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4))
    x = np.arange(len(EVALCHECK)); w = 0.34

    ax = axes[0]
    for i, (key, c, lab) in enumerate([(1, PALETTE["neutral"], "token-level greedy"),
                                       (2, PALETTE["blue_main"], "normalized EM")]):
        v = [e[key] for e in EVALCHECK]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=c, edgecolor="#272727",
                   linewidth=BAR_LW, label=lab)
        annotate(ax, b, fmt="{:.3f}", fs=6.4, pad=0.015)
    ax.set_xticks(x); ax.set_xticklabels([e[0] for e in EVALCHECK], fontsize=7.6)
    ax.set_ylim(0, 1.22); ax.set_ylabel("agreement")
    ax.set_title("(a) Injection vs full prefill")
    ax.legend(ncol=1, fontsize=6.8, loc="upper center", handlelength=1.0,
              columnspacing=1.0, handletextpad=0.5)

    ax = axes[1]
    for i, (key, c, lab) in enumerate([(5, PALETTE["blue_main"], "full prefill"),
                                       (4, PALETTE["green_3"], "cache injection")]):
        v = [e[key] for e in EVALCHECK]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=c, edgecolor="#272727",
                   linewidth=BAR_LW, label=lab)
        annotate(ax, b, fmt=fmt_em, fs=6.4, pad=0.015)
    ax.set_xticks(x); ax.set_xticklabels([e[0] for e in EVALCHECK], fontsize=7.6)
    ax.set_ylim(0, 1.22); ax.set_ylabel("Self corrected EM")
    ax.set_title("(b) The two paths agree on Self")
    ax.legend(ncol=1, fontsize=6.8, loc="upper center", handlelength=1.0,
              columnspacing=1.0, handletextpad=0.5)
    finalize(fig, "fig_evalcheck")


if __name__ == "__main__":
    fig_setting()
    fig_fourarm()
    fig_controls()
    fig_mappers()
    fig_adapter()
    fig_layers()
    fig_evalcheck()
    print("done")
