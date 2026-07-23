#!/usr/bin/env python3
"""Data figures for "Climbing the Chomsky Hierarchy" (matplotlib).

SCOPE: this file owns ONLY the two DATA plots -- the companion-measured oracle envelope and the measured
oracle-gaming collapse. The five schematic diagrams (architecture, certificate, hierarchy, language, worked)
are publication-grade TikZ and live in tikz/*.tex -- do NOT recreate them here (single source of truth).
Build everything with build_figs.sh. Non-math text uses plain characters; LaTeX symbols live inside $...$.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#3a3a3a",
    "text.color": "#1b1b1b",
    "axes.labelcolor": "#1b1b1b",
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 300,
})

INK = "#1b1b1b"; GREY = "#6f6f6f"; FAINT = "#9a9a9a"
CF = "#2f7d4f"; MCF = "#2166ac"; NEG = "#a5352c"


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_oracle():
    """Companion-measured cross-predictor envelope (model-relativity of the positive side)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(-6, 106); ax.set_ylim(0, 3.3); ax.axis("off")
    ax.text(50, 3.05, "The positive side is model-relative", ha="center", fontsize=10.5, weight="bold")
    ax.text(50, 2.6, "fraction of SOTA pseudoknot designs a predictor judges “formed” (companion-measured)",
            ha="center", fontsize=7.3, color=GREY)
    ax.plot([0, 100], [1.0, 1.0], color=INK, lw=0.9)
    for t in (0, 25, 50, 75, 100):
        ax.plot([t, t], [0.95, 1.05], color=INK, lw=0.7)
        ax.text(t, 0.68, f"{t}%", ha="center", fontsize=7, color=GREY)
    ax.add_patch(Rectangle((15, 0.94), 45, 0.12, ec="none", fc="#e8dcda"))
    ax.annotate("", xy=(60, 1.78), xytext=(15, 1.78), arrowprops=dict(arrowstyle="<->", color=NEG, lw=1.0))
    ax.text(37.5, 1.92, "cross-predictor envelope $[15,60]\\%$ (disagreement $\\Rightarrow$ abstain)",
            ha="center", fontsize=7.3, color=NEG)
    ax.plot([15], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(15, 1.24, "ProbKnot 15%", ha="center", fontsize=7.1, color=MCF)
    ax.plot([28], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(30, 1.44, "pKiss 28% (ref.)", ha="left", fontsize=7.1, color=MCF)
    ax.plot([60], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(60, 1.24, "ShapeKnots 60%", ha="center", fontsize=7.1, color=MCF)
    ax.plot([9], [1.0], "s", ms=6, color=CF, mec="white", mew=0.6, zorder=3)
    ax.text(9, 0.36, "panel-unanimous\n(conservative)", ha="center", fontsize=6.7, color=CF)
    _save(fig, "fig_cc_oracle.pdf")


def fig_collapse():
    """MEASURED (n=60): the single-oracle 'ascent' collapses ~43x under the cross-oracle panel."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels = ["CF floor\n(REMC, $-$solver)", "autonomous\n(pKiss only)",
              "autonomous\n$\\cap$ ProbKnot", "autonomous\npanel-unanimous"]
    vals = [0, 43, 2, 1]
    xs = list(range(4))
    bars = ax.bar(xs, vals, width=0.60, color=MCF, edgecolor=INK, lw=0.6, zorder=2)
    bars[0].set_color(FAINT); bars[2].set_color("#c9b47e"); bars[3].set_color(NEG)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.1, f"{v}/60", ha="center", fontsize=9, color=INK)
    ax.axhline(43, ls=(0, (5, 2.5)), color=GREY, lw=0.8, zorder=1)
    ax.text(3.45, 44.0, "DesiRNA (borrowed) 43/60", ha="right", va="bottom", fontsize=7, color=GREY)
    ax.annotate("", xy=(3, 3.0), xytext=(1, 41.5), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=NEG, lw=1.1, connectionstyle="arc3,rad=-0.30"))
    ax.text(2.52, 36, "$\\sim 43\\times$ collapse\nunder the panel", ha="center", va="center", fontsize=7.8, color=NEG)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0, 49); ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("verified held-out solves ($/60$)", fontsize=8.5)
    ax.set_title("The panel exposes oracle-gaming (measured; $n=60$)", fontsize=10, weight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    _save(fig, "fig_cc_collapse.pdf")


if __name__ == "__main__":
    fig_oracle()
    fig_collapse()
    print("wrote figs/fig_cc_{oracle,collapse}.pdf (data figures; the 5 diagrams are TikZ in tikz/*.tex)")
