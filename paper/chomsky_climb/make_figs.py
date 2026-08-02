#!/usr/bin/env python3
"""Data figures for "Climbing the Chomsky Hierarchy" (matplotlib).

SCOPE: this file owns ONLY the two DATA plots -- the companion-measured oracle envelope and the measured
oracle-gaming collapse. The five schematic diagrams (architecture, certificate, hierarchy, language, worked)
are publication-grade TikZ and live in tikz/*.tex -- do NOT recreate them here (single source of truth).
Build everything with build_figs.sh. Non-math text uses plain characters; LaTeX symbols live inside $...$.
"""
import os
import re

import matplotlib
# PGF, not Agg: the text is typeset by LaTeX itself, so the data figures carry the same Computer Modern
# as the body text and the TikZ schematics. Output is still vector PDF. Agg's "serif + mathtext cm" is
# a near-miss that shows when a matplotlib figure sits next to a TikZ one.
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{amsmath,amssymb}",
    "font.family": "serif",
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
CF = "#2f7d4f"; MCF = "#2166ac"; NEG = "#a5352c"; TAN = "#c9b47e"; TAN_EDGE = "#a98a3d"


def tint(hexc, frac):
    """Mix hexc toward white; frac = fraction of the saturated color (like TikZ `color!N`, N=frac*100)."""
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(c * frac + 255 * (1 - frac)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# Light tints for bar/hist FILLS; the saturated hue is kept for the EDGE. This mirrors the TikZ
# schematics' airy `fill=color!8..22 + draw=color` look so the data plots (Fig 7+) read as warm/light
# and consistent with Figs <=6, instead of cold solid blocks.
MCF_F = tint(MCF, 0.34); CF_F = tint(CF, 0.34); NEG_F = tint(NEG, 0.34)
TAN_F = tint(TAN, 0.55); FAINT_F = tint(FAINT, 0.42)


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
        ax.text(t, 0.68, f"{t}\\%", ha="center", fontsize=7, color=GREY)
    ax.add_patch(Rectangle((15, 0.94), 45, 0.12, ec="none", fc="#e8dcda"))
    ax.annotate("", xy=(60, 1.78), xytext=(15, 1.78), arrowprops=dict(arrowstyle="<->", color=NEG, lw=1.0))
    ax.text(37.5, 1.92, "cross-predictor envelope $[15,60]\\%$ (disagreement $\\Rightarrow$ abstain)",
            ha="center", fontsize=7.3, color=NEG)
    ax.plot([15], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(15, 1.24, "ProbKnot 15\\%", ha="center", fontsize=7.1, color=MCF)
    ax.plot([28], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(30, 1.44, "pKiss 28\\% (ref.)", ha="left", fontsize=7.1, color=MCF)
    ax.plot([60], [1.0], "o", ms=6, color=MCF, mec="white", mew=0.6, zorder=3)
    ax.text(60, 1.24, "ShapeKnots 60\\%", ha="center", fontsize=7.1, color=MCF)
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
    bars = ax.bar(xs, vals, width=0.60, facecolor=MCF_F, edgecolor=MCF, lw=1.1, zorder=2)
    bars[0].set_facecolor(FAINT_F); bars[0].set_edgecolor(GREY)
    bars[2].set_facecolor(TAN_F);   bars[2].set_edgecolor(TAN_EDGE)
    bars[3].set_facecolor(NEG_F);   bars[3].set_edgecolor(NEG)
    # Exact binomial (Clopper-Pearson) intervals, drawn because the last bar is a single design and a bare
    # "43x" annotation would headline a ratio whose denominator is 1 -- the paper says in text that nothing
    # rests on that ratio, and the figure should not say otherwise.
    cis = {0: (0.0, 6.0), 1: (32.6, 49.4), 2: (0.4, 11.6), 3: (0.03, 5.4)}
    for x, v in zip(xs, vals):
        lo, hi = cis[x]
        ax.plot([x, x], [lo, hi], color=INK, lw=1.0, zorder=4, alpha=0.55)
        ax.plot([x - 0.09, x + 0.09], [hi, hi], color=INK, lw=1.0, zorder=4, alpha=0.55)
        ax.text(x, hi + 1.6, f"{v}/60", ha="center", fontsize=9, color=INK)
    # The y-limit has to clear the tallest whisker AND its label, or the 43/60 label rides over the title.
    ax.set_ylim(0, 58); ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.axhline(43, ls=(0, (5, 2.5)), color=GREY, lw=0.8, zorder=1)
    # Reference label goes LEFT, over the empty CF-floor column: the right side is where the 43/60 whisker
    # and its label now live.
    ax.text(-0.42, 43.8, "DesiRNA (borrowed) $43/60$", ha="left", va="bottom", fontsize=7, color=GREY)
    # Straight arrow, not an arc: with a 4:1 aspect the curvature of arc3 is unpredictable in data units and
    # it was running through the annotation text. Landing point is offset up-left of bar 3 so it clears the
    # 1/60 label, which sits centred above the whisker.
    ax.annotate("", xy=(2.70, 9.0), xytext=(1.28, 40.0), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=NEG, lw=1.1))
    # Text sits to the RIGHT of that line (which passes x=1.88 at y=28), in the empty upper-right block.
    ax.text(2.06, 27.5, "$43 \\to 1$ under three\npredictors, same designs", ha="left", va="center",
            fontsize=7.8, color=NEG)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("verified held-out solves ($/60$)", fontsize=8.5)
    ax.set_title("The panel exposes oracle-gaming (measured; $n=60$)", fontsize=10, weight="bold", pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    _save(fig, "fig_cc_collapse.pdf")


RUNS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "rna_harness", "runs", "cc_ascent"))

# Arm rows in the committed stats outputs look like
#   "  baseline (hand-built)        37.0/60    20/ 185 = 0.108 [0.053,0.175]  [37, ...]"
#   "  frozen agent operators (reference)   ---   333/1104 = 0.302 [0.275,0.330]"
# so one regex reads every attribution table. Parsing them rather than transcribing keeps the figure
# and the manuscript on the same source of truth: a re-run that moves a number moves the plot.
_ARM = re.compile(
    r"^ {2}(\S.*?)\s\s+(?:([\d.]+)/60|-{3})\s+(\d+)/\s*(\d+)\s*=\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]")


def _arms(fname):
    """-> {arm label: (r_in or None, rate, lo, hi)} for one committed stats .txt."""
    path = os.path.join(RUNS, fname)
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        m = _ARM.match(line.rstrip("\n"))
        if m:
            lab, rin, _k, _n, rate, lo, hi = m.groups()
            out[lab.strip()] = (float(rin) if rin else None, float(rate), float(lo), float(hi))
    return out


def fig_attrib():
    """Every intervention on the hand-built operator, against the arm it is trying to reproduce.

    This is the seven-test attribution study as one picture. The point is not any single arm but the
    shape of the whole column: nine interventions spanning the objective, the compute allowance in both
    directions, and the two structural properties that survived a composition-controlled screen, and the
    carry-over rate does not leave the baseline's interval in any of them, while the agent-written arm
    sits outside all of them. The right panel carries the rule fixed before the runs -- carry-over must
    rise AND r_in must hold -- because a conditional rate bought by collapsing its own denominator is the
    artifact that rule exists to catch, and the 30-call arm is exactly that case.
    """
    a = _arms("attr_stats.txt"); t = _arms("tinyctl_stats.txt")
    n = _arms("ndc_stats.txt"); d = _arms("dom_stats.txt")
    if not (a and t and n and d):
        print("  (skip fig_attrib: attribution stats outputs not found)"); return

    # (display label, source dict, key in that file, group)
    SPEC = [
        ("hand-built operator (baseline)",        a, "baseline (hand-built)",                    "base"),
        ("ungated ProbKnot",                      a, "+ ungated ProbKnot",                       "obj"),
        ("scalar crossing cost",                  a, "+ scalar cost (sym)",                      "obj"),
        ("scalar cost, FP penalty $\\times2$",    a, "+ scalar, FP x2",                          "obj"),
        ("annealed acceptance",                   a, "+ annealing",                              "obj"),
        ("$3\\times$ pKiss allowance",            a, "3x pKiss allowance [compute control]",     "cmp"),
        ("$30$-call allowance (agents' own)",     t, "30-call allowance [inverse control]",      "cmp"),
        ("decoy-minimizing stem choice",          n, "+ decoy tiebreak [stage-2 transplant]",    "d3"),
        ("ensemble-gap minimization",             d, "+ ensemble-dominance bias [stage-2]",      "d3"),
        ("agent-written operators",               a, "frozen agent operators (reference)",       "agent"),
    ]
    rows = [(lab, grp) + src[key] for lab, src, key, grp in SPEC if key in src]
    if len(rows) != len(SPEC):
        print("  (skip fig_attrib: an arm is missing from the stats outputs)"); return

    COL = {"base": INK, "obj": MCF, "cmp": TAN_EDGE, "d3": NEG, "agent": CF}
    base_rate, base_lo, base_hi = rows[0][3], rows[0][4], rows[0][5]
    base_rin = rows[0][2]
    ys = list(range(len(rows) - 1, -1, -1))          # first row at the top

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.0, 3.15), gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.09})

    # -- left: carry-over to the held-out predictor, with the baseline's interval as a band
    axL.axvspan(base_lo, base_hi, color=FAINT, alpha=0.16, lw=0, zorder=0)
    axL.axvline(base_rate, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for y, (lab, grp, _rin, rate, lo, hi) in zip(ys, rows):
        c = COL[grp]
        axL.plot([lo, hi], [y, y], color=c, lw=1.5, alpha=0.85, zorder=3,
                 solid_capstyle="round")
        axL.plot([rate], [y], "o", ms=6.0, mfc=tint(c, 0.30), mec=c, mew=1.2, zorder=4)
    axL.set_yticks(ys)
    axL.set_yticklabels([lab for lab, *_ in rows], fontsize=7.8)
    for tick, (_lab, grp, *_r) in zip(axL.get_yticklabels(), rows):
        tick.set_color(INK if grp in ("base", "agent") else "#3a3a3a")
    axL.set_ylim(-0.8, len(rows) - 0.2)
    axL.set_xlim(0.0, 0.40)
    axL.set_xlabel("carry-over to the held-out predictor (95\\% CI)", fontsize=8)
    axL.tick_params(axis="y", length=0)
    axL.grid(axis="x", color=FAINT, lw=0.4, alpha=0.35, zorder=0)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.text(base_rate + 0.006, len(rows) - 0.62, "baseline $0.108$ and its interval",
             fontsize=6.8, color=GREY, va="center")

    # group brackets, so the reader sees WHAT was varied without reading the labels one by one
    # x in axes coords (left of the tick labels), y in data coords: get_yaxis_transform does exactly that
    tr = axL.get_yaxis_transform()
    for grp, name in (("obj", "objective"), ("cmp", "compute"), ("d3", "structure")):
        gy = [y for y, r in zip(ys, rows) if r[1] == grp]
        axL.plot([-0.475, -0.475], [min(gy) - 0.30, max(gy) + 0.30], color=COL[grp], lw=1.4,
                 transform=tr, clip_on=False, zorder=5, solid_capstyle="round")
        axL.text(-0.493, sum(gy) / len(gy), name, fontsize=6.9, color=COL[grp], rotation=90,
                 ha="center", va="center", transform=tr, clip_on=False)

    # -- right: the other half of the pre-fixed rule
    axR.axvline(base_rin, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for y, (_lab, grp, rin, *_r) in zip(ys, rows):
        if rin is None:
            axR.text(2.0, y, "not applicable", fontsize=6.8, color=FAINT, va="center")
            continue
        c = COL[grp]
        axR.plot([0, rin], [y, y], color=tint(c, 0.30), lw=3.4, solid_capstyle="butt", zorder=2)
        axR.plot([rin], [y], "|", ms=8, color=c, mew=1.3, zorder=3)
    axR.set_yticks(ys); axR.set_yticklabels([])
    axR.set_ylim(-0.8, len(rows) - 0.2); axR.set_xlim(0, 61)
    axR.set_xticks([15, 37])
    axR.set_xticklabels(["$15$", "$37$"], fontsize=7.5)
    axR.set_xlabel("$r_{\\mathrm{in}}$ per $60$ targets", fontsize=8)
    axR.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axR.spines[sp].set_visible(False)
    axR.annotate("\\shortstack[l]{denominator collapses;\\\\the rule rejects this arm}",
                 xy=(15.6, ys[6]), xytext=(39.5, ys[6]), fontsize=6.6, color=NEG,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=NEG, lw=0.7, shrinkA=2, shrinkB=3))

    _save(fig, "fig_cc_attrib.pdf")
    print("  wrote figs/fig_cc_attrib.pdf")


def fig_family():
    """The success-vs-size curve on the unbounded family, and the separation that stops us mis-reading it.

    The confirmed rate r_mix rises with the family parameter, which reads as a fixed-description operator
    holding up as the family grows. It is a product: r_mix = r_in x carry-over, and only the second factor
    is rising. Plotting the factors is the whole argument, and the gamer's pKiss->ProbKnot agreement is on
    the same axes because it moves the OTHER way, which is what rules out "longer stems just make
    predictors agree" as a blanket effect. Reads family_diag.txt (100 targets x 5 seeds, deterministic).
    """
    path = os.path.join(RUNS, "family_diag.txt")
    if not os.path.exists(path):
        print("  (skip fig_family: family_diag.txt not found)"); return
    xs, rin, carry, gam = [], [], [], []
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[0].isdigit():
            xs.append(int(p[0])); rin.append(float(p[1]))
            carry.append(float(p[2])); gam.append(float(p[3]))
    if not xs:
        print("  (skip fig_family: no diagonal rows parsed)"); return
    mix = [r * c for r, c in zip(rin, carry)]

    fig, ax = plt.subplots(figsize=(7.0, 2.95))
    ax.grid(axis="y", color=FAINT, lw=0.4, alpha=0.35, zorder=0)
    ax.plot(xs, mix, color=FAINT, lw=1.2, ls=(0, (3, 2)), zorder=2,
            label="$r_{\\mathrm{mix}}=r_{\\mathrm{in}}\\times$carry-over (what rises)")
    ax.plot(xs, rin, color=MCF, lw=1.6, marker="o", ms=3.4, mfc="white", mew=1.1, zorder=4,
            label="$r_{\\mathrm{in}}$: the operator's own rate")
    ax.plot(xs, carry, color=CF, lw=1.6, marker="s", ms=3.2, mfc="white", mew=1.1, zorder=4,
            label="carry-over to the held-out predictor")
    ax.plot(xs, gam, color=NEG, lw=1.1, ls=(0, (1, 1.6)), marker="^", ms=3.0, zorder=3,
            label="gamer's pKiss$\\to$ProbKnot agreement (control)")

    ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4); ax.set_ylim(-0.04, 1.17)
    ax.set_xticks([x for x in xs if x % 2 == 0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("family parameter $n{+}m$ (crossing-stem size, target set $\\mathcal{L}_{\\mathrm H}$)",
                  fontsize=8.5)
    ax.set_ylabel("rate", fontsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=3)

    # the two summary strata the text quotes, marked rather than restated
    ax.axvspan(min(xs) - 0.4, 12.5, color=MCF, alpha=0.05, lw=0, zorder=0)
    ax.axvspan(17.5, max(xs) + 0.4, color=CF, alpha=0.05, lw=0, zorder=0)
    ax.text(9.0, 1.07, "$n{+}m\\le12$", fontsize=7, color=GREY, ha="center")
    ax.text(21.0, 1.07, "$n{+}m\\ge18$", fontsize=7, color=GREY, ha="center")
    ax.annotate("\\shortstack{the operator's reach falls\\\\while the panel's willingness rises}",
                xy=(15.0, 0.585), xytext=(14.2, 0.97), fontsize=6.9, color=INK, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, shrinkA=2, shrinkB=2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.245), ncol=2, frameon=False,
              fontsize=7, handletextpad=0.5, columnspacing=1.4, handlelength=2.4)
    _save(fig, "fig_cc_family.pdf")
    print("  wrote figs/fig_cc_family.pdf")


def fig_grade():
    """Grammar-graded capability frontier: solve rate vs intrinsic crossing complexity (crossing-pair count).
    Reads runs/cc_ascent/grade_frontier.json. Shows formal complexity predicting the empirical frontier."""
    import json
    jp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
         "rna_harness", "runs", "cc_ascent", "grade_frontier.json"))
    if not os.path.exists(jp):
        print("  (skip fig_grade: %s not found)" % jp); return
    R = json.load(open(jp))["by_grade"]
    g = [r["grade"] for r in R]; nn = [r["n"] for r in R]

    def rate(key):
        return [100.0 * r[key] / r["n"] for r in R]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.plot(g, rate("gamer_pkiss"), "-o", color=GREY, lw=1.2, ms=5, mec="white", mew=0.6,
            label="single-oracle operator (pKiss)")
    ax.plot(g, rate("panel_pk_pb"), "-s", color=MCF, lw=1.4, ms=5, mec="white", mew=0.6,
            label="panel operator (pKiss $\\cap$ ProbKnot)")
    ax.plot(g, rate("panel_unanimous"), "-D", color=CF, lw=1.6, ms=5.5, mec="white", mew=0.6,
            label="panel operator (panel-unanimous)")
    ax.axhline(0, color=FAINT, lw=0.8, ls=(0, (4, 3)))
    for gi, ni in zip(g, nn):
        ax.text(gi, -9, f"$n{{=}}{ni}$", ha="center", fontsize=6.3, color=GREY)
    ax.set_xlabel("intrinsic crossing complexity: crossing-pair count (exact grade)", fontsize=8.5)
    ax.set_ylabel("held-out solve rate (\\%)", fontsize=8.5)
    ax.set_ylim(-13, 100); ax.set_xticks(g); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Solve rates stratified by crossing-pair count",
                 fontsize=10, weight="bold", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    _save(fig, "fig_cc_grade.pdf")
    print("  wrote figs/fig_cc_grade.pdf")


def _cc(rel):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rna_harness", "runs", "cc_ascent", rel))


def fig_grades():
    """Appendix C: crossing-complexity (crossing-pair count) distribution across the held-out pools."""
    import json
    from collections import Counter
    gp = _cc("grade_frontier.json"); bp = _cc("bprna/sweep.json")
    if not (os.path.exists(gp) and os.path.exists(bp)):
        print("  (skip fig_grades)"); return
    pb = {r["grade"]: r["n"] for r in json.load(open(gp))["by_grade"]}
    br = Counter(sum(1 for c in r["target"] if c in "[]{}<>") // 2 for r in json.load(open(bp))["rows"])
    grades = sorted(set(pb) | set(br))
    import numpy as np
    x = np.arange(len(grades)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.bar(x - w / 2, [pb.get(g, 0) for g in grades], w, facecolor=MCF_F, ec=MCF, lw=1.0, label="Pseudobase++ held-out ($n{=}60$)")
    ax.bar(x + w / 2, [br.get(g, 0) for g in grades], w, facecolor=CF_F, ec=CF, lw=1.0, label="bpRNA cross-source ($n{=}60$)")
    ax.set_xticks(x); ax.set_xticklabels(grades); ax.set_xlabel("crossing-pair count (intrinsic grade $g$)", fontsize=8.5)
    ax.set_ylabel("targets", fontsize=8.5)
    ax.set_title("Crossing-complexity spectrum of the held-out pools", fontsize=10, weight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7.2); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_grades.pdf"); print("  wrote fig_cc_grades.pdf")


def fig_invent():
    """Appendix E: autonomous-invention learning curve (training solve rate vs round)."""
    import json
    p = _cc("invent_operator/result.json")
    if not os.path.exists(p):
        print("  (skip fig_invent)"); return
    d = json.load(open(p))
    rds = [(r["round"], r["train_solved"]) for r in d["rounds"] if r.get("train_solved") is not None]
    if not rds:
        print("  (skip fig_invent: no rounds)"); return
    xs, ys = zip(*rds)
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(xs, ys, "-o", color=MCF, lw=1.5, ms=6, mec="white", mew=0.7)
    best = d.get("best_round")
    if best in xs:
        bi = xs.index(best); ax.plot([xs[bi]], [ys[bi]], "o", ms=10, mfc="none", mec=NEG, mew=1.5)
        ax.annotate("frozen $\\to$ held-out\n$9/60$ panel-unanimous", xy=(xs[bi], ys[bi]),
                    xytext=(xs[bi] + 0.3, ys[bi] - 4.5), fontsize=7.2, color=NEG,
                    arrowprops=dict(arrowstyle="->", color=NEG, lw=0.9))
    ax.set_xlabel("invention round (LLM writes $\\to$ sandbox-runs $\\to$ revises)", fontsize=8.5)
    ax.set_ylabel("training targets solved ($/16$)", fontsize=8.5)
    ax.set_ylim(0, 16); ax.set_xticks(list(xs))
    ax.set_title("An agent learns to write the operator", fontsize=10, weight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_invent.pdf"); print("  wrote fig_cc_invent.pdf")


def fig_stringency():
    """Appendix F: thermodynamic-dominance (energy-gap) histogram of the panel-confirmed designs."""
    import json
    p = _cc("positive_stringency.json")
    if not os.path.exists(p):
        print("  (skip fig_stringency: run positive_stringency.py first)"); return
    d = json.load(open(p))
    gaps = [r["gap"] for r in d["rows"] if "gap" in r]
    if not gaps:
        print("  (skip fig_stringency: no gaps)"); return
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ax.hist(gaps, bins=bins, facecolor=CF_F, ec=CF, lw=1.0)
    ax.axvline(1.0, color=NEG, ls=(0, (4, 2)), lw=1.0, label="1 kcal/mol dominance threshold")
    ax.set_ylim(0, 8.6)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right", handlelength=1.7, borderaxespad=0.4)
    ax.set_xlabel("free-energy gap to the best non-target fold (kcal/mol)", fontsize=8.5)
    ax.set_ylabel("panel-confirmed designs", fontsize=8.5)
    ax.set_title("Positive-side stringency: the target is the pKiss free-energy minimum",
                 fontsize=10, weight="bold", pad=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_stringency.pdf"); print("  wrote fig_cc_stringency.pdf")


def fig_sat():
    """Second domain: SAT constraint-definability class ascent (floor confined vs ascent, by CNF-width)."""
    import json
    from collections import defaultdict
    p = _cc("sat/sat_ascent.json")
    if not os.path.exists(p):
        print("  (skip fig_sat)"); return
    d = json.load(open(p)); rows = d["rows"]
    import numpy as np
    tot, fl, asc = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rows:
        w = r["cnf_width"]; tot[w] += 1
        fl[w] += int(r["floor_2cnf_realizes"]); asc[w] += int(r["ascent_realizes"])
    grades = sorted(tot)
    x = np.arange(len(grades)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.bar(x - w, [tot[g] for g in grades], w, facecolor=FAINT_F, ec=GREY, lw=0.9, label="targets")
    ax.bar(x, [fl[g] for g in grades], w, facecolor=MCF_F, ec=MCF, lw=0.9, label="floor (2-CNF) realizes")
    ax.bar(x + w, [asc[g] for g in grades], w, facecolor=CF_F, ec=CF, lw=0.9, label="ascent (general CNF) realizes")
    ax.axvline(1.5, color=NEG, ls=(0, (4, 3)), lw=0.9)
    ax.text(1.55, ax.get_ylim()[1] * 0.86, "class boundary\n(width $>2$: not\nmajority-closed)", fontsize=6.5, color=NEG)
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_xlabel("class grade: minimum CNF clause-width to define $R$", fontsize=8.5)
    ax.set_ylabel("relations", fontsize=8.5)
    ax.set_title("Second domain (SAT): floor is confined below the class boundary",
                 fontsize=9.7, weight="bold", pad=8)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_sat.pdf"); print("  wrote fig_cc_sat.pdf")


def fig_models():
    """Six frontier models as agents, under the two accountings, because the order flips between them.

    A bar chart of one accounting would rank the models, and the appendix says explicitly that we claim no
    ranking: crediting each model only on the targets it finished puts Claude-Opus-4.8 first, charging its
    timeouts as failures puts Gemini-3.1-pro first. A slope graph is the honest shape, because the crossing
    lines ARE the finding. The two models that never time out are flat by construction, which is the other
    half of it: their position is the only one that does not depend on the choice. Reads
    model_sensitivity.txt.
    """
    path = os.path.join(RUNS, "model_sensitivity.txt")
    if not os.path.exists(path):
        print("  (skip fig_models: model_sensitivity.txt not found)"); return
    rows = []
    for line in open(path):
        m = re.match(r"^\s{2}(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            name, comp, to, unan, per_c, per_60 = m.groups()
            rows.append((name, int(comp), int(to), int(unan), float(per_c), float(per_60)))
    if len(rows) != 6:
        print("  (skip fig_models: parsed %d model rows, expected 6)" % len(rows)); return

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    xs = (0.0, 1.0)
    for name, comp, to, unan, pc, p60 in rows:
        flat = (to == 0)
        col = CF if flat else MCF
        ax.plot(xs, (pc, p60), color=col, lw=1.7 if flat else 1.3,
                alpha=0.95 if flat else 0.7, zorder=3, solid_capstyle="round")
        for x, v in zip(xs, (pc, p60)):
            ax.plot([x], [v], "o", ms=5.4, mfc=tint(col, 0.30), mec=col, mew=1.2, zorder=4)
        ax.text(-0.035, pc, "%s  $%d/%d$" % (name, unan, comp), fontsize=7.4, color=INK,
                ha="right", va="center")
        ax.text(1.035, p60, "$%d/60$  %s" % (unan, name), fontsize=7.4, color=INK,
                ha="left", va="center")

    gamer = 1.0 / 60.0
    ax.axhline(gamer, color=NEG, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(0.5, gamer + 0.012, "single-oracle gamer, $1/60$", fontsize=7, color=NEG, ha="center",
            bbox=dict(fc="white", ec="none", pad=1.4))
    ax.axhspan(3 / 60, 8 / 60, color=FAINT, alpha=0.16, lw=0, zorder=0)
    ax.text(0.5, 8 / 60 + 0.008, "hand-built \\texttt{panel\\_auto}, $3$--$8/60$",
            fontsize=7, color=GREY, ha="center", bbox=dict(fc="white", ec="none", pad=1.4))

    ax.set_xlim(-0.44, 1.44); ax.set_ylim(-0.02, 0.40)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["credited on what it finished\n(unanimous / completed)",
                        "timeouts charged as failures\n(unanimous / $60$)"], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_ylabel("panel-unanimous rate", fontsize=8.5)
    ax.grid(axis="y", color=FAINT, lw=0.4, alpha=0.3, zorder=0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=CF, lw=1.7, label="no timeouts (position does not depend on the accounting)"),
                       Line2D([0], [0], color=MCF, lw=1.3, alpha=0.7, label="times out on $19$--$32$ of $60$ targets")],
              loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
              fontsize=7, handletextpad=0.6, columnspacing=1.6, handlelength=2.2)
    _save(fig, "fig_cc_models.pdf")
    print("  wrote figs/fig_cc_models.pdf")


def fig_carry():
    """Conditional carry-over with exact binomial intervals. A forest plot because the reading is
    'which intervals overlap', and six bracket pairs in a table make the reader do that by eye."""
    rows = [("agent-written operators", 57, 222, 0.257, 0.201, 0.319),
            ("$-$ objective, $-$ bias", 23, 209, 0.110, 0.071, 0.161),
            ("$-$ objective (undirected)", 24, 253, 0.095, 0.062, 0.138),
            ("panel operator", 48, 530, 0.091, 0.068, 0.118),
            ("gamer (pKiss)", 22, 643, 0.034, 0.022, 0.051),
            ("DesiRNA (min. free energy)", 1, 43, 0.023, 0.001, 0.123)]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for y, (lab, k, n, p, lo, hi) in enumerate(rows):
        hot = lab.startswith(("agent", "panel"))
        col = MCF if hot else GREY
        ax.plot([lo, hi], [y, y], color=col, lw=1.7 if hot else 1.1, solid_capstyle="butt", zorder=2)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.13, y + 0.13], color=col, lw=1.0, zorder=2)
        ax.plot([p], [y], "o", ms=7 if hot else 5.5, mfc=col, mec="white", mew=0.9, zorder=3)
        ax.text(hi + 0.011, y, f"{k}/{n}", va="center", fontsize=7.2, color=GREY)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[0].startswith(("agent", "panel")):
            t.set_color(MCF)
    ax.set_xlim(-0.005, 0.40); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("carry-over to ShapeKnots, the predictor in no arm's loop "
                  "(fraction of that arm's own successes)", fontsize=8)
    ax.grid(axis="x", color=FAINT, lw=0.4, alpha=0.45, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_cc_carry.pdf")


def fig_margin():
    """Where the headline effect sits once targets are stratified by how far random search already gets.

    The paper's largest caveat about its own positive result, and it was prose only. Parsed from the Q3 block
    of pk_margin.txt so the strata cannot drift from the committed run.
    """
    path = os.path.join(RUNS, "pk_margin.txt")
    if not os.path.exists(path):
        print("  (skip fig_margin: pk_margin.txt not found)"); return
    rows = re.findall(
        r"^\s{2}(low|mid|high)\s+Delta[^\n]*?n=(\d+)\s+panel wins\s+(\d+)\s+gamer wins\s+(\d+)"
        r"\s+paired diff\s+([+-][\d.]+)", open(path).read(), re.M)
    if len(rows) != 3:
        print("  (skip fig_margin: parsed %d strata, expected 3)" % len(rows)); return
    # collapse the middle and hardest thirds into the "Delta_bp > 0" group the text uses, so the figure and
    # Sec. 5.4 partition the 60 targets the same way: 30 targets a random compatible sequence already hits,
    # and the 30 it never does.
    lo = rows[0]
    hi_n = sum(int(r[1]) for r in rows[1:])
    hi_p = sum(int(r[2]) for r in rows[1:])
    hi_g = sum(int(r[3]) for r in rows[1:])
    rows = [("$\\Delta_{bp}=0$: a random compatible\\\\sequence already solves these",
             int(lo[1]), int(lo[2]), int(lo[3])),
            ("$\\Delta_{bp}>0$: random sampling\\\\never lands on these", hi_n, hi_p, hi_g)]
    tot = sum(r[2] for r in rows)

    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    w = 0.30
    sub = []
    for i, (lab, n, pw, gw) in enumerate(rows):
        ax.bar(i - w / 2, pw, w, color=MCF_F, edgecolor=MCF, lw=0.9, zorder=3)
        ax.bar(i + w / 2, gw, w, color=NEG_F, edgecolor=NEG, lw=0.9, zorder=3)
        ax.text(i - w / 2, pw + 0.9, "%d" % pw, ha="center", fontsize=7.6, color=INK)
        if gw:
            ax.text(i + w / 2, gw + 0.9, "%d" % gw, ha="center", fontsize=7.6, color=INK)
        sub.append("$n{=}%d$" % n)
    # Sat in the gap between the two groups on a near-horizontal leader, which reads as a floating label.
    # Placed directly over the bar it describes instead: proximity does the work and the leader goes away.
    ax.text(-w / 2, rows[0][2] + 3.2,
            "\\shortstack{$%.0f\\%%$ of every win\\\\the operator records}" % (100.0 * rows[0][2] / tot),
            fontsize=7.4, color=MCF, ha="center", va="bottom")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["\\shortstack{%s\\\\[2pt]\\footnotesize %s}" % (r[0], sub[k])
                        for k, r in enumerate(rows)], fontsize=7.6)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 53); ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel("targets won ($r_{\\mathrm{mix}}$, $15$ executions)", fontsize=8)
    ax.grid(axis="y", color=FAINT, lw=0.4, alpha=0.35, zorder=0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=MCF_F, ec=MCF, lw=0.9, label="panel operator"),
                       Patch(fc=NEG_F, ec=NEG, lw=0.9, label="gamer (pKiss only)")],
              loc="upper right", frameon=False, fontsize=7.4, handlelength=1.4)
    _save(fig, "fig_cc_margin.pdf")
    print("  wrote figs/fig_cc_margin.pdf")


def fig_paired():
    """The 15 within-cell paired differences. The sign test counts these signs; a row of 15 numbers
    in a table does not show that 14 of them point one way.

    Parsed from the per-cell McNemar block of gvp_stats.txt (b = targets only the panel operator gets,
    c = targets only the gamer gets, so b-c is the cell's paired difference) rather than transcribed. An
    earlier version of this figure hard-coded the fifteen values, which is the one thing the rest of this
    project does not permit of a number.
    """
    path = os.path.join(RUNS, "gvp_stats.txt")
    if not os.path.exists(path):
        print("  (skip fig_paired: gvp_stats.txt not found)"); return
    txt = open(path).read()
    # the r_mix section's per-cell block, which precedes the r_in section of the same shape
    body = txt.split("=== r_mix")[1].split("=== r_in")[0]
    cells = re.findall(r"s(\d)/e(\d):\s*b=(\d+),\s*c=(\d+)", body)
    if len(cells) != 15:
        print("  (skip fig_paired: expected 15 cells, parsed %d)" % len(cells)); return
    d = [int(b) - int(c) for _s, _e, b, c in sorted(cells)]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    for i, v in enumerate(d):
        col = MCF if v > 0 else NEG
        ax.plot([i, i], [0, v], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([i], [v], "o", ms=5.2, mfc=col, mec="white", mew=0.8, zorder=3)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axhline(sum(d) / len(d), color=GREY, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.text(14.6, sum(d) / len(d) + 0.3, "mean $+%.2f$" % (sum(d) / len(d)),
            fontsize=7.2, color=GREY, ha="right")
    for s in range(5):
        ax.text(3 * s + 1, 9.4, f"seed {s}", ha="center", fontsize=7.2, color=GREY)
        if s:
            ax.axvline(3 * s - 0.5, color=FAINT, lw=0.4, alpha=0.6, zorder=0)
    ax.annotate("the one execution the gamer wins", xy=(13, -1.3), xytext=(12.6, -5.2),
                fontsize=7.0, color=NEG, ha="right",
                arrowprops=dict(arrowstyle="-|>", color=NEG, lw=0.8))
    ax.set_xlim(-0.8, 14.9); ax.set_ylim(-6.6, 10.8)
    ax.set_xticks([])
    ax.set_ylabel("panel $-$ gamer,\nwithin cell ($r_{\\mathrm{mix}}$/60)", fontsize=8)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    _save(fig, "fig_cc_paired.pdf")



def fig_contrast():
    """The paper's most discriminative measurement: the same 43 targets, two objectives, and what
    survives an un-optimized predictor. A slope plot, because the reading is the divergence."""
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    arms = [("DesiRNA (min.\\ free energy)", 43, 26, MCF),
            ("invented operator (pKiss in-loop)", 43, 2, NEG)]
    for lab, a, b, col in arms:
        ax.plot([0, 1], [a, b], color=col, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot([0, 1], [a, b], "o", ms=7, mfc=col, mec="white", mew=1.0, zorder=3)
        ax.text(1.05, b, f"{b}/43" + ("  ($13\\times$ fewer)" if b == 2 else ""),
                va="center", fontsize=8.5, color=col)
        ax.text(1.05, b - 3.2, lab, va="center", fontsize=7.4, color=GREY)
    ax.text(0.0, 46.0, "43/43", ha="center", fontsize=8.5, color=INK)
    # the "same 43 targets" caveat lives in the caption; the plot area belongs to the data

    # No span annotation here. A vertical extent on a LINEAR axis is a difference (26-2=24), not a
    # ratio, so an arrow labelled "13x" would assert one quantity and draw another. The 13x lives in
    # the caption; the two endpoint labels already carry the comparison.
    ax.set_xlim(-0.18, 1.62); ax.set_ylim(-3, 50)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pKiss\n(optimized by one arm)", "ProbKnot\n(optimized by neither)"], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("designs the predictor confirms", fontsize=8)
    ax.grid(axis="y", color=FAINT, lw=0.4, alpha=0.4, zorder=0)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    _save(fig, "fig_cc_contrast.pdf")

if __name__ == "__main__":
    fig_oracle()
    fig_collapse()
    fig_attrib()
    fig_family()
    fig_grade()
    fig_grades()
    fig_invent()
    fig_stringency()
    fig_sat()
    fig_models()
    fig_carry()
    fig_contrast()
    fig_margin()
    fig_paired()
    print("wrote figs/fig_cc_*.pdf (data figs; appendix figs added when their run data is present)")
