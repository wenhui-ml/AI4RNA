#!/usr/bin/env python3
"""Data figures for "Climbing the Chomsky Hierarchy" (matplotlib).

SCOPE: this file owns ONLY the two DATA plots -- the companion-measured oracle envelope and the measured
oracle-gaming collapse. The five schematic diagrams (architecture, certificate, hierarchy, language, worked)
are publication-grade TikZ and live in tikz/*.tex -- do NOT recreate them here (single source of truth).
Build everything with build_figs.sh. Non-math text uses plain characters; LaTeX symbols live inside $...$.
"""
import os

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


def fig_family():
    """Parameterized non-CF family L_H: capability vs crossing-stem size (n=m), by arm.
    Reads runs/cc_ascent/family_ood/family_ood.json; classifies each cell as fail / pKiss-only / partial /
    panel-unanimous. Honest binary-outcome matrix (a 'curve' would over-read binary data)."""
    import json
    jp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
         "rna_harness", "runs", "cc_ascent", "family_ood", "family_ood.json"))
    if not os.path.exists(jp):
        print("  (skip fig_family: %s not found)" % jp); return
    d = json.load(open(jp))
    diag = sorted([r for r in d["rows"] if r.get("regime") == "diag"], key=lambda r: r["n"])
    if not diag:
        print("  (skip fig_family: no diagonal rows yet)"); return
    sizes = [r["n"] for r in diag]
    _all = [("always_cheap", "CF floor (REMC, $\\Omega_{cf}$)"),
            ("panel_auto", "invented operator (autonomous)"),
            ("always_sota", "DesiRNA (SOTA, borrowed)")]
    # only plot arms that were actually run (an arm absent from every row was not part of this sweep)
    present = [(a, lbl) for (a, lbl) in _all if any(r["arms"].get(a) is not None for r in diag)]
    arms = [(a, lbl, i) for i, (a, lbl) in enumerate(present)]

    def level(cell):
        if not cell or not cell.get("solved"):
            return 0                                   # fail
        p = cell.get("panel") or {}
        if p.get("unanimous"):
            return 3                                   # panel-unanimous
        if p.get("pkiss") and p.get("probknot"):
            return 2                                   # 2/3 (pKiss & ProbKnot)
        return 1                                       # pKiss-only

    COLZ = {0: ("none", FAINT), 1: (MCF, "white"), 2: ("#c9b47e", "white"), 3: (CF, "white")}
    MK = {0: "x", 1: "o", 2: "o", 3: "o"}
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for aname, _, row in arms:
        for r in diag:
            lv = level(r["arms"].get(aname))
            fc, ec = COLZ[lv]
            if lv == 0:
                ax.plot(r["n"], row, "x", ms=6, color=FAINT, mew=1.2, zorder=3)
            else:
                ax.plot(r["n"], row, MK[lv], ms=11, mfc=fc, mec=ec, mew=1.0, zorder=3)
    ax.set_yticks([row for _, _, row in arms]); ax.set_yticklabels([lbl for _, lbl, _ in arms], fontsize=8)
    ax.set_ylim(-0.6, len(arms) - 0.4); ax.set_xlim(min(sizes) - 0.6, max(sizes) + 0.6)
    ax.set_xticks(sizes); ax.set_xlabel("crossing-stem size $n=m$ (bp) on the family $\\mathcal{L}_H$", fontsize=8.5)
    ax.set_title("Capability vs.\\ stem size on the parameterized non-CF family",
                 fontsize=10, weight="bold", pad=8)
    ax.text(sizes[0], -0.42, "CF floor is $0$ for all $n,m$ (Prop.\\,1)", fontsize=6.6, color=FAINT)
    # legend
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="x", color="w", mec=FAINT, mew=1.2, ms=7, label="not realized"),
           Line2D([0], [0], marker="o", color="w", mfc=MCF, mec="white", ms=9, label="pKiss only"),
           Line2D([0], [0], marker="o", color="w", mfc="#c9b47e", mec="white", ms=9, label="pKiss $\\cap$ ProbKnot"),
           Line2D([0], [0], marker="o", color="w", mfc=CF, mec="white", ms=9, label="panel-unanimous")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
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
    """Appendix F: thermodynamic-dominance (energy-gap) histogram of the certified positives."""
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
    ax.set_ylabel("certified designs", fontsize=8.5)
    ax.set_title("Positive-side stringency: the target is the dominant fold", fontsize=10, weight="bold", pad=8)
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
    """Cross-model (open vs closed) distribution of held-out panel-unanimous rate for the agent-invented operator."""
    import json, glob
    paths = [_cc("invent_operator/result.json")] + sorted(glob.glob(_cc("invent_operator/*/result.json")))
    OPEN = {"z-ai/glm-5.2", "deepseek/deepseek-v4-pro", "moonshotai/kimi-k3"}
    short = {"z-ai/glm-5.2": "GLM-5.2", "deepseek/deepseek-v4-pro": "DeepSeek-v4-pro", "moonshotai/kimi-k3": "Kimi-k3",
             "anthropic/claude-opus-4.8": "Claude-Opus-4.8", "openai/gpt-5.6-sol": "GPT-5.6-sol",
             "google/gemini-3.1-pro-preview": "Gemini-3.1-pro"}
    seen, rows = set(), []
    for p in paths:
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); m = r["model"]
        if m in seen or r.get("heldout_panel_unanimous") is None:
            continue
        seen.add(m); rows.append((short.get(m, m), r["heldout_panel_unanimous"], m in OPEN))
    if not rows:
        print("  (skip fig_models)"); return
    rows.sort(key=lambda x: -x[1])
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    face = [CF_F if r[2] else MCF_F for r in rows]  # open=green, closed=blue (light fill)
    edge = [CF if r[2] else MCF for r in rows]       # saturated edge
    import numpy as np
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x, vals, width=0.62, color=face, edgecolor=edge, lw=1.1, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.3, f"{v}", ha="center", fontsize=8, color=INK)
    # reference band + line drawn faintly; both explained in the legend (no floating in-plot labels)
    ax.axhspan(3, 8, color=GREY, alpha=0.16, zorder=0)
    ax.axhline(1, color=NEG, ls=(0, (4, 2)), lw=0.9, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7.4)
    ax.set_ylabel("held-out panel-unanimous ($/60$)", fontsize=8.5)
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_ylim(0, 19)
    ax.set_title("An agent invents a certificate-passing operator across the frontier",
                 fontsize=10, weight="bold", pad=8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(fc=MCF_F, ec=MCF, label="closed-weight model"),
                       Patch(fc=CF_F, ec=CF, label="open-weight model"),
                       Patch(fc=GREY, alpha=0.16, ec="none", label="hand-built panel_auto (3–8)"),
                       Line2D([0], [0], color=NEG, ls=(0, (4, 2)), lw=0.9, label="single-oracle gamer (1)")],
              frameon=False, fontsize=6.9, loc="upper right", handlelength=1.7, labelspacing=0.45,
              borderaxespad=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_models.pdf"); print("  wrote fig_cc_models.pdf")



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
    """Where the headline effect sits once targets are split by whether random search reaches them."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    groups = [("$\\Delta_{bp}=0$: 30 targets a random\ncompatible sequence already solves",
               41, 5, "paired $\\Delta=+0.080$"),
              ("$\\Delta_{bp}>0$: 30 targets random\nsampling never lands on",
               7, 0, "paired $\\Delta=+0.016$")]
    w = 0.30
    for i, (lab, panel, gamer, dlab) in enumerate(groups):
        ax.bar(i - w / 2, panel, width=w, facecolor=MCF_F, edgecolor=MCF, lw=1.1, zorder=2)
        ax.bar(i + w / 2, gamer, width=w, facecolor=NEG_F, edgecolor=NEG, lw=1.1, zorder=2)
        ax.text(i - w / 2, panel + 1.0, str(panel), ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, gamer + 1.0, str(gamer), ha="center", fontsize=9, color=INK)
        ax.text(i, max(panel, gamer) + 5.4, dlab, ha="center", fontsize=7.8, color=GREY)
    ax.text(-0.15, 36.5, "84\\% of the\nheadline effect", ha="center", va="top",
            fontsize=7.8, color=NEG)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], fontsize=7.8)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 52)
    ax.set_ylabel("discordant wins,\n15 executions", fontsize=8)
    ax.grid(axis="y", color=FAINT, lw=0.4, alpha=0.45, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = [Rectangle((0, 0), 1, 1, facecolor=MCF_F, edgecolor=MCF, lw=1.0, label="panel operator"),
           Rectangle((0, 0), 1, 1, facecolor=NEG_F, edgecolor=NEG, lw=1.0, label="gamer (pKiss only)")]
    ax.legend(handles=leg, loc="upper right", frameon=False, fontsize=7.8, handlelength=1.3)
    _save(fig, "fig_cc_margin.pdf")


def fig_paired():
    """The 15 within-cell paired differences. The sign test counts these signs; a row of 15 numbers
    in a table does not show that 14 of them point one way."""
    d = [1, 4, 6, 2, 5, 1, 9, 1, 1, 2, 5, 2, 4, -1, 1]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    for i, v in enumerate(d):
        col = MCF if v > 0 else NEG
        ax.plot([i, i], [0, v], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot([i], [v], "o", ms=5.2, mfc=col, mec="white", mew=0.8, zorder=3)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.axhline(sum(d) / len(d), color=GREY, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.text(14.6, sum(d) / len(d) + 0.3, "mean $+2.87$", fontsize=7.2, color=GREY, ha="right")
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
