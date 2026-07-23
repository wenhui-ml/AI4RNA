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
    arms = [("always_cheap", "CF floor (REMC, $\\Omega_{cf}$)", 0),
            ("panel_auto", "invented operator (autonomous)", 1),
            ("always_sota", "DesiRNA (SOTA, borrowed)", 2)]

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
    ax.set_ylim(-0.6, 2.6); ax.set_xlim(min(sizes) - 0.6, max(sizes) + 0.6)
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
    ax.set_title("Formal crossing complexity predicts the capability frontier",
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
    ax.bar(x - w / 2, [pb.get(g, 0) for g in grades], w, color=MCF, ec="white", lw=0.5, label="Pseudobase++ held-out ($n{=}60$)")
    ax.bar(x + w / 2, [br.get(g, 0) for g in grades], w, color=CF, ec="white", lw=0.5, label="bpRNA cross-source ($n{=}60$)")
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
    ax.hist(gaps, bins=bins, color=CF, ec="white", lw=0.7)
    ax.axvline(1.0, color=NEG, ls=(0, (4, 2)), lw=1.0)
    ax.text(1.03, ax.get_ylim()[1] * 0.9, "dominance\nmargin $1$ kcal", fontsize=6.8, color=NEG)
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
    ax.bar(x - w, [tot[g] for g in grades], w, color=FAINT, ec="white", lw=0.5, label="targets")
    ax.bar(x, [fl[g] for g in grades], w, color=MCF, ec="white", lw=0.5, label="floor (2-CNF) realizes")
    ax.bar(x + w, [asc[g] for g in grades], w, color=CF, ec="white", lw=0.5, label="ascent (general CNF) realizes")
    ax.axvline(1.5, color=NEG, ls=(0, (4, 3)), lw=0.9)
    ax.text(1.55, ax.get_ylim()[1] * 0.86, "class boundary\n(width $>2$: not\nmajority-closed)", fontsize=6.5, color=NEG)
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_xlabel("class grade: minimum CNF clause-width to define $R$", fontsize=8.5)
    ax.set_ylabel("relations", fontsize=8.5)
    ax.set_title("Second domain (SAT): floor is confined below the class boundary",
                 fontsize=9.7, weight="bold", pad=8)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=3); _save(fig, "fig_cc_sat.pdf"); print("  wrote fig_cc_sat.pdf")


if __name__ == "__main__":
    fig_oracle()
    fig_collapse()
    fig_family()
    fig_grade()
    fig_grades()
    fig_invent()
    fig_stringency()
    fig_sat()
    print("wrote figs/fig_cc_*.pdf (data figs; appendix figs added when their run data is present)")
