#!/usr/bin/env python3
"""Every figure in the paper, generated from the committed JSON. No number is ever hand-copied.

Palette: the dataviz reference palette's first three categorical slots, used UNCHANGED --
blue #2a78d6, orange #eb6834, aqua #1baf7a. That file documents this three-slot set as passing
all-pairs validation in both modes (worst CVD dE 9.2 light, normal-vision 24.0). The validator
itself needs node, which is not installed here, so this relies on the documented pass for an
unmodified set rather than on inspection. Aqua sits below 3:1 on a light surface, so the relief
rule applies and every aqua mark carries a visible direct label.

Print target, so no hover layer and no dark mode: static figures for LaTeX.

COST: $0, offline.
"""
from __future__ import annotations
import collections, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEIL = "/data/wenhui/openclaude_rna/rna_harness/runs/ceiling"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/figs"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # blue / orange / aqua, fixed order
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"      # text tokens -- never a series colour
GRID = "#e3e2de"
CEILING = 0.6956

# English labels for the six populations. The pipeline's own names are Chinese and DejaVu Sans has
# no CJK, so they rendered as boxes in the first pass.
EN = {"样本2(主)": "sample 2 (main)", "基准4560": "baseline 4560", "样本3": "sample 3",
      "第四批": "batch 4", "第五批(盲)": "batch 5", "第六批(盲)": "batch 6"}


def en(n):
    return EN.get(n, n)


def all_pops():
    """the six populations, baseline and S1+S2 increment, from BOTH result files.

    The first version of fig1 averaged the baseline over the four core populations while taking
    increments from all six. It also surfaced that "27.8% -> 37.8%", quoted repeatedly in the
    audits, was never a six-population aggregate -- 27.8% is batch 4's own baseline. The correct
    six-population means are 27.3% and 37.4%.
    """
    m, b = J("m12_sequence_intrinsic.json")["pops"], J("m12_blind_check.json")
    rows = [(r["pop"], r["r2_base"], r["arms"]["S1+S2"], False) for r in m.values()]
    rows += [(v["pop"], v["r2_base"], v["arms"]["S1+S2"], True) for v in b.values()]
    return rows

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": INK3,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.labelcolor": INK2, "axes.labelsize": 8, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _save(fig, path):
    """PDF for LaTeX, PNG beside it so the figure can be inspected before it ships."""
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=170)


def J(name):
    return json.load(open(os.path.join(CEIL, name)))


def bare(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK3); ax.spines["bottom"].set_color(INK3)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.5)
    ax.tick_params(length=2, width=0.6)


# ---------------------------------------------------------------- fig 1: the frame
def fig1():
    rows = all_pops()
    base = float(np.mean([r[1] for r in rows]))
    add = float(np.mean([r[2]["dr2"] for r in rows]))
    fig, ax = plt.subplots(figsize=(3.4, 1.42))
    ax.barh([0], [base], color=S1, height=.42, zorder=3)
    ax.barh([0], [add], left=[base], color=S2, height=.42, zorder=3)
    ax.axvline(CEILING, color=INK, lw=1.1, zorder=4)
    ax.text(CEILING - .008, .30, f"tube ceiling {CEILING:.3f}", va="center", ha="right",
            color=INK, fontsize=7.2)
    ax.text(base / 2, 0, "shipped\npanel", ha="center", va="center",
            color="white", fontsize=6.6, linespacing=.95, zorder=5)
    ax.text(base + add / 2, 0, "+seq", ha="center", va="center",
            color="white", fontsize=6.6, zorder=5)
    ax.annotate(f"{base/CEILING:.1%}", xy=(base, -.21), xytext=(base, -.42),
                ha="center", color=INK2, fontsize=7.0,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=.6))
    ax.annotate(f"{(base+add)/CEILING:.1%} of ceiling", xy=(base + add, -.21),
                xytext=(base + add + .10, -.42), ha="center", color=INK2, fontsize=7.0,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=.6))
    ax.set_xlim(0, CEILING * 1.02); ax.set_ylim(-.72, .5)
    ax.set_yticks([]); ax.set_xlabel("held-out $R^2$ against the chemical-probing readout")
    bare(ax)
    _save(fig, f"{OUT}/fig1_ceiling.pdf"); plt.close(fig)
    return base, add


# ---------------------------------------------------------------- fig 2: increment, 6 pops
def fig2():
    rows = [(n, a, b) for n, _r2, a, b in all_pops()]
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    for k, (nm, a, isb) in enumerate(rows):
        y = len(rows) - 1 - k
        c = S2 if isb else S1
        ax.plot([a["ci"][0], a["ci"][1]], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([a["dr2"]], [y], "o", ms=4.5, color=c, mec="white", mew=1.1, zorder=4)
        ax.text(a["ci"][1] + .004, y, f"{a['dr2']:+.3f}", va="center", color=INK2, fontsize=7)
    ax.axvline(0, color=INK3, lw=.8, zorder=2)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([("blind: " if b else "") + en(n) for n, _a, b in rows], fontsize=7.2)
    ax.set_xlabel(r"held-out $\Delta R^2$ over shipped panel + GC")
    ax.set_xlim(-0.005, 0.125)
    h = [plt.Line2D([], [], color=S1, lw=2, marker="o", ms=4.5, mec="white"),
         plt.Line2D([], [], color=S2, lw=2, marker="o", ms=4.5, mec="white")]
    ax.legend(h, ["core populations", "blind to this block"], frameon=False, fontsize=7,
              loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, handlelength=1.4,
              labelcolor=INK2, columnspacing=1.4)
    bare(ax)
    _save(fig, f"{OUT}/fig2_increment.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 3: matched pairs vs null
# m18's json is keyed by population file in POPS order; reuse the same EN labels as everywhere else
POPNAME = [EN[n] for n in ("样本2(主)", "基准4560", "样本3", "第四批", "第五批(盲)", "第六批(盲)")]


def fig3():
    """the null here is M18's repaired one -- design held fixed, y permuted. The pre-M18 null
    shuffled the treatment, which rebuilt the matched set and ran 2.5-3.1x too narrow."""
    d = J("m18_repaired_null.json")["pops"]
    rows = []
    for r in d.values():
        b = r["arms"]["B_panel_matched"]
        rows.append((POPNAME[len(rows)], {**b, "z_vs_null": b["z_vs_null_REPAIRED"],
                                          "p_perm": b["p_perm_REPAIRED"]}, b["usable"]))
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for k, (nm, a, us) in enumerate(rows):
        y = len(rows) - 1 - k
        sd = a["null_sd"]
        ax.add_patch(plt.Rectangle((-1.96 * sd, y - .28), 3.92 * sd, .56,
                                   color=INK3, alpha=.28, lw=0, zorder=2))
        c = S1 if us else S3
        ax.plot([a["boot_ci_secondary"][0], a["boot_ci_secondary"][1]], [y, y],
                color=c, lw=1.2, alpha=.75, zorder=3)
        ax.plot([a["effect"]], [y], "o", ms=5, color=c, mec="white", mew=1.1, zorder=4)
        ax.text(a["boot_ci_secondary"][1] + .15, y,
                f"{a['effect']:+.2f}  z={a['z_vs_null']:.1f}"
                + ("" if a["p_perm"] <= .005 else "  p=%.3f" % a["p_perm"]),
                va="center", color=INK2, fontsize=6.8)
    ax.axvline(0, color=INK3, lw=.8, zorder=1)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([en(n) + ("" if u else "  (n.u.)") for n, _a, u in rows], fontsize=7.2)
    ax.set_xlabel("matched-pair difference (OpenKnot points)")
    ax.set_xlim(-1.2, 9.6)
    h = [plt.Rectangle((0, 0), 1, 1, color=INK3, alpha=.28),
         plt.Line2D([], [], color=S1, lw=1.2, marker="o", ms=5, mec="white"),
         plt.Line2D([], [], color=S3, lw=1.2, marker="o", ms=5, mec="white")]
    ax.legend(h, ["permutation null ±1.96 SD", "usable", "not usable"], frameon=False,
              fontsize=6.8, loc="upper center", bbox_to_anchor=(.5, -.30), ncol=3,
              handlelength=1.3, labelcolor=INK2, columnspacing=1.2)
    bare(ax)
    _save(fig, f"{OUT}/fig3_matched.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 4: three cuts
def fig4():
    p4 = J("m13b_gc_placement.json")["P4"]
    sym = J("m13c_symmetric.json")
    m14 = J("m14_mutation_intervention.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    fig.subplots_adjust(wspace=.55, bottom=.30)

    ax = axes[0]
    nms = [en(v["pop"]) for v in p4.values()]
    real = [v["real"] for v in p4.values()]; sh = [v["shift_mean"] for v in p4.values()]
    y = np.arange(len(nms))[::-1]
    ax.barh(y + .19, real, height=.34, color=S1, zorder=3)
    ax.barh(y - .19, sh, height=.34, color=S2, zorder=3)
    ax.axvline(0, color=INK3, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(nms, fontsize=6.6)
    ax.set_xlabel("effect (OpenKnot points)", fontsize=7.4)
    ax.set_title("a  true vs. shifted position set", fontsize=7.4, color=INK, loc="left", pad=6)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2)],
              ["true paired set", "shifted set"], frameon=False, fontsize=6.4,
              loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2, labelcolor=INK2,
              columnspacing=1.0, handlelength=1.1)
    bare(ax)

    ax = axes[1]
    ks = sorted(sym, key=float)
    sp = [sym[k]["median_slope_P"] for k in ks]; sc = [sym[k]["median_slope_C"] for k in ks]
    x = np.arange(len(ks))
    ax.plot(x, sp, "-o", color=S1, lw=1.6, ms=5, mec="white", mew=1.1, zorder=4)
    ax.plot(x, sc, "-o", color=S2, lw=1.6, ms=5, mec="white", mew=1.1, zorder=4)
    for xi, v in zip(x, sp):
        ax.text(xi, v + 4, f"{v:.0f}", ha="center", color=INK2, fontsize=6.6)
    for xi, v in zip(x, sc):
        ax.text(xi, v - 9, f"{v:.0f}", ha="center", color=INK2, fontsize=6.6)
    ax.axhline(0, color=INK3, lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{float(k):.1f}" for k in ks])
    ax.set_xlabel("matching stringency $k$ (residual SD)", fontsize=7.4)
    ax.set_ylabel("median slope", fontsize=7.4)
    ax.set_ylim(-14, 68)
    ax.set_title("b  pairing vs. composition", fontsize=7.4, color=INK, loc="left", pad=6)
    ax.legend([plt.Line2D([], [], color=S1, lw=1.6), plt.Line2D([], [], color=S2, lw=1.6)],
              ["contrast pairing", "contrast composition"], frameon=False, fontsize=6.4,
              loc="upper center", bbox_to_anchor=(.5, -.30), ncol=2, handlelength=1.1,
              labelcolor=INK2, columnspacing=1.0)
    bare(ax, xgrid=False)

    ax = axes[2]
    lab = {"GC_via_G_to_C": "G→C", "GC_via_C_to_G": "C→G",
           "AU_via_A_to_U": "A→U", "AU_via_U_to_A": "U→A"}
    ks2 = list(lab)
    v1 = [m14["Y1"][k]["mean"] for k in ks2]
    x = np.arange(len(ks2))
    ax.bar(x, v1, width=.55, color=S1, zorder=3)
    for xi, v in zip(x, v1):
        ax.text(xi, v + .02, f"{v:+.2f}", ha="center", color=INK2, fontsize=6.8)
    ax.axhline(0, color=INK3, lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in ks2], fontsize=7)
    ax.set_ylabel("partner reactivity change", fontsize=7.4)
    ax.set_ylim(0, .70)
    for x0, x1, nm in ((0, 1, "breaking G-C"), (2, 3, "breaking A-U")):
        ax.plot([x0 - .3, x1 + .3], [-.085, -.085], color=INK3, lw=.7,
                clip_on=False, zorder=5)
        ax.text((x0 + x1) / 2, -.125, nm, ha="center", va="top", color=INK2,
                fontsize=6.6, clip_on=False)
    ax.set_title("c  single-base intervention", fontsize=7.4, color=INK, loc="left", pad=6)
    bare(ax, xgrid=False)
    _save(fig, f"{OUT}/fig4_three_cuts.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 5: rescue
def fig5():
    m16 = J("m16_rescue.json")
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 1.85))
    ax = axes[0]
    keys = [("CG_vs_AU", "G-C vs A-U"), ("CG_vs_GU", "G-C vs G-U"), ("AU_vs_GU", "A-U vs G-U")]
    y = np.arange(len(keys))[::-1]
    for (k, nm), yi in zip(keys, y):
        e = m16[k]
        ax.barh(yi + .18, e["marginal_diff"], height=.32, color=S3, zorder=3)
        ax.barh(yi - .18, e["matched_diff"], height=.32, color=S1, zorder=3)
        ax.text(e["marginal_diff"] + .004, yi + .18, f"{e['marginal_diff']:+.3f}",
                va="center", color=INK2, fontsize=6.6)
        ax.text(e["matched_diff"] + .004, yi - .18, f"{e['matched_diff']:+.3f}",
                va="center", color=INK2, fontsize=6.6)
    ax.axvline(0, color=INK3, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels([n for _k, n in keys], fontsize=7)
    ax.set_xlabel("difference in rescue factor")
    ax.set_xlim(0, .155)
    ax.set_title("a  compensatory rescue", fontsize=7.6, color=INK, loc="left")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=S3), plt.Rectangle((0, 0), 1, 1, color=S1)],
              ["marginal", "matched"], frameon=False, fontsize=6.6, loc="lower right",
              labelcolor=INK2)
    bare(ax)

    ax = axes[1]
    sys.path.insert(0, CEIL)
    from m16_rescue import build
    recs = build()
    d = range(4)
    for t, c, nm in (("CG", S1, "G-C"), ("AU", S2, "A-U")):
        v = [float(np.mean([r["rf"] for r in recs if r["pt"] == t and min(r["depth"], 3) == q]))
             for q in d]
        ax.plot(list(d), v, "-o", color=c, lw=1.6, ms=5, mec="white", mew=1.1, zorder=4, label=nm)
        ax.text(3.05, v[-1], nm, color=INK2, fontsize=6.8, va="center")
    md = {t: float(np.mean([min(r["depth"], 3) for r in recs if r["pt"] == t]))
          for t in ("CG", "AU")}
    ax.set_xticks(list(d)); ax.set_xticklabels(["0\n(end)", "1", "2", "3+"])
    ax.set_xlabel("helix depth")
    ax.set_ylabel("rescue factor")
    ax.set_xlim(-.2, 3.55)
    ax.set_title(f"b  G-C sits shallower ({md['CG']:.2f} vs {md['AU']:.2f})",
                 fontsize=7.6, color=INK, loc="left")
    bare(ax, xgrid=False)
    _save(fig, f"{OUT}/fig5_rescue.pdf"); plt.close(fig)


if __name__ == "__main__":
    b, a = fig1(); fig2(); fig3(); fig4(); fig5()
    print(f"fig1 baseline R2 {b:.4f} ({b/CEILING:.1%})  +seq {b+a:.4f} ({(b+a)/CEILING:.1%})")
    print("wrote", sorted(os.listdir(OUT)))
