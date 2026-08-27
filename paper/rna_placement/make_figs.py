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


# ---------------------------------------------------------------- fig 6: the mediation
def fig6():
    """Predicted attenuation against observed, one point per design method.

    The predicted value is beta_depth times that method's own CG-minus-AU depth gap, computed
    without reference to the observed attenuation; the two coordinates are therefore independent
    calculations and the diagonal is the claim, not a fit.
    """
    m = J("m20_designer_confounding.json")
    P = m["A3"]["points"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    lim = [min(min(p["pred"], p["obs"]) for p in P) * 1.12, 0.002]
    ax.plot(lim, lim, color=INK3, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for p in P:
        human = p["method"] in ("Eterna", "Starting sequence")
        ax.plot([p["pred"]], [p["obs"]], "o", ms=6 if human else 4.5,
                color=S2 if human else S1, mec="white", mew=1.0, zorder=3)
    for p in P:
        if p["method"] in ("Eterna", "Rosetta"):
            ax.annotate(p["method"], (p["pred"], p["obs"]), textcoords="offset points",
                        xytext=(7, -3), fontsize=6.6, color=INK2)
    a2 = m["A2"]
    ax.plot([a2["predicted_attenuation"]], [a2["observed_attenuation"]], "D", ms=6.5,
            color=S3, mec="white", mew=1.1, zorder=4)
    ax.annotate("pooled", (a2["predicted_attenuation"], a2["observed_attenuation"]),
                textcoords="offset points", xytext=(6, 5), fontsize=6.8, color=INK2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"predicted attenuation  $g_m \times \beta_{\rm depth}$")
    ax.set_ylabel("observed attenuation")
    ax.set_title(f"depth accounts for {100*a2['mediated_fraction']:.0f}% of the pooled gap",
                 fontsize=7.6, color=INK2, pad=5)
    bare(ax, xgrid=False); ax.grid(axis="x", color=GRID, lw=0.5)
    _save(fig, f"{OUT}/fig6_mediation.pdf"); plt.close(fig)
    return a2


# ---------------------------------------------------------------- fig 7: the gradient
def fig7():
    """The CG-minus-AU depth gap by design method, with the permutation null for the method
    component drawn beside it. Design-side only: no outcome variable enters this figure."""
    m = J("m20_designer_confounding.json"); a1 = m["A1"]
    pm = a1["per_method"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                                  gridspec_kw={"width_ratios": [2.05, 1]})
    y = np.arange(len(pm))[::-1]
    for k, p in enumerate(pm):
        human = p["method"] in ("Eterna", "Starting sequence")
        ax.barh(y[k], p["gap"], height=.62, color=S2 if human else S1,
                alpha=1.0 if human else .82, zorder=3)
        ax.text(p["gap"] - .022, y[k], f"{p['gap']:+.3f}", va="center", ha="right",
                fontsize=6.4, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([p["method"].replace("_", " ") + ("  (human)" if p["method"]
                        in ("Eterna", "Starting sequence") else "") for p in pm], fontsize=6.8)
    ax.set_xlabel("CG $-$ AU helix-depth gap    (more negative: G-C placed shallower)")
    ax.set_xlim(min(p["gap"] for p in pm) * 1.22, 0)
    bare(ax)
    # The run stores the null's mean and SD, not its 2000 draws, so the null is drawn as an
    # interval. A histogram here would be a shape we did not keep.
    mu, sd = a1["null_mean"], a1["null_sd"]
    ax2.add_patch(plt.Rectangle((mu - 1.96 * sd, -.3), 3.92 * sd, .6,
                                color=INK3, alpha=.30, lw=0, zorder=2))
    ax2.plot([mu], [0], "|", ms=11, color=INK2, mew=1.2, zorder=3)
    ax2.plot([a1["method_component"]], [0], "o", ms=7, color=S2, mec="white", mew=1.1, zorder=4)
    ax2.annotate(f"observed {a1['method_component']:.3f}\n$z={a1['z']:.2f}$, "
                 f"$p={a1['p_perm']:.3f}$",
                 (a1["method_component"], 0), textcoords="offset points", xytext=(0, 13),
                 ha="center", fontsize=6.6, color=INK2)
    ax2.annotate("permutation null\nmean $\\pm$ 1.96 SD", (mu, 0),
                 textcoords="offset points", xytext=(0, -20), ha="center",
                 fontsize=6.4, color=INK2)
    ax2.set_xlabel("method component,\nafter removing target effects", fontsize=7.2)
    ax2.set_ylim(-.85, .85); ax2.set_yticks([])
    bare(ax2, xgrid=False); ax2.grid(axis="x", color=GRID, lw=0.5)
    fig.tight_layout(w_pad=1.6)
    _save(fig, f"{OUT}/fig7_gradient.pdf"); plt.close(fig)
    return a1


# ---------------------------------------------------------------- fig 0: architecture
def fig0():
    """Where each result sits and what it feeds. NO MAGNITUDE APPEARS IN THIS FIGURE.

    Every other figure recomputes its coordinates from an artifact, so a printed number cannot
    drift from its run; a structure diagram has no such artifact, so it carries quantity names
    and section pointers only.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_axis_off()
    ax.set_xlim(-2, 105); ax.set_ylim(-2, 68)
    H = 12.0
    ROW = {"top": 52.0, "mid": 28.0, "bot": 4.5}
    C3 = [16.0, 50.0, 84.0]                       # top band centres
    C4 = [15.0, 39.0, 63.5, 88.0]                 # middle band centres
    CB = [24.0, 56.0, 88.0]                       # bottom band centres

    def box(cx, y, txt, fc, ec, bold=False, w=26.0, fs=6.7):
        ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, H,
                                    boxstyle="round,pad=0.0,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(cx, y + H / 2, txt, ha="center", va="center", fontsize=fs, color=INK,
                zorder=4, weight="bold" if bold else "normal", linespacing=1.42)

    def down(cx1, cx2, ytop, ybot):
        ax.add_patch(FancyArrowPatch((cx1, ytop), (cx2, ybot + H), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color=INK2, zorder=2,
                                     shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.0"))

    # band labels live in the caption; on the canvas they collided with the arrows.

    box(C3[0], ROW["top"], "designed sequences\n$\\times$ target structures\n9 design methods",
        "#ffffff", INK3)
    box(C3[1], ROW["top"], "compensatory rescue\nan intervention on one pair\nno predictor involved",
        "#eaf2fd", S1, bold=True, w=30)
    box(C3[2], ROW["top"], "the design score\npredictor panel enters\nonly as a covariate",
        "#ffffff", INK3)
    ax.add_patch(FancyArrowPatch((C3[0] + 13, ROW["top"] + H / 2), (C3[1] - 15, ROW["top"] + H / 2),
                                 arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2, zorder=2))
    ax.add_patch(FancyArrowPatch((C3[1] + 15, ROW["top"] + H / 2), (C3[2] - 13, ROW["top"] + H / 2),
                                 arrowstyle="-", mutation_scale=8, lw=0.9, color=INK3, zorder=2,
                                 ls=(0, (2, 2))))

    box(C4[0], ROW["mid"], "pair-type effect\non rescue, matched\n(Sec. 4.2)",
        "#eaf2fd", S1, bold=True, w=23)
    box(C4[1], ROW["mid"], "mediator I\nthe model's own\n$\\Delta G_{\\rm loc}$  (Sec. 4.3)",
        "#eaf2fd", S1, bold=True, w=23)
    box(C4[2], ROW["mid"], "mediator II\nplacement $g$ and\n$\\beta_{\\rm depth}$  (Sec. 4.4)",
        "#eaf2fd", S1, bold=True, w=23)
    box(C4[3], ROW["mid"], "matched-pair effect\non the design score\n(Sec. 4.5)",
        "#ffffff", INK3, w=23)
    down(C3[1] - 8, C4[0], ROW["top"], ROW["mid"])
    down(C3[1] - 2, C4[1], ROW["top"], ROW["mid"])
    down(C3[1] + 4, C4[2], ROW["top"], ROW["mid"])
    down(C3[2], C4[3], ROW["top"], ROW["mid"])

    box(CB[0], ROW["bot"], "mediates none of it\nformation is not a function\nof the local term  (Sec. 5.2)",
        "#fdf0e9", S2, bold=True, w=32, fs=6.5)
    box(CB[1], ROW["bot"], "the mediation identity\nattenuation $= g \\times \\beta_{\\rm depth}$\n(Sec. 5.1)",
        "#fdf0e9", S2, bold=True, w=32, fs=6.5)
    box(CB[2], ROW["bot"], "the gradient is a property\nof the design method\n(Sec. 4.6, 5.2)",
        "#fdf0e9", S2, w=32, fs=6.5)
    down(C4[0], CB[0] - 6, ROW["mid"], ROW["bot"])
    down(C4[1], CB[0] + 6, ROW["mid"], ROW["bot"])
    down(C4[0], CB[1] - 6, ROW["mid"], ROW["bot"])
    down(C4[2], CB[1] + 6, ROW["mid"], ROW["bot"])
    down(C4[2], CB[2] - 6, ROW["mid"], ROW["bot"])
    down(C4[3], CB[2] + 6, ROW["mid"], ROW["bot"])

    ax.text(-1, -1.5, "blue: nothing on this path calls a folding predictor", fontsize=6.3, color=S1)
    _save(fig, f"{OUT}/fig0_arch.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 8: the permutation rule
def fig8():
    """Why the null must permute the outcome. Measured, from m17d_typeI_diag.json."""
    d = J("m17d_typeI_diag.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4),
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    tot = d["total"]
    rates = [tot["A"][0] / tot["A"][1], tot["B"][0] / tot["B"][1]]
    ax.bar([0, 1], rates, width=.5, color=[S2, S1], zorder=3)
    ax.axhline(d["nominal"], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.text(1.52, d["nominal"], "  nominal 0.005", va="center", fontsize=6.6, color=INK2)
    for i, (r, t) in enumerate(zip(rates, ("A", "B"))):
        ax.text(i, r + .0026, f"{r:.4f}\n{tot[t][0]}/{tot[t][1]}", ha="center",
                fontsize=6.6, color=INK2, linespacing=1.3)
    ax.set_xticks([0, 1]); ax.set_xlim(-.6, 1.9)
    ax.set_xticklabels(["permute the\ntreatment", "permute the\noutcome"], fontsize=7)
    ax.set_ylabel("type-I error at the 0.005 gate", fontsize=7.4)
    ax.set_ylim(0, max(rates) * 1.35)
    bare(ax, xgrid=False); ax.grid(axis="y", color=GRID, lw=.5)

    pops = list(d["A"]); y = np.arange(len(pops))[::-1]
    sa = [float(np.std(d["A"][k]["eff"])) for k in pops]
    sb = [float(np.std(d["B"][k]["eff"])) for k in pops]
    ax2.barh(y + .19, sa, height=.34, color=S2, zorder=3, label="design rebuilt by the permutation")
    ax2.barh(y - .19, sb, height=.34, color=S1, zorder=3, label="design held fixed")
    ax2.set_yticks(y); ax2.set_yticklabels([en(d["A"][k]["pop"]) if "pop" in d["A"][k]
                                            else k for k in pops], fontsize=6.6)
    ax2.set_xlabel("SD of the statistic under each null", fontsize=7.4)
    ax2.legend(frameon=False, fontsize=6.4, loc="lower right", labelcolor=INK2)
    bare(ax2)
    fig.tight_layout(w_pad=1.4)
    _save(fig, f"{OUT}/fig8_null.pdf"); plt.close(fig)



# ---------------------------------------------------------------- fig 9: the energy test
def fig9():
    """M21. Left: where the two pair types can share an NN local free energy at all.
    Centre: what conditioning on it does, against what conditioning on a placebo does.
    Right: why the product-of-coefficients route disagrees."""
    d = J("m21_thermo_mediator.json")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.4, 2.5))

    edges = d["bin_edges"]
    lab = ["$\\leq$%.1f" % edges[0]] + \
          ["%.1f..%.1f" % (edges[i], edges[i + 1]) for i in range(len(edges) - 1)] + \
          ["$>$%.1f" % edges[-1]]
    x = np.arange(len(lab))
    for k, (t, col) in enumerate((("CG", S1), ("AU", S2), ("GU", S3))):
        h = d["dG_by_type"][t]["bin_hist"]
        a1.bar(x + (k - 1) * .27, h, width=.26, color=col, zorder=3, label=t.replace("CG", "C--G")
               .replace("AU", "A--U").replace("GU", "G--U"))
    a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=5.8, rotation=30, ha="right")
    a1.set_xlabel("$\\Delta G_{\\mathrm{loc}}$ bin (kcal/mol)", fontsize=7.4)
    a1.set_ylabel("target pairs", fontsize=7.4)
    a1.legend(frameon=False, fontsize=6.4, labelcolor=INK2)
    a1.set_title("where the types overlap", fontsize=7.6, color=INK)
    bare(a1, xgrid=False)

    pm = d["P3"]["placebo_retention_mean"] * 100
    ps = d["P3"]["placebo_retention_sd"] * 100
    re = d["B1"]["retention"] * 100
    a2.axvspan(pm - 1.96 * ps, pm + 1.96 * ps, color="#f1f0ec", zorder=1)
    a2.axvline(pm, color=INK3, lw=.8, ls=(0, (3, 2)), zorder=2)
    a2.axvline(100, color=INK2, lw=.7, zorder=2)
    ys = [3, 2, 1, 0]
    vals = [re, d["sensitivity_bins"]["3"]["retention"] * 100,
            d["sensitivity_bins"]["8"]["retention"] * 100, pm]
    names = ["real, %d bins" % d["nbins"], "real, 3 bins", "real, 8 bins", "placebo mediator"]
    cols = [S1, S1, S1, INK3]
    a2.scatter(vals, ys, s=26, color=cols, zorder=4)
    a2.errorbar([pm], [0], xerr=[1.96 * ps], color=INK3, lw=1.0, capsize=2, zorder=3)
    a2.set_yticks(ys); a2.set_yticklabels(names, fontsize=6.6)
    a2.set_xlabel("contrast retained after conditioning (%)", fontsize=7.4)
    a2.set_title("conditioning on the model's own energy", fontsize=7.6, color=INK)
    a2.set_ylim(-.7, 3.7)
    bare(a2)

    bw = d["B2"]["binwise"]
    gx = [b["dG_mean"] for b in bw]; gy = [b["rf_centred"] for b in bw]
    a3.axhline(0, color=INK3, lw=.7)
    a3.plot(gx, gy, "-o", color=S1, ms=4, lw=1.1, zorder=3)
    xs = np.linspace(min(gx), max(gx), 20)
    a3.plot(xs, d["B2"]["beta_dG"] * (xs - np.mean(gx)), color=S2, lw=1.0,
            ls=(0, (4, 2)), zorder=2, label="the linear fit $\\beta_{\\Delta G}$")
    a3.set_xlabel("$\\Delta G_{\\mathrm{loc}}$ (kcal/mol)", fontsize=7.4)
    a3.set_ylabel("rescue, centred in stratum", fontsize=7.4)
    a3.legend(frameon=False, fontsize=6.4, labelcolor=INK2, loc="lower left")
    a3.set_title("the assumption the linear route needs", fontsize=7.6, color=INK)
    bare(a3, xgrid=False)

    fig.tight_layout(w_pad=1.3)
    _save(fig, f"{OUT}/fig9_energy.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 10: depth and method
def fig10():
    """Left: the contrast by helix depth -- the terminal-penalty domain is depth 0 only.
    Right: the contrast method by method, which is the axis the survivorship objection runs on."""
    d3 = J("m21_thermo_mediator.json")["B3"]
    a22 = J("m22_review_arms.json")["A"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6),
                                 gridspec_kw={"width_ratios": [1, 1.55]})
    ks = ["depth0", "depth1", "depth2plus"]
    nm = ["0\n(helix end)", "1", "$\\geq$2\n(interior)"]
    v = [d3[k]["diff"] for k in ks]
    a1.axhline(0, color=INK2, lw=.7)
    bars = a1.bar(np.arange(3), v, width=.55,
                  color=[S1 if d3[k]["p_perm"] <= .005 else INK3 for k in ks], zorder=3)
    for i, k in enumerate(ks):
        a1.text(i, v[i] + .004, "$p=%.4f$" % d3[k]["p_perm"], ha="center", fontsize=6.2,
                color=INK2)
    a1.set_xticks(np.arange(3)); a1.set_xticklabels(nm, fontsize=6.8)
    a1.set_ylabel("C--G $-$ A--U rescue", fontsize=7.4)
    a1.set_xlabel("helix depth", fontsize=7.4)
    a1.set_title("not confined to the penalty's domain", fontsize=7.6, color=INK)
    a1.set_ylim(-.005, max(v) * 1.28)
    bare(a1, xgrid=False)

    rows = a22["per_method"]
    y = np.arange(len(rows))[::-1]
    a2.axvline(0, color=INK2, lw=.7)
    a2.scatter([r["diff"] for r in rows], y, s=[max(8, r["weight"] / 6) for r in rows],
               color=[S1 if r["diff"] > 0 else S2 for r in rows], zorder=4)
    a2.set_yticks(y)
    a2.set_yticklabels([r["method"] for r in rows], fontsize=6.2)
    a2.set_xlabel("C--G $-$ A--U rescue, matched within design", fontsize=7.4)
    a2.set_title("%d of %d methods positive, sign test $p=%.4f$"
                 % (a22["n_positive"], a22["n_methods"], a22["binom_p"]), fontsize=7.6, color=INK)
    bare(a2)
    fig.tight_layout(w_pad=1.4)
    _save(fig, f"{OUT}/fig10_depth_method.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 11: not one line
def fig11():
    """If formation were a function of the local NN energy, the three pair-type contrasts would
    lie on one line through the origin whose slope is the directly estimated beta."""
    d = J("m21_thermo_mediator.json")
    b4, col = d["B4"], d["B4_collinearity"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(0, color=INK3, lw=.7); ax.axvline(0, color=INK3, lw=.7)
    xs = np.linspace(-2.7, 0.15, 20)
    ax.plot(xs, d["B2"]["beta_dG"] * xs, color=INK3, lw=1.0, ls=(0, (4, 2)), zorder=2,
            label="$\\beta_{\\Delta G}$, estimated directly")
    for (k, lab, c) in (("CG_vs_AU", "C--G $-$ A--U", S1), ("CG_vs_GU", "C--G $-$ G--U", S2),
                        ("AU_vs_GU", "A--U $-$ G--U", S3)):
        v = b4[k]
        ax.scatter([v["dG_diff"]], [v["rescue_diff"]], s=34, color=c, zorder=4, label=lab)
        ax.plot([0, v["dG_diff"]], [0, v["rescue_diff"]], color=c, lw=.8, alpha=.55, zorder=3)
    ax.set_xlabel("$\\Delta\\Delta G_{\\mathrm{loc}}$ between the types (kcal/mol)", fontsize=7.4)
    ax.set_ylabel("$\\Delta$ rescue, matched", fontsize=7.4)
    ax.set_title("implied slopes spread $%.2f\\times$" % col["spread_ratio"],
                 fontsize=7.6, color=INK)
    ax.legend(frameon=False, fontsize=6.2, labelcolor=INK2, loc="lower left")
    bare(ax, xgrid=False)
    fig.tight_layout()
    _save(fig, f"{OUT}/fig11_collinear.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig0(); fig8()
    b, a = fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    fig9(); fig10(); fig11()
    print(f"fig1 baseline R2 {b:.4f} ({b/CEILING:.1%})  +seq {b+a:.4f} ({(b+a)/CEILING:.1%})")
    print("wrote", sorted(os.listdir(OUT)))
