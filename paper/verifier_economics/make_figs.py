"""Generate all paper figures from the experiment JSONs. Clean, colorblind-safe, vector PDF.
Figures: ladder schematic, cross-domain frontier, pseudoknot wrong-oracle (99.6%), controlled Goodhart
with CI bands, cost ladder, routing risk-frontier. All numbers from rna_harness/runs/*."""
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RUNS = "/data/wenhui/openclaude_rna/rna_harness/runs"
FIGS = os.path.dirname(__file__) + "/figs"
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     # serif (STIX) to match the LaTeX (Computer Modern) body; math in the same family
                     "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
C = {"match": "#4C72B0", "umfe": "#DD8452", "pfe": "#C44E52", "cheap": "#4C72B0", "strong": "#C44E52",
     "sat": "#55A868", "gold": "#C44E52", "branch": "#8172B3", "gray": "#7f7f7f"}


def _load(p):
    return json.load(open(os.path.join(RUNS, p)))


# ---- F0: the verifier graph (conceptual schematic) ------------------------------------------------
def f0():
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    # nested strictness ladder as concentric rounded boxes (match superset of uMFE superset of pfe)
    boxes = [("match", 0.3, 0.3, 4.6, 3.4, C["match"], r"$\mathcal{A}_{\mathrm{match}}$: fold$=T$"),
             ("umfe", 0.9, 0.85, 3.4, 2.3, C["umfe"], r"$\mathcal{A}_{\mathrm{uMFE}}$: unique MFE"),
             ("pfe", 1.6, 1.45, 2.0, 1.1, C["pfe"], r"$\mathcal{A}_{\mathrm{pfe}}$: $P(T)\!\geq\!\theta$")]
    for _, x, y, w, h, col, lab in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fill=True, fc=col, ec="white", alpha=0.28, lw=1.5))
    ax.text(2.6, 3.35, r"$\mathcal{A}_{\mathrm{match}}$", ha="center", fontsize=9, color=C["match"])
    ax.text(2.6, 2.55, r"$\mathcal{A}_{\mathrm{uMFE}}$", ha="center", fontsize=9, color="#a5591f")
    ax.text(2.6, 1.95, r"$\mathcal{A}_{\mathrm{pfe}}(\theta)$", ha="center", fontsize=9, color=C["pfe"])
    ax.text(2.6, 0.05, "nested strictness ladder (one oracle)", ha="center", fontsize=8, color=C["gray"])
    ax.text(2.6, 3.75, r"$\mathcal{A}_{\mathrm{pfe}}\subseteq\mathcal{A}_{\mathrm{uMFE}}\subseteq\mathcal{A}_{\mathrm{match}}\ (\theta>\frac{1}{2})$",
            ha="center", fontsize=8.5)
    # the oracle branch (pseudoknots) -- a separate box, NOT a superset
    ax.add_patch(FancyBboxPatch((6.2, 1.2), 3.4, 1.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fill=True, fc=C["branch"], ec="white", alpha=0.3, lw=1.5))
    ax.text(7.9, 2.4, "pKiss oracle", ha="center", fontsize=9, color=C["branch"])
    ax.text(7.9, 1.95, "(crossing pairs)", ha="center", fontsize=8, color=C["branch"])
    ax.text(7.9, 1.45, "separate branch", ha="center", fontsize=7.5, color=C["gray"])
    ax.text(7.9, 0.75, "wrong oracle here", ha="center", fontsize=8, color=C["gold"])
    ax.text(7.9, 0.45, r"$\Rightarrow 99.6\%$ false-accept", ha="center", fontsize=8, color=C["gold"])
    # arrow: abstain / domain-of-validity gate between ladder and branch
    ar = FancyArrowPatch((5.0, 2.0), (6.15, 2.0), arrowstyle="-|>", mutation_scale=12,
                         color=C["gray"], lw=1.3)
    ax.add_patch(ar)
    ax.text(5.6, 2.25, "abstain if\nout of domain", ha="center", fontsize=7, color=C["gray"])
    fig.tight_layout(); fig.savefig(FIGS + "/f0_ladder.pdf"); plt.close(fig)
    print("f0_ladder.pdf")


# ---- F1: cross-domain verifier gap ----------------------------------------------------------------
def f1():
    fr = _load("verifier_frontier_multiseed/frontier.json")["agg"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    rungs = ["match", "umfe", "pfe"]
    surv = [fr["survival"][r] * 100 for r in rungs]
    ci = [fr["survival_ci"][r] for r in rungs]
    err = [[surv[i] - ci[i][0] for i in range(3)], [ci[i][1] - surv[i] for i in range(3)]]
    ax[0].bar(range(3), surv, color=[C[r] for r in rungs], yerr=err, capsize=3, ecolor="#333")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["match", "uMFE", "pfe"])
    ax[0].set_ylabel("acceptance survival (%)"); ax[0].set_title("(a) RNA: strength ladder")
    for i, v in enumerate(surv):
        ax[0].text(i, min(v + 4, 104), f"{v:.0f}", ha="center", fontsize=8)
    ax[0].set_ylim(0, 112)
    order = ["<50", "50-99", "100-199", ">=200"]
    bins = fr["bins"]
    xs = [b for b in order if b in bins]
    pt = [bins[b]["mean_pT"] for b in xs]
    ax[1].plot(range(len(xs)), pt, "o-", color=C["pfe"])
    ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(xs, rotation=20, fontsize=7)
    ax[1].set_ylabel("mean $P$(target)"); ax[1].set_xlabel("target length")
    ax[1].set_title("(b) RNA robustness"); ax[1].set_ylim(0, 1)
    ax[1].axhline(0.9, ls="--", lw=0.8, color="gray"); ax[1].text(0, 0.92, "pfe $\\theta$=0.9", fontsize=6, color="gray")
    rows = _load("sat_exhaustive/sat_exhaustive.json")["rows"]   # exhaustive: rho over ALL solutions, no solver
    al = sorted(set(r["alpha"] for r in rows))
    med = [st.median([r["median_rho_over_solutions"] for r in rows
                      if r["alpha"] == a and r["median_rho_over_solutions"] is not None] or [0]) for a in al]
    ax[2].plot(al, med, "s-", color=C["sat"])
    ax[2].set_xlabel(r"clause density $\alpha$"); ax[2].set_ylabel(r"median $\rho$ over solutions")
    ax[2].set_title("(c) SAT robustness (exhaustive)"); ax[2].set_ylim(0, max(med) * 1.3)
    ax[2].axvline(4.267, ls="--", lw=0.8, color="gray"); ax[2].text(4.27, max(med) * 1.1, "3-SAT\nthr.", fontsize=6, color="gray")
    fig.tight_layout(); fig.savefig(FIGS + "/f1_frontier.pdf"); plt.close(fig)
    print("f1_frontier.pdf")


# ---- F2: the wrong oracle (99.6% pseudoknot false accepts) ----------------------------------------
def f_pk():
    d = _load("track_b_pk_full/fp_demo.json")
    rows = d["rows"]; N = d["n"]; FP = d["false_positives"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    # (a) fp-rate vs pseudoknot complexity (n_pk_pairs bins)
    def b(k):
        return "1-2" if k <= 2 else "3-4" if k <= 4 else "5-8" if k <= 8 else "≥9"
    bins = defaultdict(lambda: [0, 0])
    for r in rows:
        g = bins[b(r["n_pk_pairs"])]; g[0] += 1; g[1] += int(r["false_positive"])
    order = ["1-2", "3-4", "5-8", "≥9"]
    xs = [k for k in order if k in bins]
    rate = [100 * bins[k][1] / bins[k][0] for k in xs]
    ns = [bins[k][0] for k in xs]
    ax[0].bar(range(len(xs)), rate, color=C["gold"])
    for i, (rt, nn) in enumerate(zip(rate, ns)):
        ax[0].text(i, min(rt + 1.5, 100), f"{rt:.0f}%\n(n={nn})", ha="center", fontsize=7)
    ax[0].set_xticks(range(len(xs))); ax[0].set_xticklabels(xs)
    ax[0].set_xlabel("crossing pairs in target"); ax[0].set_ylabel("MFE-oracle false-accept (%)")
    ax[0].set_title(f"(a) wrong oracle: {FP}/{N} = {100*FP/N:.1f}%"); ax[0].set_ylim(0, 112)
    # (b) scatter: length vs crossing-pair count, colored by false positive
    fp_x = [r["n"] for r in rows if r["false_positive"]]; fp_y = [r["pkiss_bp"] for r in rows if r["false_positive"]]
    ok_x = [r["n"] for r in rows if not r["false_positive"]]; ok_y = [r["pkiss_bp"] for r in rows if not r["false_positive"]]
    ax[1].scatter(fp_x, fp_y, s=12, color=C["gold"], alpha=0.6, label=f"false accept ({len(fp_x)})")
    ax[1].scatter(ok_x, ok_y, s=30, color=C["sat"], marker="D", label=f"abstain/correct ({len(ok_x)})")
    ax[1].set_xlabel("target length $n$"); ax[1].set_ylabel("crossing bp (pKiss)")
    ax[1].set_title("(b) every complexity level fails"); ax[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS + "/f_pk.pdf"); plt.close(fig)
    print("f_pk.pdf")


# ---- F3: controlled Goodhart (RNA) with bootstrap CI bands ----------------------------------------
def f2():
    d = _load("goodhart_multiseed/goodhart_multiseed.json")
    rows, budgets, th = d["rows"], d["budgets"], d["theta"]
    def rate(arm, B, pred):
        u = [r for r in rows if r["arm"] == arm and r["budget"] == B]
        return 100 * sum(pred(r) for r in u) / len(u)
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    # cheap arm: proxy (exact-match) RISES while robustness stalls/falls -> Goodhart divergence
    cheap_proxy = [rate("cheap", B, lambda r: r["bp"] == 0) for B in budgets]
    cheap_rob = [rate("cheap", B, lambda r: r["p_target"] >= th) for B in budgets]
    strong_rob = [rate("strong", B, lambda r: r["p_target"] >= th) for B in budgets]
    ax.plot(budgets, cheap_proxy, "o--", color=C["cheap"], label="cheap arm: proxy (exact-match)")
    ax.plot(budgets, cheap_rob, "o-", color=C["gold"], label="cheap arm: robustness")
    ax.plot(budgets, strong_rob, "s-", color=C["sat"], label="strong arm: robustness")
    ax.annotate("proxy↑", (budgets[-1], cheap_proxy[-1]), fontsize=7, color=C["cheap"], ha="right", va="bottom")
    ax.annotate("robustness↓", (budgets[-1], cheap_rob[-1]), fontsize=7, color=C["gold"], ha="right", va="top")
    ax.set_xscale("log"); ax.set_xlabel("optimization budget (evals)")
    ax.set_ylabel("acceptance rate (%)"); ax.set_ylim(0, 80)
    ax.set_title("RNA Goodhart: proxy–objective divergence"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIGS + "/f2_goodhart.pdf"); plt.close(fig)
    print("f2_goodhart.pdf")


# ---- F4: cost ladder (RNA + SAT) ------------------------------------------------------------------
def f3():
    rna = _load("verifier_cost/cost.json")["per"]
    sat = _load("sat_verifier_deep/cost.json")["per"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for r, lab in [("match", "match"), ("umfe", "uMFE"), ("pfe", "pfe")]:
        ax[0].plot(rna[r]["n"], [m * 1e3 for m in rna[r]["med"]], "o-", color=C[r], label=lab, ms=3)
    if rna.get("pkiss", {}).get("n"):
        ax[0].plot(rna["pkiss"]["n"], [m * 1e3 for m in rna["pkiss"]["med"]], "^--", color="gray", label="pKiss (branch)", ms=3)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("length n")
    ax[0].set_ylabel("verify wall (ms)"); ax[0].set_title("(a) RNA: cost per rung"); ax[0].legend(frameon=False, fontsize=7)
    labs = {"V0": "V0 (satisfy)", "rho(r=1)": r"$\rho$ (r=1)", "V(r=2)": "r=2 ball"}
    cols = {"V0": C["match"], "rho(r=1)": C["umfe"], "V(r=2)": C["pfe"]}
    for r in ["V0", "rho(r=1)", "V(r=2)"]:
        ns = [t[0] for t in sat[r]]; ms = [t[1] * 1e3 for t in sat[r]]
        ax[1].plot(ns, ms, "s-", color=cols[r], label=labs[r], ms=3)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("n variables")
    ax[1].set_ylabel("verify wall (ms)"); ax[1].set_title("(b) SAT: cost per rung"); ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS + "/f3_cost.pdf"); plt.close(fig)
    print("f3_cost.pdf")


# ---- F5: routing risk frontier --------------------------------------------------------------------
def f_routing():
    r = _load("routing_reject_heavy/routing.json")          # verified reject-heavy workload
    ag = r["sound"]["always_gold_ms"]                       # normalize away workload-specific ms units
    fr = sorted(r["frontier"], key=lambda x: x["epsilon"])
    eps = [p["epsilon"] for p in fr]
    frac = [p["E_cost_ms"] / ag for p in fr]                # E[cost] as a fraction of always-gold
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    ax.axhline(1.0, ls=":", lw=1.0, color=C["gray"]); ax.text(0.5, 1.02, "always-gold", fontsize=6.5, color=C["gray"])
    ax.plot(eps, frac, "o-", color=C["gold"], label="risk-calibrated stop")
    ax.set_xlabel(r"tolerated FP rate $\epsilon$")
    ax.set_ylabel("E[cost] / always-gold cost")
    ax.set_title("Robustness cost is workload-incompressible")
    ax.text(0.03, frac[0] + 0.06, "flat at gold for all $\\epsilon\\leq0.99$:\n"
            "match-accept FP $=10/10=100\\%$, so no cheaper\nrung is admissible; savings are reject-only",
            fontsize=6.0, color=C["gray"])
    ax.set_ylim(0, 1.15)
    fig.tight_layout(); fig.savefig(FIGS + "/f_routing.pdf"); plt.close(fig)
    print("f_routing.pdf")


# ---- F7: LLM code-verifier cross-domain validation (multi-model + provided-k ablation) -----------
_LLM_MODELS = [("gpt-3.5", "llm_gh_openai_gpt-3.5-turbo"),
               ("llama-3.1-8b", "llm_gh_meta-llama_llama-3.1-8b-instruct"),
               ("qwen-coder-32b", "llm_gh_qwen_qwen-2.5-coder-32b-instruct"),
               ("gpt-4o-mini", "llm_gh_openai_gpt-4o-mini")]


def f_llm():
    import numpy as np, json
    EP = RUNS + "/llm_evalplus_v2"
    R = json.load(open(EP + "/analysis_v2.json"))   # reproducible rerun: fixed pre-registered split, all artifacts saved
    order = ["llama-3.1-8b", "qwen36-local"]        # weak (OpenRouter open) -> strong (local pinned)
    lab = {"llama-3.1-8b": "llama-3.1-8b (weak)", "qwen36-local": "qwen3.6-27b (strong)"}
    cols = {"llama-3.1-8b": C["gold"], "qwen36-local": C["sat"]}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    # (a) risk-coverage-cost: as the cheap rung tightens (more base tests), coverage falls and risk falls.
    for nm in order:
        rc = R[nm]["risk_coverage"]; qs = sorted(float(q) for q in rc)
        cov = [100 * rc[f"{q:.1f}"]["coverage"] for q in qs]
        risk = [100 * rc[f"{q:.1f}"]["risk_cond_fp"] for q in qs]
        ax[0].plot(cov, risk, "o-", ms=3, color=cols[nm], label=lab[nm])
        for q in (0.1, 0.5, 1.0):                    # annotate a few rungs with #tests
            k = f"{q:.1f}"
            ax[0].annotate(f"{rc[k]['mean_tests']:.0f}t", (100 * rc[k]["coverage"], 100 * rc[k]["risk_cond_fp"]),
                           fontsize=5.2, color=cols[nm], xytext=(2, 2), textcoords="offset points")
    ax[0].set_xlabel("coverage (% candidates accepted)"); ax[0].set_ylabel("risk: cond.\\ false-accept (%)")
    ax[0].set_title("(a) risk--coverage--cost (dial the rung)")
    ax[0].legend(frameon=False, fontsize=6.3)
    # (b) Goodhart under best-of-12 on HELD-OUT GOLD: cheap vs strong selection (null)
    x = np.arange(len(order))
    cg = [R[nm]["cheap_gold"][0] for nm in order]; sg = [R[nm]["strong_gold"][0] for nm in order]
    ax[1].bar(x - 0.19, cg, 0.38, color=C["gold"], label="cheap-selected")
    ax[1].bar(x + 0.19, sg, 0.38, color=C["sat"], label="strong-selected")
    for i, nm in enumerate(order):
        ax[1].annotate(f"$+{sg[i]-cg[i]:.1f}$", (i, max(cg[i], sg[i]) + 1.5), ha="center", fontsize=6.0)
    ax[1].set_xticks(x); ax[1].set_xticklabels([lab[n] for n in order], rotation=15, fontsize=6.3, ha="right")
    ax[1].set_ylabel("gold robustness (%)"); ax[1].set_ylim(40, 100)
    ax[1].set_title("(b) best-of-12 on held-out gold (null)")
    ax[1].legend(frameon=False, fontsize=6.3, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS + "/f_llm.pdf"); plt.close(fig)
    print("f_llm.pdf")


def f_framework():
    """Graphical abstract: the paper's pipeline as a 4-stage flow.
    problem -> a family of decidable verifiers (ladder + branch) -> two over-acceptance axes ->
    measured cost-robustness frontier + sound routing."""
    fig, ax = plt.subplots(figsize=(7.4, 2.05))
    ax.set_xlim(0, 16); ax.set_ylim(0.72, 5.15); ax.axis("off")
    YB, YT = 1.5, 5.0; YM = (YB + YT) / 2

    def panel(x0, x1, col):
        for f, a, lw in ((True, 0.09, 0), (False, 1.0, 1.7)):
            ax.add_patch(FancyBboxPatch((x0, YB), x1 - x0, YT - YB, boxstyle="round,pad=0.015,rounding_size=0.13",
                                        fill=f, fc=col, ec=col, alpha=a, lw=lw))

    def title(xc, t, col="#1a1a2e"):
        ax.text(xc, YT - 0.42, t, ha="center", va="center", fontsize=8, weight="bold", color=col)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, YM), (x1, YM), arrowstyle="-|>", mutation_scale=14,
                                     color=C["gray"], lw=1.8))

    B1, B2, B3, B4 = (0.4, 3.3), (3.8, 7.0), (7.5, 11.0), (11.5, 15.6)
    c1, c2, c3, c4 = 1.85, 5.4, 9.25, 13.55
    for (x0, x1), col in ((B1, C["gray"]), (B2, C["match"]), (B3, C["gold"]), (B4, C["sat"])):
        panel(x0, x1, col)
    arrow(B1[1], B2[0]); arrow(B2[1], B3[0]); arrow(B3[1], B4[0])

    # Panel 1: one problem
    title(c1, "One problem")
    ax.text(c1, 3.55, r"target $T$", ha="center", va="center", fontsize=9.5)
    ax.text(c1, 2.75, "RNA pseudoknot", ha="center", va="center", fontsize=7, color=C["gray"])
    ax.text(c1, 2.35, "or SAT formula", ha="center", va="center", fontsize=7, color=C["gray"])

    # Panel 2: family of verifiers (nested ladder icon + pKiss branch)
    title(c2, "Verifier family")
    for (dx, dy, w, h, c) in [(4.15, 2.05, 1.9, 1.9, C["match"]), (4.45, 2.35, 1.3, 1.3, C["umfe"]),
                              (4.72, 2.62, 0.76, 0.76, C["pfe"])]:
        ax.add_patch(FancyBboxPatch((dx, dy), w, h, boxstyle="round,pad=0.008,rounding_size=0.06",
                                    fill=True, fc=c, ec="white", alpha=0.32, lw=1.1))
    ax.text(5.1, 4.08, "nested rungs", ha="center", va="center", fontsize=6.3, color=C["match"])
    ax.add_patch(FancyBboxPatch((6.2, 2.35), 0.65, 1.35, boxstyle="round,pad=0.008,rounding_size=0.06",
                                fill=True, fc=C["branch"], ec="white", alpha=0.36, lw=1.1))
    ax.text(6.52, 3.95, "pKiss", ha="center", va="center", fontsize=5.8, color=C["branch"])
    ax.text(6.52, 2.08, "branch", ha="center", va="center", fontsize=5.8, color=C["branch"])
    ax.text(c2, 1.78, r"$\mathrm{match}\supseteq\mathrm{uMFE}\supseteq\mathrm{pfe}$",
            ha="center", va="center", fontsize=6.2, color=C["gray"])
    ax.text(c2, 1.02, "“solved?” = which node you pick", ha="center", va="center",
            fontsize=7, style="italic", color=C["gold"])

    # Panel 3: two over-acceptance axes
    title(c3, "Over-acceptance")
    ax.text(c3, 3.9, "Axis 1: wrong oracle", ha="center", va="center", fontsize=7.2, color=C["gold"])
    ax.text(c3, 3.5, r"$99.6\%$ / $28\%$ false accepts", ha="center", va="center", fontsize=6.6, color=C["pfe"])
    ax.text(c3, 2.8, "Axis 2: too-weak rung", ha="center", va="center", fontsize=7.2, color=C["gold"])
    ax.text(c3, 2.4, r"$92\%$ collapse", ha="center", va="center", fontsize=6.6, color=C["pfe"])
    ax.text(c3, 1.82, r"determinism $\neq$ safety", ha="center", va="center", fontsize=6.6, style="italic", color=C["gray"])

    # Panel 4: priced frontier (curve above, labels below, both inside)
    title(c4, "Priced frontier")
    xs = [12.05, 12.75, 13.5, 14.25, 15.0]; ys = [3.0, 3.45, 3.75, 3.95, 4.05]
    ax.plot(xs, ys, "o-", color=C["sat"], ms=3, lw=1.7)
    ax.text(c4, 2.4, r"robustness $\uparrow$ vs cost $\uparrow$", ha="center", va="center", fontsize=6.6, color=C["gray"])
    ax.text(c4, 1.92, "sound cheapest-first cascade", ha="center", va="center", fontsize=6.4, color=C["gray"])

    fig.savefig(FIGS + "/f_framework.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS + "/f_framework.png", dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("f_framework.pdf")


def f_projection():
    """Axis-1 mechanism: specification projection. A crossing target is silently projected onto its nested
    skeleton by the blind MFE oracle; the design 'solves' the skeleton while the crossing stem never forms."""
    fig, ax = plt.subplots(figsize=(7.4, 2.05))
    ax.set_xlim(0, 14.2); ax.set_ylim(0.55, 3.75); ax.axis("off")
    BY, BH = 1.15, 1.75                       # boxes span y in [1.15, 2.90]; titles sit ABOVE at y=3.18

    def dbox(x, w, col, a=0.12):
        ax.add_patch(FancyBboxPatch((x, BY), w, BH, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fill=True, fc=col, ec=col, alpha=a, lw=1.4))
        ax.add_patch(FancyBboxPatch((x, BY), w, BH, boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fill=False, ec=col, lw=1.4))

    def title(xc, s, col="black"):
        ax.text(xc, 3.18, s, ha="center", va="center", fontsize=8.2, weight="bold", color=col)

    ymid, ylo = 2.28, 1.55                   # two content rows inside each box
    # (1) target with crossing pairs
    dbox(0.2, 3.5, C["branch"]); title(1.95, "target $T$ (pseudoknot)", C["branch"])
    ax.text(1.95, ymid, r"$((\,[[\,))\,]]$", ha="center", va="center", fontsize=12)
    ax.text(1.95, ylo, "nested () + crossing []", ha="center", va="center", fontsize=7, color=C["gray"])
    # blind oracle arrow
    ax.add_patch(FancyArrowPatch((3.8, 2.02), (5.15, 2.02), arrowstyle="-|>", mutation_scale=12, color=C["gold"], lw=1.6))
    ax.text(4.48, 2.42, "MFE oracle", ha="center", fontsize=7.2, color=C["gold"])
    ax.text(4.48, 1.62, "cannot see []", ha="center", fontsize=6.8, color=C["gold"], style="italic")
    # (2) projected skeleton
    dbox(5.25, 3.4, C["match"]); title(6.95, "projected skeleton", C["match"])
    ax.text(6.95, ymid, r"$((\;..\;))\,..$", ha="center", va="center", fontsize=12)
    ax.text(6.95, ylo, "crossing erased", ha="center", va="center", fontsize=7, color=C["pfe"])
    # design arrow
    ax.add_patch(FancyArrowPatch((8.7, 2.02), (10.05, 2.02), arrowstyle="-|>", mutation_scale=12, color=C["gray"], lw=1.6))
    ax.text(9.38, 2.42, "design", ha="center", fontsize=7.2, color=C["gray"])
    # (3) verdict
    dbox(10.15, 3.9, C["gold"]); title(12.1, "silent false accept", C["gold"])
    ax.text(12.1, 2.42, "MFE: “solved” (250/251)", ha="center", va="center", fontsize=7.2)
    ax.text(12.1, 1.90, "pKiss: crossing stem", ha="center", va="center", fontsize=7.2, color=C["pfe"])
    ax.text(12.1, 1.52, r"absent in $99.6\%$ of cases", ha="center", va="center", fontsize=7.2, color=C["pfe"])
    fig.savefig(FIGS + "/f_projection.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS + "/f_projection.png", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("f_projection.pdf")


def f_costdecomp():
    """Cost decomposition: standalone vs incremental (reuse) vs cascade, and the measured cross-oracle cost."""
    d = _load("cost_decompose/cost_decompose.json")
    dec = d["decompose"]; xo = d["cross_oracle"]; ns = [r["n"] for r in dec]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].plot(ns, [r["pfe_sa"] * 1e3 for r in dec], "o-", color=C["pfe"], label="pfe standalone", ms=3)
    ax[0].plot(ns, [r["pfe_inc"] * 1e3 for r in dec], "s--", color=C["umfe"], label="pfe incremental (reuse, 66%)", ms=3)
    ax[0].plot(ns, [r["match"] * 1e3 for r in dec], ":", color=C["gray"], label="match (cheap rung)", ms=3)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("length n"); ax[0].set_ylabel("verify wall (ms)")
    ax[0].set_title("(a) reuse measured: incr.\\ = 66% of standalone"); ax[0].legend(frameon=False, fontsize=6.3)
    ax[1].plot([r["n"] for r in xo], [r["pkiss"] * 1e3 for r in xo], "^-", color=C["branch"], label="pKiss (cross-oracle)", ms=3)
    ax[1].plot([r["n"] for r in xo], [r["probknot"] * 1e3 for r in xo], "s-", color=C["sat"], label="ProbKnot (cross-oracle)", ms=3)
    ax[1].plot(ns, [r["pfe_sa"] * 1e3 for r in dec], "o--", color=C["pfe"], label="pfe (same-oracle)", ms=3)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("length n"); ax[1].set_ylabel("wall (ms)")
    ax[1].set_title("(b) cross-oracle cost (sub-second)"); ax[1].legend(frameon=False, fontsize=6.3)
    fig.tight_layout(); fig.savefig(FIGS + "/f_costdecomp.pdf"); plt.close(fig)
    print("f_costdecomp.pdf")


def f_oracle():
    """Axis-1 oracle-dependence envelope + the near-miss (crossing-stem) distribution among false accepts."""
    from collections import Counter
    d = _load("axis1_oracle_panel/panel.json"); s = d["summary"]; rows = d["rows"]
    fa = s["per_oracle_false_accept"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    labs = ["ProbKnot", "pKiss", "ShapeKnots"]; keys = ["probknot", "pkiss", "shapeknots"]
    vals = [fa[k]["false_accept_pct"] for k in keys]; cols = [C["sat"], C["pfe"], C["branch"]]
    ax[0].bar(labs, vals, color=cols)
    u = s["unanimous_false_accept"]["pct"]
    ax[0].axhline(u, ls="--", color=C["gray"], label=f"unanimous subset {u:.0f}%")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7)
    ax[0].set_ylabel("false-accept rate (%)"); ax[0].set_ylim(0, 70)
    ax[0].set_title("(a) Axis-1 rate is oracle-dependent"); ax[0].legend(frameon=False, fontsize=7)
    frows = [r for r in rows if r.get("mfe_reported_solved") and not r.get("forms_T_pkiss")
             and r.get("minbp_pkiss") is not None]
    mb = [r["minbp_pkiss"] for r in frows]; cnt = Counter(mb); xs = sorted(cnt); ys = [cnt[x] for x in xs]
    bcols = [C["match"] if x <= 2 else (C["gray"] if x < 5 else C["pfe"]) for x in xs]
    ax[1].bar([str(x) for x in xs], ys, color=bcols)
    ax[1].set_xlabel("pKiss bp-distance to full target"); ax[1].set_ylabel("# pKiss false accepts")
    near = sum(1 for x in mb if x <= 2)
    ax[1].set_title(f"(b) near-miss: {near}/{len(mb)} within 2 bp (blue)")
    fig.tight_layout(); fig.savefig(FIGS + "/f_oracle.pdf"); plt.close(fig)
    print("f_oracle.pdf")


if __name__ == "__main__":
    f_framework(); f_projection()
    f0(); f1(); f_pk(); f2(); f3(); f_routing(); f_llm(); f_costdecomp(); f_oracle()
    print("done ->", FIGS)
