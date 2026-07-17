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
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
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
    ax.text(0.03, frac[0] + 0.06, "flat at gold for all $\\epsilon\\leq0.9$:\n"
            "match-accept FP $\\approx100\\%$, so no cheaper\nrung is admissible; savings are reject-only",
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
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    # (a) provided-strength ablation: FP-mass at cheap rung k=1 vs k=3, per model
    names, fp1, fp3 = [], [], []
    llama = None
    for nm, slug in _LLM_MODELS:
        d = _load(slug + "/llm_goodhart.json"); a = d["agg"]
        names.append(nm); fp1.append(100 - a["k1"]["survival"]); fp3.append(100 - a["k3"]["survival"])
        if "llama" in slug:
            llama = (d["budgets"], a["k1"]["byN"])
    x = np.arange(len(names))
    ax[0].bar(x - 0.19, fp1, 0.38, color=C["gold"], label="cheap rung k=1 (weak)")
    ax[0].bar(x + 0.19, fp3, 0.38, color=C["match"], label="cheap rung k=3 (stronger)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, fontsize=6.5, ha="right")
    ax[0].set_ylabel("false-accept mass (%)"); ax[0].set_title("(a) dial the rung $\\to$ dial over-accept")
    ax[0].legend(frameon=False, fontsize=6.5)
    # (b) Goodhart best-of-N on the clearest model (llama-3.1-8b): cheap flat, strong climbs
    if llama:
        budg, byN = llama
        cr = [byN[str(N)]["cheap_robust"] for N in budg]
        sr = [byN[str(N)]["strong_robust"] for N in budg]
        cp = [byN[str(N)]["cheap_proxy"] for N in budg]
        ax[1].plot(budg, cp, "o--", color=C["cheap"], label="cheap: proxy")
        ax[1].plot(budg, cr, "o-", color=C["gold"], label="cheap: robustness")
        ax[1].plot(budg, sr, "s-", color=C["sat"], label="strong: robustness")
        ax[1].set_xscale("log", base=2); ax[1].set_xlabel("best-of-N budget")
        ax[1].set_ylabel("acceptance (%)"); ax[1].set_ylim(60, 104)
        ax[1].set_title("(b) Goodhart (llama-3.1-8b)"); ax[1].legend(frameon=False, fontsize=6.3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIGS + "/f_llm.pdf"); plt.close(fig)
    print("f_llm.pdf")


if __name__ == "__main__":
    f0(); f1(); f_pk(); f2(); f3(); f_routing(); f_llm()
    print("done ->", FIGS)
