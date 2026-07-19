"""R3.3+R3.4 re-analysis of the saved HumanEval+ test-vectors -- NO regeneration.

Fixes two reviewer defects using only the saved per-candidate boolean vectors [base..., plus...]:
 (3.3A) SELECTION/EVALUATION LEAKAGE: the strong arm previously selected on base+plus and was scored on
        base+plus. Here we split the plus tests into TUNING (used only for the strong arm's selection) and
        HELD-OUT GOLD (used only for evaluation, never for selection by either arm), over many random
        splits. The Goodhart-under-selection comparison (cheap-select vs strong-select, both scored on GOLD)
        is then leakage-free.
 (3.4)  METRIC UNIFICATION: over-acceptance reported BOTH as the joint mass P(pass cheap AND fail gold) and
        the conditional rate P(fail gold | pass cheap), each with raw numerator/denominator and a
        problem-clustered bootstrap 95% CI. Dial-ability reported as the marginal reduction per added base
        test with a CI over random base-test orderings (so it is not read as the set-inclusion tautology)."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs", "llm_evalplus")
MODELS = [("llama-3.1-8b", "meta-llama_llama-3.1-8b-instruct"),
          ("gpt-3.5", "openai_gpt-3.5-turbo"),
          ("gpt-4o-mini", "openai_gpt-4o-mini"),
          ("qwen-coder-32b", "qwen_qwen-2.5-coder-32b-instruct")]


def _clustered_ci(per_problem_num_den, rng, B=2000):
    """Bootstrap a ratio sum(num)/sum(den) by resampling PROBLEMS (cluster). Returns (pt, lo, hi)."""
    nums = np.array([x[0] for x in per_problem_num_den], float)
    dens = np.array([x[1] for x in per_problem_num_den], float)
    tot_d = dens.sum()
    pt = 100 * nums.sum() / tot_d if tot_d else 0.0
    if len(nums) == 0:
        return pt, 0.0, 0.0
    idx = np.arange(len(nums)); boots = []
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = dens[s].sum()
        boots.append(100 * nums[s].sum() / d if d else 0.0)
    return pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


PROPS = [round(0.1 * i, 1) for i in range(1, 11)]      # proportion-of-base-tests rung grid (0.1..1.0)


def analyze(per, rng, n_splits=100):
    """per: list of {n_base, vecs:[[bool...]]}. Returns leakage-free metrics."""
    # ---- (3.4) over-acceptance at cheap rung k=1 (first base test): joint + conditional, GOLD = held-out plus
    # aggregate over random gold/tune splits
    joint_pp, cond_pp = [], []       # per-problem (num, den) for joint / conditional, averaged over splits
    cheap_gold, strong_gold = [], [] # per-problem best-of-N gold-pass indicator (avg over splits)
    dial = {}                        # k -> list of per-problem conditional FP (avg over splits) [composition drifts]
    dial_prop = {}                   # proportion p -> list of per-problem (num,den) [ALL tasks at every p]
    ntask_k = {}                     # k -> #tasks with n_base>=k (to expose the drifting composition)
    for p in per:
        nb = p["n_base"]; vecs = [np.array(v, bool) for v in p["vecs"]]
        if not vecs or nb == 0 or len(vecs[0]) <= nb:
            continue
        plus_idx = np.arange(nb, len(vecs[0]))
        base = np.stack([v[:nb] for v in vecs])           # [N, nb]
        plus = np.stack([v[nb:] for v in vecs])           # [N, U]
        U = plus.shape[1]
        jn = jd = cn = cd = 0.0; cg = sg = 0.0
        kfp = {k: [0.0, 0.0] for k in range(1, nb + 1)}
        pfp = {pp: [0.0, 0.0] for pp in PROPS}
        for _ in range(n_splits):
            perm = rng.permutation(U)
            half = U // 2
            tune, gold = perm[:half], perm[half:]
            if len(gold) == 0:
                gold, tune = perm, perm[:0]
            gold_ok = plus[:, gold].all(1)                # GOLD correctness (held-out; never used to select)
            # over-acceptance at k=1 (first base test)
            passk1 = base[:, 0]
            jn += float((passk1 & ~gold_ok).sum()); jd += float(len(vecs))
            cd_ = float(passk1.sum()); cn_ = float((passk1 & ~gold_ok).sum())
            cn += cn_; cd += cd_
            # best-of-N: cheap selects on base only; strong selects on base+tune; both scored on GOLD
            cheap_pick = int(np.lexsort((np.arange(len(vecs)), -base.sum(1)))[0])
            strong_score = base.sum(1) + plus[:, tune].sum(1)
            strong_pick = int(np.lexsort((np.arange(len(vecs)), -strong_score))[0])
            cg += float(gold_ok[cheap_pick]); sg += float(gold_ok[strong_pick])
            # dial-ability: conditional FP as cheap rung strengthens (random base ordering per split)
            bperm = rng.permutation(nb)
            for k in range(1, nb + 1):                     # absolute-k curve (composition drifts with k)
                passk = base[:, bperm[:k]].all(1)
                kfp[k][0] += float((passk & ~gold_ok).sum()); kfp[k][1] += float(passk.sum())
            for pp in PROPS:                               # proportion-of-base-tests rung (ALL tasks at every p)
                kk = max(1, int(np.ceil(pp * nb)))
                passp = base[:, bperm[:kk]].all(1)
                pfp[pp][0] += float((passp & ~gold_ok).sum()); pfp[pp][1] += float(passp.sum())
        joint_pp.append((jn / n_splits, jd / n_splits))
        cond_pp.append((cn / n_splits, cd / n_splits))
        cheap_gold.append((cg / n_splits, 1.0)); strong_gold.append((sg / n_splits, 1.0))
        for k in kfp:
            dial.setdefault(k, []).append((kfp[k][0] / n_splits, kfp[k][1] / n_splits))
            ntask_k[k] = ntask_k.get(k, 0) + 1
        for pp in PROPS:
            dial_prop.setdefault(pp, []).append((pfp[pp][0] / n_splits, pfp[pp][1] / n_splits))
    j = _clustered_ci(joint_pp, rng); c = _clustered_ci(cond_pp, rng)
    cg = _clustered_ci(cheap_gold, rng); sg = _clustered_ci(strong_gold, rng)
    # dial-ability curve (conditional FP vs k), pooled -- keep point est (back-compat) AND full CI + n-tasks
    dcurve = {k: _clustered_ci(v, rng)[0] for k, v in sorted(dial.items())}
    dcurve_ci = {k: _clustered_ci(v, rng) for k, v in sorted(dial.items())}
    dcurve_prop = {pp: _clustered_ci(v, rng) for pp, v in sorted(dial_prop.items())}
    n_tot = len(joint_pp)
    return {"n_problems": n_tot,
            "joint_mass": j, "conditional_rate": c,
            "cheap_gold": cg, "strong_gold": sg,
            "raw": {"joint_num": sum(x[0] for x in joint_pp), "joint_den": sum(x[1] for x in joint_pp),
                    "cond_num": sum(x[0] for x in cond_pp), "cond_den": sum(x[1] for x in cond_pp)},
            "dial_curve": dcurve,
            "dial_curve_ci": dcurve_ci,                    # k -> (pt, lo, hi); composition drifts (see n_tasks_k)
            "n_tasks_k": ntask_k,                          # k -> #tasks contributing at absolute rung k
            "dial_curve_prop": dcurve_prop}                # proportion p -> (pt, lo, hi); ALL n_problems at every p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {}
    print("=== HumanEval+ LEAKAGE-FREE re-analysis (gold = held-out plus tests; never used for selection) ===")
    print(f"{'model':16} | joint P(passcheap&failgold) | cond P(failgold|passcheap) | GOLD best-of-N cheap->strong")
    for nm, slug in MODELS:
        fs = [f for f in glob.glob(os.path.join(RUNS, slug + "*.json")) if "summary" not in f]
        if not fs:
            print(f"  {nm}: no per-problem json"); continue
        per = json.load(open(fs[0]))["per"]
        r = analyze(per, rng, a.splits); out[nm] = r
        j, c, cg, sg = r["joint_mass"], r["conditional_rate"], r["cheap_gold"], r["strong_gold"]
        print(f"{nm:16} | {j[0]:5.1f}% [{j[1]:.1f},{j[2]:.1f}]  ({r['raw']['joint_num']:.0f}/{r['raw']['joint_den']:.0f}) "
              f"| {c[0]:5.1f}% [{c[1]:.1f},{c[2]:.1f}]  ({r['raw']['cond_num']:.0f}/{r['raw']['cond_den']:.0f}) "
              f"| {cg[0]:.1f}% -> {sg[0]:.1f}%  (gap {sg[0]-cg[0]:+.1f})")
    print("\n  dial-ability (conditional FP vs #base tests required, pooled, random orderings):")
    for nm in out:
        dc = out[nm]["dial_curve"]; ks = sorted(dc)
        print(f"    {nm:16}: " + " ".join(f"k{k}={dc[k]:.1f}%" for k in ks))
    json.dump(out, open(os.path.join(RUNS, "reanalysis.json"), "w"), indent=1)
    print(f"\n  wrote {os.path.join(RUNS, 'reanalysis.json')}")


if __name__ == "__main__":
    main()
