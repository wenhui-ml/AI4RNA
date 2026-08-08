#!/usr/bin/env python3
"""Does our solver climb OUR criterion while sliding down reality's? The two readings of the tier decline.

WHAT IS ALREADY MEASURED. Running the solver longer makes its typical candidate worse by RibonanzaNet's
reckoning, monotonically -- median crossed-pair F1 0.776 / 0.714 / 0.708 / 0.688 at 5 / 15 / 60 / 300 s,
paired difference -0.058 over 57 targets, worse on 41 of them, p = 2.1e-06 -- while the best candidate in
the pool does not move at all (+0.000). Separately, our panel predicts the tube at rho +0.263 where
RibonanzaNet manages +0.421, and the two criteria agree with each other at only +0.395.

WHY THAT IS NOT YET A FINDING. Two different stories fit it, and they call for opposite fixes:

    OVER-OPTIMISATION   the solver really is improving -- on ITS objective -- and our objective diverges
                        from reality, so the extra seconds buy real progress up the wrong hill.
                        Fix: change what the generator optimises.

    DILUTION            the solver is not improving on anything; longer runs simply emit more candidates
                        (21 -> 200) and the extra ones are worse, dragging the median down while the
                        ceiling stays put.
                        Fix: cap the ladder. Nothing is wrong with the objective, we are just paying for
                        candidates that are not worth having.

The decisive difference is what OUR OWN criterion says about the same candidates over the same time, and
that has never been looked at -- every tier comparison so far has been scored by RibonanzaNet alone.

PRE-REGISTERED, before any adjudication runs. Same candidates scored both ways, subsampled identically from
the 5 s and 300 s tiers of `search_convergence_pools.json`. Independent unit is the target.

    O1  OUR CRITERION OVER TIME. Panel mean crossing recall, 300 s tier minus 5 s tier, paired within
        target, Wilcoxon, one-sided greater.
    O2  REALITY PROXY OVER TIME. RNet crossed-pair F1 on the SAME subsample, one-sided less. This
        re-confirms the known decline on the subsample actually used, so O1 and O2 are commensurable.

    VERDICT, fixed here:
        O1 rises AND O2 falls   -> OVER-OPTIMISATION. The generator's objective is the thing to change.
        O1 flat AND O2 falls    -> DILUTION. Cap the ladder; the objective is not implicated.
        O1 falls too            -> the solver degrades on its own objective as well, which would mean the
                                   long tiers are simply broken and is a bug report, not a research finding.

COST: $0 API. Local CPU and local oracle binaries. Runs in the HARNESS venv, and shells out to the
RibonanzaNet venv for the RNet half, because no single venv has both ViennaRNA and torch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

RNET_PY = "/data/wenhui/rnet_env/bin/python"
ALPHA = 0.025


def work(item):
    puz, tier, seq, target = item
    from rna_harness.cognitive.certify import _knot_recall
    try:
        kr = _knot_recall(seq, target)
    except Exception as e:                                                     # noqa: BLE001
        return {"puzzle": puz, "tier": tier, "error": repr(e)[:120]}
    vals = [v for v in kr["per_oracle"].values() if v is not None]
    if not vals:
        return {"puzzle": puz, "tier": tier, "error": "all oracles failed"}
    return {"puzzle": puz, "tier": tier, "seq": seq,
            "panel_mean": sum(vals) / len(vals),
            "panel_k": sum(1 for v in vals if v >= 1.0),
            "confirmed": bool(kr["knot_confirmed"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=os.path.join(HERE, "search_convergence_pools.json"))
    ap.add_argument("--per-tier", type=int, default=20)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "overoptimization.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    P = json.load(open(a.pools))
    T, tiers = P["targets"], P["tiers"]
    lo, hi = tiers[0], tiers[-1]
    pools = {}
    for r in P["rows"]:
        if r["tier"] in (lo, hi) and r["pool"]:
            pools[(r["puzzle"], r["tier"])] = r["pool"]

    rng = random.Random(a.seed)
    jobs = []
    for (p, tl), pool in sorted(pools.items()):
        for s in rng.sample(pool, min(a.per_tier, len(pool))):
            jobs.append((p, tl, s, T[p]["target"]))
    print(f"抽样:每靶标每档最多 {a.per_tier} 条,{lo}s 与 {hi}s 两档,合计 {len(jobs)} 条")
    print(f"同一批序列两把尺子都量一遍,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R.append(res)
            if n % 400 == 0 or n == len(jobs):
                print(f"    面板 {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    R = [r for r in R if "error" not in r]
    print(f"  面板裁决成功 {len(R)}")

    # ---- RNet half, in the other venv -----------------------------------------------------------
    req = [{"puzzle": r["puzzle"], "tier": r["tier"], "seq": r["seq"],
            "target": T[r["puzzle"]]["target"]} for r in R]
    tmp_in = os.path.join(HERE, ".overopt_in.json")
    tmp_out = os.path.join(HERE, ".overopt_out.json")
    json.dump(req, open(tmp_in, "w"))
    helper = os.path.join(HERE, "_overopt_rnet.py")
    with open(helper, "w") as fh:
        fh.write(
            "import json,sys,os\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "from pool_rerank_score import _score_chunk\n"
            "from concurrent.futures import ProcessPoolExecutor\n"
            "req=json.load(open(sys.argv[1])); W=int(sys.argv[3])\n"
            "jobs=[((i,),r['seq'],r['target']) for i,r in enumerate(req)]\n"
            "CH=max(1,len(jobs)//(W*8)); ch=[jobs[i:i+CH] for i in range(0,len(jobs),CH)]\n"
            "sc={}\n"
            "with ProcessPoolExecutor(max_workers=W) as ex:\n"
            "    for res in ex.map(_score_chunk, ch):\n"
            "        for k,v in res: sc[k[0]]=v\n"
            "json.dump([sc.get(i) for i in range(len(req))], open(sys.argv[2],'w'))\n")
    print(f"\n  转到 rnet venv 给同一批 {len(req)} 条序列打分…", flush=True)
    subprocess.run([RNET_PY, helper, tmp_in, tmp_out, str(a.workers)], check=True, cwd=HERE)
    f1s = json.load(open(tmp_out))
    for r, v in zip(R, f1s):
        r["f1c"] = v
    R = [r for r in R if r.get("f1c") is not None]
    for f in (tmp_in, tmp_out, helper):
        os.remove(f)

    by = {}
    for r in R:
        by.setdefault(r["puzzle"], {}).setdefault(r["tier"], []).append(r)

    print(f"\n=== 同一批序列,两把尺子,{lo}s 对 {hi}s ===")
    out = {"lo": lo, "hi": hi, "n_scored": len(R), "per_tier_sample": a.per_tier}
    for label, key, alt in (("O1 我们的面板(平均交叉召回)", "panel_mean", "greater"),
                            ("O2 RibonanzaNet(交叉配对 F1)", "f1c", "less")):
        ps = [p for p in by if lo in by[p] and hi in by[p]]
        d = [st.median([x[key] for x in by[p][hi]]) - st.median([x[key] for x in by[p][lo]])
             for p in ps]
        w, pv, _ = wilcoxon(d, alt)
        print(f"  {label:32s} {hi}s−{lo}s 中位 {st.median(d):+.3f}  "
              f"上升 {sum(1 for x in d if x>0):2d} 下降 {sum(1 for x in d if x<0):2d} / {len(ps)}  "
              f"p({alt}) = {'n/a' if pv is None else f'{pv:.3g}'}")
        out[key] = {"n_targets": len(ps), "median_diff": st.median(d), "p": pv, "alt": alt,
                    "up": sum(1 for x in d if x > 0), "down": sum(1 for x in d if x < 0),
                    # kept so a tail or a threshold can be re-derived without re-running the 10 minutes of
                    # panel adjudication; the first version of this script saved only the aggregates and a
                    # wrong-tailed p-value could not be recomputed from what was on disk
                    "per_target_diff": {p: v for p, v in zip(ps, d)}}

    # O1 is tested one-sided GREATER (does our own criterion improve), O2 one-sided LESS (does the reality
    # proxy decline), so a small p means something different in each row and they must not be compared
    # directly. O1's own downward direction needs its own test rather than a sign check.
    o1_up = out["panel_mean"]["p"] is not None and out["panel_mean"]["p"] < ALPHA
    o2_dn = out["f1c"]["p"] is not None and out["f1c"]["p"] < ALPHA
    ps1 = list(out["panel_mean"]["per_target_diff"].values())
    _, p_o1_dn, _ = wilcoxon(ps1, "less")
    out["panel_mean"]["p_decline"] = p_o1_dn
    o1_dn = p_o1_dn is not None and p_o1_dn < ALPHA
    out["verdict"] = ("过度优化 —— 求解器在自己的判据上确有进步,而在贴近现实的那把尺子上退步;"
                      "要改的是生成器优化的目标" if (o1_up and o2_dn) else
                      "长档在自己的判据上也在退步 —— 这是 bug,不是研究发现" if o1_dn else
                      "稀释 —— 求解器没有在任何判据上变好,只是吐出更多且更差的候选;"
                      "该做的是给阶梯封顶,与目标函数无关")
    print(f"\n  判读:{out['verdict']}")
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
