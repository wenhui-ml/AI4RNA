#!/usr/bin/env python3
"""A16: turn DesiRNA's ACGU-composition constraint ON and see whether the search finds different, better work.

Declared in `a16_declaration.json`, committed and pushed before the 57-target run.

WHY THIS IS NOT ALREADY ANSWERED. `generation_lever.py`'s B3 closed the CORRELATIONAL version: inside the
candidates we already generated, composition does not predict crossed-pair F1 (rho median +0.078 against a
0.15 threshold, pass=false). That answers "would re-ranking the existing pool by composition help". It does
NOT answer "would constraining composition make the search find different candidates", which changes the
trajectory and can produce molecules the pool never contained. The absence of a correlation does not exclude
an intervention. And the lead was written in generation_lever's own docstring: DesiRNA exposes
`-acgu {off,on}` and our adapter has never passed it -- every run this project has done carries `ACGUoff`.

A 3-TARGET SMOKE RAN FIRST (`acgu_smoke.log`) for one reason: if the flag never reaches DesiRNA the whole
experiment is void. It reaches it -- sequences differ and GC drops (0.588->0.559, 0.542->0.375, 0.594->0.375),
with `screened` at 200 in both arms and no verdict lost on any of the three.

THE CONFOUND IS MADE VISIBLE RATHER THAN ASSUMED AWAY. DesiRNA is wall-clock bounded, so the same seed sees a
different number of candidates under different machine load; this project measured a 22.9% verdict flip rate
between same-configuration replicates. Three defences, all declared: the two arms run BACK TO BACK in one
worker so load is as similar as it can be within a pair; `screened` and `wall_s` are recorded per run and the
paired difference in `screened` is printed as a first-class result; and the unit is the TARGET with the median
over three seeds, not 342 independent runs.

HARD VETO. If ACGU-on costs ANY target a certificate tier it held with the flag off, this cannot be read as an
improvement whatever k does. Same rule as grade_max_bench's P2: there is nothing worth a lost certificate.

COST: $0, local CPU. 57 targets x 2 arms x 3 seeds = 342 solver runs at the `cheap` tier. DesiRNA forks 10
replicas per job, so --workers 12 is about 120 cores, leaving headroom under the 160 ceiling for other users.
"""
from __future__ import annotations

import argparse, collections, json, os, random, statistics as st, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

TIERS = ("certified", "knot_confirmed", "knot_screened", "cannot_certify", "not_found")
RANK = {t: i for i, t in enumerate(TIERS)}          # lower index = stronger tier


def work(job):
    """Both arms of one (target, seed), back to back in one process so load is shared within the pair."""
    tid, target, budget, seed = job
    out = {"task_id": tid, "seed": seed}
    for arm, args in (("off", ""), ("on", "-acgu on")):
        os.environ["DESIRNA_ARGS"] = args
        from rna_harness.oracles_pk import _oracle_pairs_cached
        _oracle_pairs_cached.cache_clear()                 # each arm pays its own folds
        from rna_harness.cognitive.certify import certify_design
        t0 = time.time()
        try:
            c = certify_design(target, budget=budget, seed=seed)
        except Exception as e:                                                   # noqa: BLE001
            out[arm] = {"error": repr(e)[:160]}; continue
        s = c.sequence or ""
        d = c.detail or {}
        out[arm] = {"verdict": c.verdict, "seq": s,
                    "k": len((d.get("knot") or {}).get("confirmed_by") or []),
                    "gc": (s.count("G") + s.count("C")) / len(s) if s else None,
                    "screened": d.get("n_candidates_screened") or 0,
                    "wall_s": round(time.time() - t0, 1)}
    return out


def boot(vals, draws, seed):
    rng = random.Random(seed)
    v = list(vals); n = len(v)
    if n < 5:
        return None, None, None
    o = sorted(st.median([v[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    return st.median(v), o[int(0.025 * draws)], o[min(draws - 1, int(0.975 * draws))]


def line(label, vals, draws, seed):
    if len(vals) < 5:
        print(f"  {label:30s} 单位 {len(vals):3d} —— 太少"); return {"n": len(vals), "no_data": True}
    pt, lo, hi = boot(vals, draws, seed)
    pos = sum(1 for x in vals if x > 0); neg = sum(1 for x in vals if x < 0)
    ex = lo > 0 or hi < 0
    print(f"  {label:30s} 靶标 {len(vals):3d}  中位 {pt:+.4f}  均值 {st.mean(vals):+.4f}  "
          f"正/负 {pos:3d}/{neg:3d}  CI [{lo:+.4f}, {hi:+.4f}]{'  排除 0' if ex else '  跨过 0'}")
    return {"n": len(vals), "median": pt, "mean": st.mean(vals), "n_pos": pos, "n_neg": neg,
            "ci": [lo, hi], "excludes_zero": bool(ex),
            "excludes_zero_positive": bool(ex and lo > 0),
            "excludes_zero_negative": bool(ex and hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=57)
    ap.add_argument("--budget", default="cheap")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "acgu_intervention.json"))
    a = ap.parse_args()
    D = json.load(open(os.path.join(HERE, "a16_declaration.json")))
    seeds = [int(x) for x in a.seeds.split(",")]

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())[:a.n]
    jobs = [(str(t.task_id), str(t.target), a.budget, s) for t in tasks for s in seeds]
    print(f"A16:{len(tasks)} 靶标 × 2 臂 × {len(seeds)} 种子 = {len(jobs)*2} 次求解器运行,"
          f"预算 {a.budget},{a.workers} 进程(DesiRNA 每任务 fork 10 副本 → 约 {a.workers*10} 核)", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(work, jobs, chunksize=1), 1):
            R.append(r)
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)} 对  {(time.time()-t0)/60:.1f} min", flush=True)

    err = [r for r in R if "error" in r.get("off", {}) or "error" in r.get("on", {})]
    print(f"\n  出错的运行对 {len(err)}/{len(R)}" + (f"  例:{err[0]}" if err else ""))
    ok = [r for r in R if "error" not in r.get("off", {}) and "error" not in r.get("on", {})]

    per = collections.defaultdict(list)
    for r in ok:
        per[r["task_id"]].append(r)
    res = {"declaration": D, "n_targets": len(per), "n_pairs": len(ok), "n_error_pairs": len(err),
           "rows": R}

    print(f"\n=== 先看标志到底有没有起作用(声明:GC 不变则实验作废)===")
    gc = [st.median([x["on"]["gc"] - x["off"]["gc"] for x in v]) for v in per.values()
          if all(x["on"]["gc"] is not None and x["off"]["gc"] is not None for x in v)]
    res["gc_diff"] = line("GC(on − off)", gc, a.draws, 1)

    print(f"\n=== 混杂可见性(声明:必须报,不能只报差值)===")
    res["screened_diff"] = line("screened(on − off)",
                                [st.median([x["on"]["screened"] - x["off"]["screened"] for x in v])
                                 for v in per.values()], a.draws, 2)
    res["wall_diff"] = line("wall_s(on − off)",
                            [st.median([x["on"]["wall_s"] - x["off"]["wall_s"] for x in v])
                             for v in per.values()], a.draws, 3)

    print(f"\n=== 硬性否决:有没有靶标丢掉它在 off 臂拿到的证书档位 ===")
    lost = []
    for tid, v in per.items():
        bo = min(RANK[x["off"]["verdict"]] for x in v)      # best tier the off arm ever reached
        bn = min(RANK[x["on"]["verdict"]] for x in v)
        if bn > bo:
            lost.append((tid, TIERS[bo], TIERS[bn]))
    print(f"  丢档位的靶标 {len(lost)}" + (f"  {lost[:6]}" if lost else "  ✓ 无"))
    res["lost_tier"] = lost
    cnt = {arm: collections.Counter(x[arm]["verdict"] for x in ok) for arm in ("off", "on")}
    print(f"  档位计数 off {dict(cnt['off'])}")
    print(f"           on  {dict(cnt['on'])}")
    res["tier_counts"] = {k: dict(v) for k, v in cnt.items()}

    print(f"\n=== 主检验:k 的配对差(单位 = 靶标,三种子取中位,双侧)===")
    res["k_diff"] = line("k(on − off)", [st.median([x["on"]["k"] - x["off"]["k"] for x in v])
                                         for v in per.values()], a.draws, 4)

    K, G = res["k_diff"], res["gc_diff"]
    if G.get("no_data") or not G["excludes_zero"]:
        v = "flag_does_nothing"
    elif lost or K.get("excludes_zero_negative"):
        v = "harmful"
    elif K.get("excludes_zero_positive"):
        v = "improvement"
    else:
        v = "null"
    print(f"\n=== 判定(按声明)===")
    print(f"  【结论】{v} —— {D['outcomes_fixed_in_advance'][v]}")
    if v == "improvement":
        print(f"  注意:按声明它【仍需一次留出确认才能上线】,本次不改产品。")
    res["verdict"] = v
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
