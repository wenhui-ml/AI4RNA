#!/usr/bin/env python3
"""A2 of the controlled scrape A/B: does the fix change what the product actually hands back?

RUNS IN THE RibonanzaNet venv. `scrape_ab.py` produced both arms from a byte-identical solver trajectory,
so anything measured here is the scrape and nothing else -- no seed noise, no load-dependent step drift.

Scoring rule, fixed in the pre-registration and repeated here so it cannot drift: a target neither arm
delivers scores 0.0 in both and contributes nothing to the paired test; a target only one arm delivers
scores 0.0 in the arm that failed it, so a coverage loss is priced as a quality loss rather than dropped.
Two-sided, alpha 0.05, paired within target.

COST: $0. Local CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pool_rerank_score import _score_chunk                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "scrape_ab.json"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "scrape_ab_score.json"))
    a = ap.parse_args()

    from m2r_pairwise import wilcoxon

    S = json.load(open(a.src))
    T = S["targets"]
    seq = {(r["arm"], r["puzzle"]): r.get("sequence") for r in S["rows"] if "error" not in r}
    errs = [r for r in S["rows"] if "error" in r]
    if errs:
        print(f"  {len(errs)} 次运行报错:{errs[0]['error'][:80]}")

    jobs = [((arm, p), seq[(arm, p)], T[p]["target"])
            for (arm, p) in sorted(seq) if seq[(arm, p)]]
    print(f"要打分的交付序列 {len(jobs)} 条,{a.workers} 进程\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // max(1, a.workers))
    chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]
    sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, chunks):
            for k, v in res:
                sc[k] = v

    ps = sorted(T)
    new = [sc.get(("new", p), 0.0) for p in ps]
    old = [sc.get(("old", p), 0.0) for p in ps]
    dn = sum(1 for p in ps if seq.get(("new", p)))
    do = sum(1 for p in ps if seq.get(("old", p)))

    print(f"=== A1 覆盖 ===")
    print(f"  新抓取 {dn}/{len(ps)}    旧抓取 {do}/{len(ps)}")
    only_new = [p for p in ps if seq.get(("new", p)) and not seq.get(("old", p))]
    only_old = [p for p in ps if seq.get(("old", p)) and not seq.get(("new", p))]
    print(f"  仅新臂交付 {len(only_new)} {only_new}")
    print(f"  仅旧臂交付 {len(only_old)} {only_old}")

    d = [x - y for x, y in zip(new, old)]
    w, p2, n2 = wilcoxon(d, "two-sided")
    print(f"\n=== A2 交付质量(未交付计 0.0)===")
    print(f"  新抓取中位 {st.median(new):.3f}    旧抓取中位 {st.median(old):.3f}")
    print(f"  配对差 中位 {st.median(d):+.3f},新臂更好 {sum(1 for x in d if x>0)},"
          f"更差 {sum(1 for x in d if x<0)},持平 {sum(1 for x in d if x==0)} / {len(ps)}")
    print(f"  p = {'n/a (差异非零的靶标不足 6 个)' if p2 is None else f'{p2:.3g}'}(双侧,靶标为独立单元)")
    sig = p2 is not None and p2 < 0.05
    verdict = ("新抓取显著更好" if sig and st.median(d) > 0 else
               "新抓取显著更差 —— 这是正确性修复的代价,必须照此上报" if sig and st.median(d) < 0 else
               "交付质量无可检出差异 —— 这是一次纯正确性修复:不再交付求解器自己丢弃的中间态,"
               "而交付质量没有可测量的变化")
    print(f"  判读:{verdict}")

    changed = [p for p in ps if seq.get(("new", p)) != seq.get(("old", p))]
    print(f"\n=== A3 交付序列变化的靶标 {len(changed)}/{len(ps)} ===")
    if changed:
        print(f"  {changed}")

    json.dump({"A1": {"new_delivered": dn, "old_delivered": do,
                      "only_new": only_new, "only_old": only_old},
               "A2": {"new_median": st.median(new), "old_median": st.median(old),
                      "paired_median_diff": st.median(d), "p": p2,
                      "better": sum(1 for x in d if x > 0), "worse": sum(1 for x in d if x < 0),
                      "tied": sum(1 for x in d if x == 0), "verdict": verdict},
               "A3": {"n_changed": len(changed), "changed": changed},
               "per_target": {p: {"new": sc.get(("new", p), 0.0), "old": sc.get(("old", p), 0.0)}
                              for p in ps}},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
