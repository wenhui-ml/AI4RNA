#!/usr/bin/env python3
"""Stage B: selection failure or generation failure, on the five targets we hand back at F1 = 0.000?

RUNS IN THE RibonanzaNet venv. Stage A collected each pool with the shipped ladder, the pinned hash seed
and the same seed as the shipped run, and recorded the panel's verdict on every candidate. Nothing here
touches the solver; RNet never entered the design loop and is the only judge the screen cannot have been
fitted to.

    SELECTION FAILURE  the pool holds a candidate above 0.5 and we shipped 0.000. The panel is emitting
                       confident false positives; the defect is in adjudication.
    GENERATION FAILURE the pool holds nothing above 0.5. The solver never built the pseudoknot here, which
                       is the same hole the `mcc` objective addresses.

The distinction matters because four of the five carry `knot_confirmed` -- the label whose text tells the
user three independent predictors agree. A confident wrong answer with a right one in the same pool is a
delivery defect, not a ranking gap, and it is reported as one.

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

from pool_rerank_score import _score_chunk                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "zero_delivery_diag.json"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "zero_delivery_diag_score.json"))
    a = ap.parse_args()

    P = json.load(open(a.src))
    GOOD, TS = P["good_threshold"], P["target_struct"]
    jobs = [((p, i), s, TS[p]) for p, pool in P["pools"].items() for i, s in enumerate(pool)]
    print(f"{len(P['pools'])} 个靶标,{len(jobs)} 条候选\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (a.workers * 4))
    sc = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))

    print(f"{'靶标':6s} {'出厂档位':16s} {'池':>5s} {'交付':>7s} {'池内最好':>9s} {'池中位':>8s} "
          f"{'>0.5 条数':>10s} {'诊断':>10s}")
    out = {}
    for p, pool in sorted(P["pools"].items()):
        v = [sc[(p, i)] for i in range(len(pool)) if (p, i) in sc]
        if not v:
            print(f"{p:6s} {P['verdicts'][p]:16s} {'空池':>5s}")
            continue
        ng = sum(1 for x in v if x > GOOD)
        why = "选择失败" if ng else "生成失败"
        out[p] = {"verdict": P["verdicts"][p], "delivered": P["delivered"][p], "n_pool": len(v),
                  "pool_best": max(v), "pool_median": st.median(v), "n_good": ng, "diagnosis": why}
        print(f"{p:6s} {P['verdicts'][p]:16s} {len(v):5d} {P['delivered'][p]:7.3f} {max(v):9.3f} "
              f"{st.median(v):8.3f} {ng:10d} {why:>10s}")

    sel = [p for p, v in out.items() if v["n_good"]]
    gen = [p for p in out if p not in sel]
    fp = [p for p in sel if out[p]["verdict"] == "knot_confirmed"]
    print(f"\n=== 判读 ===")
    print(f"  选择失败(池里有 >{GOOD} 的答案,我们没给):{len(sel)}/{len(out)}  {sel}")
    print(f"  生成失败(池里根本没有):              {len(gen)}/{len(out)}  {gen}")
    print(f"  其中带 `knot_confirmed` 的选择失败:    {len(fp)}  {fp}")
    if fp:
        print(f"  -> 面板发出了【自信的假阳性】:三个预测器一致同意的那条 F1 = 0.000,"
              f"同一池里有更好的。交付缺陷,不是排名差距。")
    if gen:
        print(f"  -> {gen} 上整池都没做出假结 —— 与 `mcc` 目标函数针对的是同一个洞。")

    # For the false positives, trace WHICH candidate the panel confirmed against what the pool held, so the
    # fix lands on adjudication rather than on a guess.
    tr = {}
    for p in sel:
        pool, pan = P["pools"][p], {x["seq"]: x for x in P["panel"].get(p, [])}
        rows = [{"i": i, "f1c": sc.get((p, i)), "panel_mean": pan.get(s, {}).get("panel_mean"),
                 "panel_k": pan.get(s, {}).get("panel_k"),
                 "confirmed": pan.get(s, {}).get("confirmed")} for i, s in enumerate(pool)]
        conf = [r for r in rows if r["confirmed"]]
        best = max(rows, key=lambda r: (r["f1c"] or 0))
        tr[p] = {"n_confirmed": len(conf),
                 "confirmed_f1_median": st.median([r["f1c"] for r in conf if r["f1c"] is not None])
                 if conf else None,
                 "confirmed_f1_max": max([r["f1c"] for r in conf if r["f1c"] is not None]) if conf else None,
                 "pool_best": best}
        print(f"  {p}: 面板确认 {len(conf)} 条,其 F1 中位 {tr[p]['confirmed_f1_median']}, "
              f"最好 {tr[p]['confirmed_f1_max']};全池最好 {best['f1c']:.3f} "
              f"(该条 panel_k={best['panel_k']}, mean={best['panel_mean']})")

    json.dump({"good_threshold": GOOD, "per_target": out, "selection_failures": sel,
               "generation_failures": gen, "confirmed_false_positives": fp, "trace": tr},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
