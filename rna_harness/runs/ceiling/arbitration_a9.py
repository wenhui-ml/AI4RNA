#!/usr/bin/env python3
"""A9: the criterion against certain truth, on populations hard enough that the two rulers actually disagree.

Declared in `arbitration_a9_declaration.json`, committed before any number here existed. Read that file for
the design; the two things worth repeating are:

  THE NESTED POPULATION IS BIASED AGAINST THE CRITERION, ON PURPOSE AND BY DECLARATION. Decoys are built by
  RNA.inverse_fold, and pKiss and ShapeKnots share the Turner parameter family with ViennaRNA while
  RibonanzaNet does not, so the panel is advantaged at the absolute level. The primary test is therefore the
  CONTRAST between the top and bottom regions, which a uniform advantage cannot create; "the top region
  clears 0.5" is the conservative secondary. A pass here is strong. A failure cannot be attributed, and is
  recorded as such rather than as a refutation.

  THE CROSSNATIVE POPULATION HAS NO SHARED ERROR PATH BUT ALMOST NO POWER. Same-length cells reuse the same
  molecules, so the independent unit is the LENGTH BAND, and only eight bands can fill a cell of eight. It is
  reported as exploratory and can neither confirm nor refute.

A8's population (natives vs random sequences) reached its declared null at 15 discordant pairs, and the
reason is structural: the criterion is only defined on disagreement, and both rulers get an easy
discrimination right. That is kept in `arbitration_absolute.py`, not deleted.

STAGES, because the rulers live in different environments -- see run_a9_chain.sh.

COST: $0. All local.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, statistics as st, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_absolute import ranks01                                     # noqa: E402

N_OTHER = 7
DECL = os.path.join(HERE, "arbitration_a9_declaration.json")


def paths(pop):
    return (os.path.join(HERE, f"a9_{pop}_pop.json"), os.path.join(HERE, f"a9_{pop}_rnet.json"),
            os.path.join(HERE, f"a9_{pop}_panel.json"))


def _matched():
    """The 250 Pseudobase targets whose wild-type sequence is recoverable, keyed as A8 keyed them."""
    from rna_harness.experiments.axis1_native_control import parse_natives
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs
    nat = parse_natives()
    out = []
    for t in load_pseudobase():
        h = nat.get((len(t.target), frozenset(pk_pairs(t.target))))
        if h:
            out.append((t.task_id, t.target, h[1]))
    return sorted(out)


def build(pop):
    P, _, _ = paths(pop)
    M = _matched()
    cells = []
    if pop == "nested":
        from calibrate_strict import nested_decoy
        # nested_decoy() already retries inverse_fold internally and returns None when it cannot hit the
        # skeleton exactly. On some targets it almost always fails -- pseudobase:88 (108 nt, 12-pair skeleton)
        # yields one decoy per 56 seconds -- so an unbounded outer retry spends minutes on a single target for
        # a cell it will then drop anyway. Bounded by wall clock per target, with what was dropped REPORTED
        # rather than silently absorbed: a population whose hard cases are quietly missing is a biased
        # population.
        BUDGET_S, short = 20.0, []
        for n_, (tid, tgt, seq) in enumerate(M, 1):
            t0 = time.time()
            got, seen = [], {seq}
            while len(got) < N_OTHER and time.time() - t0 < BUDGET_S:
                d = nested_decoy(tgt)
                if d and d not in seen:
                    seen.add(d); got.append(d)
            if len(got) < N_OTHER:
                short.append((tid, len(tgt), len(got)))
            if len(got) < 2:                      # a cell needs >= 3 rows to standardise either ruler
                continue
            cells.append({"id": tid, "target": tgt, "band": len(tgt),
                          "rows": [{"kind": "native", "seq": seq}]
                                  + [{"kind": "other", "seq": d} for d in got]})
            if n_ % 25 == 0:
                print(f"    {n_}/{len(M)} 靶标,已成格 {len(cells)}", flush=True)
        print(f"  {BUDGET_S:.0f}s 预算内凑不满 {N_OTHER} 条诱饵的靶标 {len(short)} 个"
              f"(其中不足 2 条、整格丢弃的 {sum(1 for x in short if x[2] < 2)} 个)")
        if short:
            print(f"    最难的几个(靶标, 长度, 拿到几条): {sorted(short, key=lambda x: x[2])[:6]}")
    elif pop == "crossnative":
        byL = collections.defaultdict(list)
        for tid, tgt, seq in M:
            byL[len(tgt)].append((tid, seq))
        for tid, tgt, seq in M:
            pool = [(o, s) for o, s in byL[len(tgt)] if o != tid]
            if len(pool) < 2:
                continue
            rng = random.Random(int(hashlib.sha1(f"A9|{tid}".encode()).hexdigest()[:12], 16))
            take = rng.sample(pool, min(N_OTHER, len(pool)))
            cells.append({"id": tid, "target": tgt, "band": len(tgt),
                          "rows": [{"kind": "native", "seq": seq}]
                                  + [{"kind": "other", "seq": s} for _o, s in take]})
    else:
        raise SystemExit(pop)
    n = sum(len(c["rows"]) for c in cells)
    sz = sorted(len(c["rows"]) for c in cells)
    print(f"[{pop}] 格 {len(cells)},序列 {n},每格 {sz[0]}–{sz[-1]} 条(中位 {sz[len(sz)//2]});"
          f"长度档 {len({c['band'] for c in cells})}")
    json.dump({"pop": pop, "n_cells": len(cells), "n_seqs": n, "cells": cells}, open(P, "w"))
    print(f"写出 {P}")


def stage_rnet(pop, workers):
    from pool_rerank_score import _score_chunk
    P, R, _ = paths(pop)
    D = json.load(open(P))
    jobs = [((c["id"], i), r["seq"], c["target"]) for c in D["cells"] for i, r in enumerate(c["rows"])]
    print(f"[{pop}] {len(jobs)} 条过 RibonanzaNet,{workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (workers * 8))
    sc, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res))
    json.dump({"n": len(sc), "minutes": (time.time() - t0) / 60,
               "f1c": {f"{k[0]}|{k[1]}": v for k, v in sc.items()}}, open(R, "w"))
    print(f"  完成 {len(sc)}/{len(jobs)},{(time.time()-t0)/60:.1f} 分钟")


def stage_panel(pop, workers):
    from panel_f1_designs import work
    P, _, PN = paths(pop)
    D = json.load(open(P))
    jobs = [(f"{c['id']}|{i}", r["seq"], c["target"], 0.0, 0.0, r["kind"])
            for c in D["cells"] for i, r in enumerate(c["rows"])]
    print(f"[{pop}] {len(jobs)} 条过三家出厂判官,{workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R.append(res)
            if n % 400 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    bad = [r for r in R if "error" in r]
    print(f"  成功 {len(R)-len(bad)},失败 {len(bad)}")
    json.dump({"n": len(R) - len(bad), "n_failed": len(bad),
               "rows": [{"key": r["puzzle"], "panel_recall_product": r["panel_recall_product"],
                         "panel_f1": r["panel_f1"]} for r in R if "error" not in r]}, open(PN, "w"))
    print(f"  写出 {PN}")


def pairs_of(D, RN, pan, col):
    """Discordant (native, other) pairs per cell, each labelled by whether the LEARNED ruler chose truth."""
    cells, dropped = {}, 0
    for c in D["cells"]:
        rows = []
        for i, r in enumerate(c["rows"]):
            k = f"{c['id']}|{i}"
            if k in RN and k in pan:
                rows.append({"kind": r["kind"], "f1c": RN[k], "panel": pan[k][col]})
        if len(rows) < 3:
            dropped += 1; continue
        vf, vp = [x["f1c"] for x in rows], [x["panel"] for x in rows]
        if st.pstdev(vf) == 0 or st.pstdev(vp) == 0:
            dropped += 1; continue
        for x, a_, b_ in zip(rows, ranks01(vf), ranks01(vp)):
            x["rf"], x["rp"] = a_, b_
        nat = [x for x in rows if x["kind"] == "native"][0]
        ps = []
        for o in (x for x in rows if x["kind"] == "other"):
            dr, dp = nat["f1c"] - o["f1c"], nat["panel"] - o["panel"]
            if dr == 0 or dp == 0 or (dr > 0) == (dp > 0):
                continue
            ps.append({"learned_right": dr > 0,
                       "d": abs(nat["rf"] - o["rf"]) - abs(nat["rp"] - o["rp"]),
                       "band": c["band"]})
        if ps:
            cells[c["id"]] = ps
    return cells, dropped


def boot(units, draws, rng, contrast=False, cut_lo=None, cut_hi=None):
    """Cluster bootstrap over `units` (dict unit -> pairs). One vote per unit.

    contrast=False: fraction learned_right. contrast=True: (top region fraction) - (bottom region fraction),
    recomputed inside each bootstrap draw so the two regions are resampled together, which is what makes the
    difference a paired quantity rather than two independent intervals compared by eye.
    """
    ks = list(units)
    if not ks:
        return None, None, None

    def frac(sel, lo, hi):
        num = den = 0
        for k in sel:
            ps = [p for p in units[k] if lo < p["d"] <= hi]
            if ps:
                num += sum(1 for p in ps if p["learned_right"]) / len(ps); den += 1
        return num / den if den else None

    def est(sel):
        if not contrast:
            return frac(sel, -1e9, 1e9)
        a, b = frac(sel, cut_hi, 1e9), frac(sel, -1e9, cut_lo)
        return None if a is None or b is None else a - b

    pt = est(ks)
    out = sorted(v for v in (est([ks[rng.randrange(len(ks))] for _ in ks]) for _ in range(draws))
                 if v is not None)
    if not out:
        return pt, None, None
    return pt, out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def analyse(pop, draws, seed):
    D0 = json.load(open(DECL))
    P, R, PN = paths(pop)
    D = json.load(open(P)); RN = json.load(open(R))["f1c"]
    pan = {r["key"]: r for r in json.load(open(PN))["rows"]}
    unit_is_band = (pop == "crossnative")
    res = {"pop": pop, "unit": "length_band" if unit_is_band else "cell"}
    print(f"[{pop}] 格 {D['n_cells']},序列 {D['n_seqs']};独立单位 = "
          f"{'长度档(格之间共用分子)' if unit_is_band else '格'}")

    for col in ("panel_recall_product", "panel_f1"):
        cuts = D0["cuts_reused_not_refitted"][col]
        lo_c, hi_c = cuts["bottom"], cuts["top"]
        tag = "出厂列(主)" if col == "panel_recall_product" else "观测列(次)"
        cells, dropped = pairs_of(D, RN, pan, col)
        npair = sum(len(v) for v in cells.values())
        print(f"\n=== {tag} {col};切点复用 顶 {hi_c:.5f} 底 {lo_c:.5f}(未重拟合)===")
        print(f"  可用格 {len(cells)}(丢弃 {dropped}),相悖的(野生型,难零假设)配对 {npair}")
        row = {"n_cells": len(cells), "n_pairs": npair, "n_dropped": dropped}
        if npair < 100:
            print("  按声明:相悖配对 < 100 → 无数据,记无结论。")
            row["verdict"] = "no_data_as_declared"; res[col] = row; continue
        units = cells
        if unit_is_band:
            byb = collections.defaultdict(list)
            for v in cells.values():
                for p in v:
                    byb[p["band"]].append(p)
            units = dict(byb)
            print(f"  按长度档聚类:{len(units)} 个档(功效不足,探索性)")
        for name, lo, hi in (("底档", -1e9, lo_c), ("中档", lo_c, hi_c), ("顶档", hi_c, 1e9)):
            sub = {k: [p for p in v if lo < p["d"] <= hi] for k, v in units.items()}
            sub = {k: v for k, v in sub.items() if v}
            n = sum(len(v) for v in sub.values())
            if n < 20:
                print(f"  {name} n={n} —— 太少,不给数"); row[name] = {"n": n}; continue
            pt, l_, h_ = boot(sub, draws, random.Random(seed))
            print(f"  {name} n={n:5d} 单位{len(sub):3d}  跟学习型正确率 {pt:.1%} [{l_:.1%},{h_:.1%}]"
                  f"{'  ✓上越' if l_ > 0.5 else ('  ✓下越' if h_ < 0.5 else '  跨0.5')}")
            row[name] = {"n": n, "n_units": len(sub), "frac": pt, "ci": [l_, h_],
                         "clears_above": bool(l_ > 0.5), "clears_below": bool(h_ < 0.5)}
        pt, l_, h_ = boot(units, draws, random.Random(seed + 7), True, lo_c, hi_c)
        if pt is None or l_ is None:
            print("  对比:某一档在太多单位里为空,配对差算不出来")
            row["contrast"] = None
        else:
            print(f"  【主检验】对比 顶−底 = {pt:+.1%}  聚类 CI [{l_:+.1%},{h_:+.1%}]"
                  f"{'  ✓ 排除 0' if l_ > 0 else '  跨过 0'}")
            row["contrast"] = {"diff": pt, "ci": [l_, h_], "excludes_zero": bool(l_ > 0)}
        res[col] = row

    m = res.get("panel_recall_product", {})
    print(f"\n=== 判定([{pop}],按声明)===")
    if m.get("verdict") == "no_data_as_declared":
        print("  无数据 → 无结论。")
    elif pop == "crossnative":
        print("  探索性人群,按声明既不确认也不反驳,只作为无共享误差路径的一眼。")
    else:
        c = m.get("contrast") or {}
        top = m.get("顶档", {})
        print(f"  主检验(对比排除 0): {'✓ 成立' if c.get('excludes_zero') else '✗ 不成立'}")
        print(f"  次检验(顶档越过 0.5,更保守): {'✓ 成立' if top.get('clears_above') else '✗ 不成立'}")
        if not c.get("excludes_zero"):
            print("  按声明:人群天生偏向面板,失败【不可归因】,记为「测过、未拿到」而不是判据被反驳。")
        res["A9_CONTRAST_PASS"] = bool(c.get("excludes_zero"))
        res["A9_LEVEL_PASS"] = bool(top.get("clears_above"))
    out = os.path.join(HERE, f"arbitration_a9_{pop}.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True, choices=["nested", "crossnative"])
    ap.add_argument("--stage", required=True, choices=["build", "rnet", "panel", "analyse"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"build": lambda: build(a.pop), "rnet": lambda: stage_rnet(a.pop, a.workers),
     "panel": lambda: stage_panel(a.pop, a.workers),
     "analyse": lambda: analyse(a.pop, a.draws, a.seed)}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
