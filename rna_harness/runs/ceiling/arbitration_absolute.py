#!/usr/bin/env python3
"""A8: does the arbitration criterion hold against a truth with no noise in it? Declared before running.

See `arbitration_absolute_declaration.json`, committed before any number here existed. In short: A6 and A7
scored the criterion against measured tube values, and this sample put the tube's own replicate noise at 4.1
points, so both are RELATIVE results. Objective 4's protocol asks for absolute sensitivity. Natives give it:
a wild type does form its published structure, a random sequence of the same length essentially does not, so
in a (native, random) pair the right answer is certain.

THREE STAGES, because the two rulers live in different environments:

    --stage build   the population: natives matched by pair set, 7 randoms each, cells of 8. Writes the
                    sequence list only, so both scorers read the same molecules.
    --stage rnet    RibonanzaNet crossed-pair F1, via pool_rerank_score._score_chunk unchanged so the numbers
                    are the ones every prior replay used. RUNS IN /data/wenhui/rnet_env.
    --stage panel   the three shipped oracles, via panel_f1_designs.work unchanged. RUNS IN the `rna` env.
    --stage analyse the declared test. Reuses the cuts already frozen and committed per panel column; they
                    are NOT refitted on this population.

WHAT MAKES THIS DIFFERENT FROM calibrate_precision.py, which also folds natives and randoms: that run asks
whether a single design passes a rung. This one asks which of two designs a PAIR of rulers should be believed
about, which is a question about the rulers rather than about the designs, and it is the question objective 4
scores. It reuses that run's population so the two are comparable.

COST: $0. All local.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, random, statistics as st, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

N_RANDOM = 7                      # cell size 8, declared
POP = os.path.join(HERE, "arbitration_absolute_pop.json")
RNET = os.path.join(HERE, "arbitration_absolute_rnet.json")
PANEL = os.path.join(HERE, "arbitration_absolute_panel.json")


def build():
    """Natives matched by pair set, plus N_RANDOM same-length randoms each. Deterministic."""
    from rna_harness.experiments.axis1_native_control import parse_natives
    from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs
    nat = parse_natives()
    tasks = {t.task_id: t.target for t in load_pseudobase()}
    cells = []
    for tid, tgt in sorted(tasks.items()):
        hit = nat.get((len(tgt), frozenset(pk_pairs(tgt))))
        if not hit:
            continue
        seq = hit[1]
        # a per-target stream so adding a target never changes another target's randoms
        rng = random.Random(int(hashlib.sha1(f"A8|{tid}".encode()).hexdigest()[:12], 16))
        rows = [{"id": tid, "kind": "native", "seq": seq}]
        seen = {seq}
        while len(rows) < 1 + N_RANDOM:
            r = "".join(rng.choice("ACGU") for _ in seq)
            if r in seen:
                continue
            seen.add(r)
            rows.append({"id": tid, "kind": "random", "seq": r})
        cells.append({"id": tid, "target": tgt, "rows": rows})
    n = sum(len(c["rows"]) for c in cells)
    print(f"匹配上的野生型 {len(cells)} 条,每格 1+{N_RANDOM},序列合计 {n}")
    json.dump({"n_cells": len(cells), "n_seqs": n, "n_random_per_cell": N_RANDOM, "cells": cells},
              open(POP, "w"))
    print(f"写出 {POP}")


def stage_rnet(workers):
    from pool_rerank_score import _score_chunk
    P = json.load(open(POP))
    jobs = [((c["id"], i), r["seq"], c["target"]) for c in P["cells"] for i, r in enumerate(c["rows"])]
    print(f"{len(jobs)} 条序列过 RibonanzaNet,{workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    CH = max(1, len(jobs) // (workers * 8))
    sc, t0, done = {}, time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_score_chunk, [jobs[i:i + CH] for i in range(0, len(jobs), CH)]):
            sc.update(dict(res)); done += 1
            if done % 10 == 0:
                print(f"    {len(sc)}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump({"n": len(sc), "minutes": (time.time() - t0) / 60,
               "f1c": {f"{k[0]}|{k[1]}": v for k, v in sc.items()}}, open(RNET, "w"))
    print(f"完成 {len(sc)}/{len(jobs)},{(time.time()-t0)/60:.1f} 分钟 -> {RNET}")


def stage_panel(workers):
    from panel_f1_designs import work
    P = json.load(open(POP))
    jobs = [(f"{c['id']}|{i}", r["seq"], c["target"], 0.0, 0.0, r["kind"])
            for c in P["cells"] for i, r in enumerate(c["rows"])]
    print(f"{len(jobs)} 条序列过三家出厂判官,{workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, res in enumerate(ex.map(work, jobs, chunksize=4), 1):
            R.append(res)
            if n % 200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    bad = [r for r in R if "error" in r]
    print(f"  成功 {len(R)-len(bad)},失败 {len(bad)}"
          + (f" 例:{bad[0]['error'][:70]}" if bad else ""))
    json.dump({"n": len(R) - len(bad), "n_failed": len(bad), "minutes": (time.time() - t0) / 60,
               "rows": [{"key": r["puzzle"], "kind": r["method"],
                         "panel_recall_product": r["panel_recall_product"],
                         "panel_f1": r["panel_f1"]} for r in R if "error" not in r]}, open(PANEL, "w"))
    print(f"写出 {PANEL}")


def ranks01(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = ((i + j) / 2.0) / max(1, len(vals) - 1)
        i = j + 1
    return out


def cluster(percell, draws, rng):
    """One vote per cell. Each cell's fraction is fixed, so a draw is a mean of resampled cell fractions."""
    fs = [sum(1 for p in v if p["learned_right"]) / len(v) for v in percell.values()]
    if not fs:
        return None, None, None
    n = len(fs)
    out = sorted(sum(fs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return sum(fs) / n, out[int(0.025 * draws)], out[min(draws - 1, int(0.975 * draws))]


def analyse(draws, seed):
    D = json.load(open(os.path.join(HERE, "arbitration_absolute_declaration.json")))
    P = json.load(open(POP)); RN = json.load(open(RNET))["f1c"]; PA = json.load(open(PANEL))["rows"]
    pan = {r["key"]: r for r in PA}
    res = {"declaration": D}
    print(f"格 {P['n_cells']},序列 {P['n_seqs']};RNet 打分 {len(RN)},面板打分 {len(pan)}")

    for col in ("panel_recall_product", "panel_f1"):
        cuts = D["cuts_reused_not_refitted"][col]
        tag = "出厂列(主)" if col == "panel_recall_product" else "观测列(次)"
        print(f"\n=== {tag} {col};切点复用 顶 {cuts['top']:.5f} 底 {cuts['bottom']:.5f}(未重拟合)===")
        cells, dropped = {}, 0
        for c in P["cells"]:
            rows = []
            for i, r in enumerate(c["rows"]):
                k = f"{c['id']}|{i}"
                if k in RN and k in pan:
                    rows.append({"kind": r["kind"], "f1c": RN[k], "panel": pan[k][col]})
            if len(rows) < 3:
                dropped += 1; continue
            vf, vp = [x["f1c"] for x in rows], [x["panel"] for x in rows]
            if st.pstdev(vf) == 0 or st.pstdev(vp) == 0:
                dropped += 1; continue           # a ruler with no spread in this cell has no opinions
            for x, a_, b_ in zip(rows, ranks01(vf), ranks01(vp)):
                x["rf"], x["rp"] = a_, b_
            ps = []
            for i in range(len(rows)):
                for j in range(len(rows)):
                    if i >= j:
                        continue
                    x, y = rows[i], rows[j]
                    if x["kind"] == y["kind"]:            # random-random carries no truth
                        continue
                    nat, rnd = (x, y) if x["kind"] == "native" else (y, x)
                    dr, dp = nat["f1c"] - rnd["f1c"], nat["panel"] - rnd["panel"]
                    if dr == 0 or dp == 0 or (dr > 0) == (dp > 0):
                        continue                          # agreement, or a ruler with no opinion
                    ps.append({"learned_right": dr > 0,   # truth = the native is better
                               "d": abs(nat["rf"] - rnd["rf"]) - abs(nat["rp"] - rnd["rp"])})
            if ps:
                cells[c["id"]] = ps
        npair = sum(len(v) for v in cells.values())
        print(f"  可用格 {len(cells)}(丢弃 {dropped}),相悖的(野生型,随机)配对 {npair}")
        row = {"n_cells": len(cells), "n_pairs": npair, "n_cells_dropped": dropped}
        if npair < 100:
            print(f"  按声明:相悖配对 < 100 → 本检验【无数据】,记无结论。两把尺子在野生型-随机上几乎不相悖。")
            row["verdict"] = "no_data_as_declared"
            res[col] = row; continue
        for name, lo, hi in (("底档", -1e9, cuts["bottom"]), ("中档", cuts["bottom"], cuts["top"]),
                             ("顶档", cuts["top"], 1e9)):
            pc = {k: [p for p in v if lo < p["d"] <= hi] for k, v in cells.items()}
            pc = {k: v for k, v in pc.items() if v}
            n = sum(len(v) for v in pc.values())
            if n < 20:
                print(f"  {name} n={n} —— 太少,不给数"); row[name] = {"n": n}; continue
            pt, l_, h_ = cluster(pc, draws, random.Random(seed))
            print(f"  {name} n={n:5d} 格{len(pc):3d}  跟学习型正确率 {pt:.1%} [{l_:.1%},{h_:.1%}]"
                  f"{'  ✓ 上越 0.5' if l_ > 0.5 else ('  ✓ 下越 0.5' if h_ < 0.5 else '  跨 0.5')}")
            row[name] = {"n": n, "n_cells": len(pc), "frac": pt, "ci": [l_, h_],
                         "clears_above": bool(l_ > 0.5), "clears_below": bool(h_ < 0.5)}
        res[col] = row

    m = res["panel_recall_product"]
    print(f"\n=== 判定(按声明,主口径 = 出厂列 × 秩差 × 每格一票)===")
    if m.get("verdict") == "no_data_as_declared":
        print("  无数据,按声明记无结论 —— ④ 这一格仍缺绝对灵敏度。")
        res["A8_PASS"] = None
    else:
        ok = bool(m.get("顶档", {}).get("clears_above"))
        one = not bool(m.get("底档", {}).get("clears_above"))
        print(f"  顶档 {'✓ 越过 0.5 —— 判据在确定真值下也成立' if ok else '✗ 跨过 0.5 —— 绝对灵敏度拿不到'}")
        print(f"  底档 {'未从上方越过 —— 单侧性在确定真值下同样成立' if one else '也越过了 —— 这里是双侧的'}")
        res["A8_PASS"] = ok; res["A8_ONE_SIDED"] = one
    out = os.path.join(HERE, "arbitration_absolute.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "rnet", "panel", "analyse"])
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"build": lambda: build(), "rnet": lambda: stage_rnet(a.workers),
     "panel": lambda: stage_panel(a.workers), "analyse": lambda: analyse(a.draws, a.seed)}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
