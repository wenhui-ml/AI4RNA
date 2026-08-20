#!/usr/bin/env python
"""J1 -- can the pKiss penalty observation D1 threw off be turned into a gain on the wet-lab ruler?

Pre-registration: j1_pkiss_retune_declaration.json (commit ed3e96dd), which states its own honest ceiling:
F0's whole net R^2 is 0.047 across three oracles, 110 cells give intervals of order 0.02, so the plausible
effect is BELOW the detection floor and a null here is a power statement, not evidence of no effect.

D1 itself did not hold (primary no_data, mirror no_specificity) and none of its conclusions are inherited.
What is inherited is one level observation: raising --Kpenalty lowered spurious crossings AND raised true
crossing recall on both topologies at once, while --Hpenalty's default 9 had no room below it and was
destructive above it.  The candidates were picked after seeing those tables -- not blind on the criterion
side -- but they were picked on Pseudobase designs that carry no wet-lab label, and the population tested
here is our own design pools, which the selection never touched.

The void gate is the one A16 got wrong.  A16 asked whether its flag had taken effect by checking that a GC
interval excluded zero -- a statistic standing in for a question that had an unambiguous direct answer in
the same run.  Here the gate IS the direct observation: count the designs whose confirm count k changes,
and a configuration that changes none of them did nothing and reports nothing.

C4 = (H12, K14) exists in order to be worse.  D1 says an H penalty at or above 10 tears the true crossings
out, so if C4 does not come out negative then that reading never reached this population and no positive
result from the other candidates may be called a gain.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402
from r1_info_ladder import ORACLES, FINE, r2_within, demean_by_cell             # noqa: E402

PKISS = "/home/ubuntu/miniconda3/envs/pk/bin/pKiss"
CAND = {"C0": (9.0, 12.0), "C1": (9.0, 14.0), "C2": (6.0, 14.0), "C3": (4.0, 14.0), "C4": (12.0, 14.0)}
HYPOTHESES = ("C1", "C2", "C3")
CONTROL = "C4"
ALPHA = 0.05 / len(HYPOTHESES)                     # Bonferroni over the three hypotheses, declared
FLOOR_CELLS = 20
NBOOT = 1000
NPERM_POINT = 200                                  # permutations for the reported point net R^2
NPERM_BOOT = 8                                     # permutations inside each bootstrap replicate (common
                                                   # random numbers across configs, so the nulls cancel)
POPS = [("第四批", "panel_f1_fourth.json"), ("样本3", "panel_f1_third.json")]


def crossing_subset(pairs):
    ps = sorted(pairs); out = set()
    for a, (i, j) in enumerate(ps):
        for k, l in ps[a + 1:]:
            if i < k < j < l or k < i < l < j:
                out.add((i, j)); out.add((k, l))
    return out


def _fold(job):
    seq, h, k = job
    try:
        r = subprocess.run([PKISS, "--mode", "mfe", "--Hpenalty", str(h), "--Kpenalty", str(k), seq],
                           capture_output=True, text=True, timeout=900)
    except Exception as e:                                                       # noqa: BLE001
        return (seq, h, k, None, repr(e)[:80])
    out = (r.stdout or "").strip().split()
    if r.returncode != 0 or not out or len(out[-1]) != len(seq):
        return (seq, h, k, None, f"rc={r.returncode}")
    return (seq, h, k, out[-1], None)


def pkiss_cols(struct, cross):
    """pKiss's six columns, using panel_f1_designs.py's definitions VERBATIM -- including the deliberate
    distinction between recall_product (credit for the pair even if the oracle did not cross it) and
    recall (restricted to the oracle's own crossing subset, the only convention under which precision
    means anything).  Any drift here and the comparison is not between the same two things."""
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    ps = pk_pairs(struct)
    pc = crossing_subset(ps)
    rec_prod = len(ps & cross) / len(cross)
    hit = len(pc & cross)
    rec = hit / len(cross)
    pre = (hit / len(pc)) if pc else 0.0
    f1 = (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else 0.0
    return {"recall_product": rec_prod, "recall": rec, "precision": pre, "f1": f1,
            "n_pred_cross": len(pc), "n_pred_total": len(ps)}


def blocks(po):
    """(F0, F1extra, F2extra) for one molecule from a per_oracle dict.  Same shape as R1's feats()."""
    rp = [po[o]["recall_product"] for o in ORACLES]
    k = sum(1 for x in rp if x >= 1.0)
    f0 = [1.0 if k == j else 0.0 for j in (1, 2, 3)]
    fine = [float(po[o][c]) for o in ORACLES for c in FINE]
    return f0, [float(x) for x in rp], fine, k


def net_r2(y, X, sl, nperm, rng, ci):
    obs = r2_within(y, X[:, sl])
    vals = []
    for _ in range(nperm):
        yp = y.copy()
        for c in range(ci.max() + 1):
            m = ci == c
            yp[m] = rng.permutation(yp[m])
        vals.append(r2_within(yp, X[:, sl]))
    return obs - float(np.mean(vals)), obs, float(np.mean(vals))


def run_pop(name, path, targets, fold, workers, out):
    from rna_harness.benchmarks.rnainvbench import pk_pairs, _round_only
    rec, _sp, _meta = load_fresh(os.path.join(HERE, path), "panel_recall_product")
    keep = []
    for r in rec:
        po = r.get("per_oracle") or {}
        if sorted(po) != sorted(ORACLES) or not r.get("seq"):
            continue
        if any(po[o] is None or po[o].get("recall_product") is None for o in ORACLES):
            continue
        t = targets.get(r["puzzle"])
        if not t:
            continue
        cross = pk_pairs(t["target"]) - pk_pairs(_round_only(t["target"]))
        if not cross:
            continue
        r["_cross"] = cross
        keep.append(r)
    print(f"\n=== {name} ===\n  可用分子 {len(keep)} / {len(rec)}")

    # ---- per-config columns.  C0 comes straight out of the panel JSON; the rest are refolded.
    cols, kvals, missing = {}, {}, collections.Counter()
    for cn in CAND:
        cols[cn], kvals[cn] = {}, {}
    for r in keep:
        for cn, (h, k) in CAND.items():
            if cn == "C0":
                po = {o: dict(r["per_oracle"][o]) for o in ORACLES}
            else:
                st = fold.get(f"{r['seq']}|{h}|{k}")
                if st is None:
                    missing[cn] += 1
                    continue
                po = {o: dict(r["per_oracle"][o]) for o in ORACLES}
                po["pkiss"] = pkiss_cols(st, r["_cross"])
            f0, f1e, f2e, kk = blocks(po)
            cols[cn][id(r)] = f0 + f1e + f2e
            kvals[cn][id(r)] = kk
    if sum(missing.values()):
        print(f"  缺折叠结果 {dict(missing)}")

    usable = [r for r in keep if all(id(r) in cols[cn] for cn in CAND)]
    print(f"  五个配置都齐的分子 {len(usable)}")

    # ---- THE VOID GATE: an unambiguous direct observation, not a statistic
    print("\n  作废闸门(k 改变的条数 —— 直接观察,不是统计量):")
    gate = {}
    for cn in CAND:
        if cn == "C0":
            continue
        ch = [(kvals["C0"][id(r)], kvals[cn][id(r)]) for r in usable]
        n_ch = sum(1 for a, b in ch if a != b)
        up = sum(1 for a, b in ch if b > a); dn = sum(1 for a, b in ch if b < a)
        dist = collections.Counter(kvals[cn][id(r)] for r in usable)
        gate[cn] = n_ch
        flag = "  <<< 作废:什么都没做" if n_ch == 0 else ""
        print(f"    {cn} (H{CAND[cn][0]:.0f}/K{CAND[cn][1]:.0f})  k 改变 {n_ch}/{len(usable)}"
              f"  升 {up} 降 {dn}   k 分布 {dict(sorted(dist.items()))}{flag}")
    d0 = collections.Counter(kvals["C0"][id(r)] for r in usable)
    print(f"    C0 (出厂 H9/K12)                              k 分布 {dict(sorted(d0.items()))}")

    cells = collections.defaultdict(list)
    for r in usable:
        cells[(r["puzzle"], r["method"])].append(r)
    rows = [(c, g) for c, g in cells.items() if len(g) >= 3]
    NF = 3 + 3 + len(ORACLES) * len(FINE)
    print(f"\n  格 {len(rows)}(每格 >=3 条);F2 特征数 {NF}")
    if len(rows) < FLOOR_CELLS:
        print(f"  低于声明下限 {FLOOR_CELLS} 格 → 无数据")
        return {"pop": name, "no_data": True, "n_cells": len(rows)}

    BL = {"F0": slice(0, 3), "F2": slice(0, NF)}
    y0, _, ci = demean_by_cell(rows, cols["C0"])
    X = {cn: demean_by_cell(rows, cols[cn])[1] for cn in CAND}
    dof = len(y0) - NF - len(rows)
    print(f"  参与拟合 {len(y0)} 条;自由度 {len(y0)} − {NF} − {len(rows)} = {dof}")
    if dof <= 0:
        return {"pop": name, "no_data": True, "dof": dof}

    point = {}
    for lvl, sl in BL.items():
        for cn in CAND:
            rng = np.random.default_rng(12345)                  # same permutations for every config
            nr, obs, nul = net_r2(y0, X[cn], sl, NPERM_POINT, rng, ci)
            point[(lvl, cn)] = {"net": nr, "obs": obs, "null": nul}

    print(f"\n  点估计(置换 {NPERM_POINT} 次,所有配置共用同一批置换 → 零模型可比):")
    print(f"    {'级':4}{'配置':6}{'样本内 R^2':>12}{'置换零均值':>12}{'净 R^2':>10}{'Δ净 vs C0':>12}")
    for lvl in ("F0", "F2"):
        for cn in CAND:
            p = point[(lvl, cn)]
            d = p["net"] - point[(lvl, "C0")]["net"]
            print(f"    {lvl:4}{cn:6}{p['obs']:12.4f}{p['null']:12.4f}{p['net']:10.4f}"
                  f"{(('' if cn=='C0' else f'{d:+.4f}')):>12}")

    # ---- cluster bootstrap over cells, common random numbers across configs
    print(f"\n  cluster bootstrap {NBOOT} 次(按格重抽,每复本内所有配置共用同一批置换):")
    boots = {(l, c): [] for l in BL for c in CAND}
    rngb = np.random.default_rng(7)
    for b in range(NBOOT):
        pick = rngb.integers(0, len(rows), len(rows))
        sub = [rows[i] for i in pick]
        yb, _, cib = demean_by_cell(sub, cols["C0"])
        if len(yb) < NF + len(sub) + 1:
            continue
        Xb = {cn: demean_by_cell(sub, cols[cn])[1] for cn in CAND}
        seed = int(rngb.integers(0, 2 ** 31 - 1))
        for lvl, sl in BL.items():
            for cn in CAND:
                r2n, _o, _n = net_r2(yb, Xb[cn], sl, NPERM_BOOT, np.random.default_rng(seed), cib)
                boots[(lvl, cn)].append(r2n)
    res = {}
    print(f"    {'级':4}{'配置':6}{'Δ净 R^2 中位':>14}{'95% CI':>24}{'排除 0':>9}{'校正后':>9}")
    for lvl in ("F0", "F2"):
        for cn in CAND:
            if cn == "C0":
                continue
            a = np.array(boots[(lvl, cn)]); z = np.array(boots[(lvl, "C0")])
            n = min(len(a), len(z))
            d = a[:n] - z[:n]
            if n < 100:
                continue
            med = float(np.median(d))
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            clo = float(np.percentile(d, 100 * ALPHA / 2)); chi = float(np.percentile(d, 100 * (1 - ALPHA / 2)))
            exc = "是" if (lo > 0 or hi < 0) else "否"
            cexc = "是" if (clo > 0 or chi < 0) else "否"
            res[(lvl, cn)] = {"median": med, "ci": [lo, hi], "ci_corrected": [clo, chi],
                              "excl": exc == "是", "excl_corrected": cexc == "是", "nboot": n}
            print(f"    {lvl:4}{cn:6}{med:+14.4f}   [{lo:+.4f},{hi:+.4f}]{exc:>9}{cexc:>9}")
    out.append({"pop": name, "n_usable": len(usable), "n_cells": len(rows), "gate": gate,
                "point": {f"{l}|{c}": point[(l, c)] for l, c in point},
                "boot": {f"{l}|{c}": res[(l, c)] for l, c in res}})
    return out[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cache", default=os.path.join(HERE, "j1_fold.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "j1_pkiss_retune.json"))
    a = ap.parse_args()

    from leaderboard_design import targets
    T = targets()

    seqs = set()
    for _n, p in POPS:
        d = json.load(open(os.path.join(HERE, p)))
        for r in d["designs"]:
            if r.get("seq") and r["puzzle"] in T:
                seqs.add(r["seq"])
    print(f"要折叠的唯一序列 {len(seqs)};新配置 {len(CAND)-1} 个")

    fold = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    jobs = sorted({(s, h, k) for s in seqs for cn, (h, k) in CAND.items()
                   if cn != "C0" and f"{s}|{h}|{k}" not in fold})
    print(f"  待折叠 {len(jobs)} 次,{a.workers} 进程")
    fails = collections.Counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, (s, h, k, st, err) in enumerate(ex.map(_fold, jobs, chunksize=4), 1):
                if st is None:
                    fails[err] += 1
                else:
                    fold[f"{s}|{h}|{k}"] = st
                if n % 2000 == 0:
                    print(f"    folded {n}/{len(jobs)}  失败 {sum(fails.values())}", flush=True)
                    json.dump(fold, open(a.cache, "w"))
        json.dump(fold, open(a.cache, "w"))
    print(f"  折叠失败 {sum(fails.values())} {dict(fails)};缓存 {len(fold)}")

    out = []
    for nm, p in POPS:
        run_pop(nm, p, T, fold, a.workers, out)

    # ---- verdict, strictly by the declaration's outcome table, on the PRIMARY population and level 1
    pri = out[0] if out else None
    print("\n=== 判读(按声明:主人群 = 第四批,主检验 = level 1 = F0 = 出厂档位 k)===")
    verdict, why = "no_data", "主人群没有可用结果"
    if pri and not pri.get("no_data"):
        voided = [c for c in HYPOTHESES if pri["gate"].get(c, 0) == 0]
        ctl = pri["boot"].get(f"F0|{CONTROL}")
        if len(voided) == len(HYPOTHESES):
            verdict, why = "void", f"C1 C2 C3 全部作废(k 改变条数为 0):{voided}"
        elif ctl is None:
            verdict, why = "void_control", "负对照 C4 拿不到区间"
        elif not (ctl["ci_corrected"][1] < 0 or ctl["ci"][1] < 0):
            verdict = "void_control"
            why = (f"负对照 C4 的 Δ净 R^2 {ctl['median']:+.4f} CI {ctl['ci']} 【不】显著为负 —— "
                   "D1 那个『Hpen>=10 拆掉真交叉』的读法没有传递到这个人群,J1 的前提不成立,"
                   "任何候选的正结果都不许写成收益")
        else:
            wins = [c for c in HYPOTHESES if c not in voided
                    and (pri["boot"].get(f"F0|{c}") or {}).get("excl_corrected")
                    and (pri["boot"][f"F0|{c}"]["median"] > 0)]
            bad = [c for c in HYPOTHESES if c not in voided
                   and (pri["boot"].get(f"F0|{c}") or {}).get("excl_corrected")
                   and (pri["boot"][f"F0|{c}"]["median"] < 0)]
            if wins:
                verdict = "gain"
                why = f"校正水平上排除 0 且为正:{wins};负对照 C4 {ctl['median']:+.4f} 显著为负"
            elif bad:
                verdict = "worse"
                why = f"显著为负:{bad} —— 重调是有害的,出厂配置比我以为的更接近局部最优"
            else:
                verdict = "null"
                why = "没有候选在校正水平上排除 0"
        if voided and verdict != "void":
            why += f";作废的候选 {voided}(k 改变条数为 0,不许报它们的数)"
    print(f"  结论 {verdict}")
    print(f"  {why}")
    print("  功效如实说明(声明里已写在跑之前):110 个格、ΔR^2 区间宽度 0.02 量级,所以本实验只能探测")
    print("  0.01 量级以上的净 R^2 变化;而 F0 的净 R^2 总共只有 0.047、pKiss 是三个判官之一,")
    print("  可信的效应量很可能【小于探测下限】。【所以 null 是一句功效陈述,不是『没有效应』的证据。】")
    print("  而且就算全胜也不改变本周主线:热力学面板 21 列净 R^2 = 0.103,RibonanzaNet 单列 = 0.164。")

    json.dump({"verdict": verdict, "why": why, "alpha_corrected": ALPHA,
               "candidates": {k: list(v) for k, v in CAND.items()}, "pops": out},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
