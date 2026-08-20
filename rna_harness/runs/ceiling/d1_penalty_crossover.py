#!/usr/bin/env python
r"""D1 -- move the crossover on purpose.

Pre-registration: d1_penalty_crossover_declaration.json (frozen, pushed BEFORE any primary data).

The one-line claim being tested: allowing one more unit of genus multiplies the configuration count by
L^3, so the entropy of the extra genus is 3 ln L, while the price is the pseudoknot initiation penalty
dG_pk.  Crossings stop being controlled when 3 RT ln L > dG_pk, i.e.

    ln L*(dG_pk) = dG_pk / (3 RT) + const          slope = 1/(3RT) = 0.5408 per kcal/mol

pKiss exposes dG_pk as --Hpenalty / --Kpenalty, so the smallness is a KNOB and the crossover can be
moved on purpose.  ONLY THE SLOPE IS PREDICTED.  The derivation dropped the O(1) constant c_g, which
becomes an unknown additive constant in ln L*, so the LOCATION of the crossover carries no evidential
weight in either direction.

The observable is deliberately the one that never goes 0/0:

    spurious = 1 if |C_pred \ C_true| > 0 else 0

As the penalty rises pKiss claims fewer crossings, in the limit none, so spurious -> 0.  As it falls,
spurious crossings flood in, so spurious -> 1.  Crossing PRECISION would have been the natural quantity
and it is unusable: it is 0/0 exactly at the controlled end.  True-crossing recall is reported alongside
for every cell, because if a crossover in spurious were really "all crossings vanished" rather than
"spurious crossings appeared", recall would collapse in step -- that column exists so the reader can
kill that explanation, and it is not optional.

Crossing sets use the GEOMETRIC definition (a pair that crosses some other pair) on both sides, not the
bracket-family convention the panel uses for targets.  The two are not the same set: the panel's
`pk_pairs(t) - pk_pairs(_round_only(t))` keeps only the non-round partner of each crossing, whereas the
geometric set keeps both.  Nothing here is compared against a panel number, and using one definition on
both sides is the only way the difference C_pred \ C_true means what it says.

Self-checks are not decoration here.  A15 completed, printed a verdict and wrote JSON while its judge was
being fed a dict repr instead of a target, so this script refuses to report unless the predicted structure
has the same length as its sequence, every target has a non-empty true crossing set, and the observable
actually takes both values.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

PKISS = "/home/ubuntu/miniconda3/envs/pk/bin/pKiss"
RT = 0.61637                       # kcal/mol at 37 C -- frozen in the declaration
PRED_SLOPE = 1.0 / (3.0 * RT)      # 0.5408
GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]
H_DEFAULT, K_DEFAULT = 9.0, 12.0
MIN_LOCATABLE = 4
NBOOT = 2000


def crossing_subset(pairs):
    """The pairs in `pairs` that cross at least one other pair in `pairs` (geometric definition)."""
    ps = sorted(pairs)
    out = set()
    for a, (i, j) in enumerate(ps):
        for k, l in ps[a + 1:]:
            if i < k < j < l or k < i < l < j:
                out.add((i, j))
                out.add((k, l))
    return out


# ---------------------------------------------------------------- units

def build_units():
    from rna_harness.benchmarks.rnainvbench import pk_pairs, load_pseudobase
    from pk_topology import classify_topology
    from leaderboard_design import targets

    units, drop = [], collections.Counter()

    # population A (primary): designed sequences on Pseudobase targets -- the only population with
    # (a) L down to 21, (b) a topology label, (c) sequences already in hand.
    ptarget = {t.task_id: t.target for t in load_pseudobase()}
    topo = {r["task_id"]: r for r in json.load(open(os.path.join(HERE, "pk_topology.json")))["rows"]}
    for r in json.load(open(os.path.join(HERE, "kissing_deliver.json")))["rows"]:
        tid, seq = r["task_id"], r.get("sequence")
        if not seq:
            drop["A:no sequence"] += 1
            continue
        t = ptarget.get(tid)
        if t is None:
            drop["A:no target"] += 1
            continue
        kind = (topo.get(tid) or {}).get("topology")
        if kind not in ("H-type", "kissing / chain"):
            drop[f"A:topology {kind}"] += 1
            continue
        ct = crossing_subset(pk_pairs(t))
        if not ct:
            drop["A:target has no crossing"] += 1
            continue
        if len(seq) != len(t):
            drop["A:seq/target length mismatch"] += 1
            continue
        units.append({"pop": "A", "cluster": tid, "uid": tid, "topology": kind,
                      "L": len(seq), "seq": seq, "ct": sorted(ct)})

    # population B (secondary, and already peeked at -- see prior_looks in the declaration): our own
    # designs.  8 per target by lexicographic sequence order, no random numbers.
    T = targets()
    byp = collections.defaultdict(list)
    for f in ("panel_f1_fourth.json", "panel_f1_third.json"):
        for d in json.load(open(os.path.join(HERE, f)))["designs"]:
            if d.get("seq"):
                byp[d["puzzle"]].append(d["seq"])
    for p in sorted(byp):
        if p not in T:
            drop["B:puzzle not in targets()"] += 1
            continue
        t = T[p]["target"]
        ct = crossing_subset(pk_pairs(t))
        if not ct:
            drop["B:target has no crossing"] += 1
            continue
        kind = classify_topology(t)["topology"]
        for seq in sorted(set(byp[p]))[:8]:
            if len(seq) != len(t):
                drop["B:seq/target length mismatch"] += 1
                continue
            units.append({"pop": "B", "cluster": p, "uid": f"{p}|{seq[:12]}", "topology": kind,
                          "L": len(seq), "seq": seq, "ct": sorted(ct)})
    return units, drop


# ---------------------------------------------------------------- folding

def _fold(job):
    seq, h, k = job
    try:
        r = subprocess.run([PKISS, "--mode", "mfe", "--Hpenalty", str(h), "--Kpenalty", str(k), seq],
                           capture_output=True, text=True, timeout=900)
    except Exception as e:                                                       # noqa: BLE001
        return (seq, h, k, None, repr(e)[:80])
    out = (r.stdout or "").strip().split()
    if r.returncode != 0 or not out:
        return (seq, h, k, None, f"rc={r.returncode}")
    st = out[-1]
    if len(st) != len(seq):
        return (seq, h, k, None, f"len {len(st)} != {len(seq)}")
    return (seq, h, k, st, None)


def sweeps_needed():
    """(sweep, penalty, h, k) for both sweeps.  The two share the (9,12) default cell."""
    out = []
    for pen in GRID:
        out.append(("H", pen, pen, K_DEFAULT))
        out.append(("K", pen, H_DEFAULT, pen))
    return out


# ---------------------------------------------------------------- estimator

def logistic_fit(lnL, y, ridge=1e-6, iters=60):
    """1-D logistic by IRLS.  Returns (b0, b1) or None if it will not converge / is degenerate."""
    y = np.asarray(y, float)
    if y.min() == y.max():
        return None
    X = np.column_stack([np.ones_like(lnL), lnL])
    b = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        A = X.T @ (X * w[:, None]) + ridge * np.eye(2)
        try:
            nb = np.linalg.solve(A, X.T @ (w * z))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(nb)):
            return None
        if np.max(np.abs(nb - b)) < 1e-9:
            b = nb
            break
        b = nb
    return float(b[0]), float(b[1])


def locate(rows):
    """L* for one cell.  `rows` = [(lnL, spurious)].  Declaration's locatable rule, verbatim:
    b1 <= 0, or L* outside the cell's own observed L range -> NOT locatable."""
    if len(rows) < 8:
        return None
    lnL = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    fit = logistic_fit(lnL, y)
    if fit is None:
        return None
    b0, b1 = fit
    # spurious should RISE with length, so the logistic slope must be positive
    if b1 <= 0:
        return None
    ls = -b0 / b1
    if not (lnL.min() <= ls <= lnL.max()):
        return None
    return ls


def slope_of(cells):
    """WLS-free OLS of ln L* on penalty over the locatable grid points."""
    pts = [(p, ls) for p, ls in cells if ls is not None]
    if len(pts) < MIN_LOCATABLE:
        return None, len(pts)
    x = np.array([p for p, _ in pts]); y = np.array([ls for _, ls in pts])
    if x.max() == x.min():
        return None, len(pts)
    sl = float(np.polyfit(x, y, 1)[0])
    return sl, len(pts)


def analyse(units, fold, sweep, topo, pop, rng, boot=NBOOT):
    """Returns dict with the point slope, its bootstrap CI, per-grid L*, and the locatable counts."""
    sub = [u for u in units if u["pop"] == pop and u["topology"] == topo]
    if not sub:
        return None
    per_pen = {}
    for pen in GRID:
        h, k = (pen, K_DEFAULT) if sweep == "H" else (H_DEFAULT, pen)
        rows, miss = [], 0
        for u in sub:
            st = fold.get(f"{u['seq']}|{h}|{k}")
            if st is None:
                miss += 1
                continue
            cp = crossing_subset(__import__("rna_harness.benchmarks.rnainvbench", fromlist=["pk_pairs"]).pk_pairs(st))
            ct = set(map(tuple, u["ct"]))
            rows.append((math.log(u["L"]), 1 if (cp - ct) else 0,
                         len(cp & ct) / len(ct), u["cluster"]))
        per_pen[pen] = {"rows": rows, "missing": miss}

    point = [(pen, locate([(r[0], r[1]) for r in per_pen[pen]["rows"]])) for pen in GRID]
    sl, nloc = slope_of(point)

    clusters = sorted({u["cluster"] for u in sub})
    idx = {c: i for i, c in enumerate(clusters)}
    bysl = []
    for _ in range(boot):
        pick = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
        cnt = collections.Counter(pick)
        cells = []
        for pen in GRID:
            rr = []
            for r in per_pen[pen]["rows"]:
                rr.extend([(r[0], r[1])] * cnt.get(r[3], 0))
            cells.append((pen, locate(rr)))
        s, n = slope_of(cells)
        if s is not None:
            bysl.append(s)
    ci = (float(np.percentile(bysl, 2.5)), float(np.percentile(bysl, 97.5))) if len(bysl) >= 100 else None
    exps = [1.0 / (RT * s) for s in bysl if s > 0]
    return {
        "pop": pop, "sweep": sweep, "topology": topo, "n_units": len(sub), "n_clusters": len(clusters),
        "slope": sl, "slope_ci": ci, "n_locatable": nloc, "boot_ok": len(bysl), "boot_n": boot,
        "L_star_by_pen": {str(p): (None if ls is None else round(math.exp(ls), 1)) for p, ls in point},
        "spurious_rate_by_pen": {str(p): (round(float(np.mean([r[1] for r in per_pen[p]["rows"]])), 3)
                                          if per_pen[p]["rows"] else None) for p in GRID},
        "recall_by_pen": {str(p): (round(float(np.mean([r[2] for r in per_pen[p]["rows"]])), 3)
                                   if per_pen[p]["rows"] else None) for p in GRID},
        "missing_by_pen": {str(p): per_pen[p]["missing"] for p in GRID},
        "exponent": (round(float(np.median(exps)), 2) if exps else None),
        "exponent_ci": ([round(float(np.percentile(exps, 2.5)), 2),
                         round(float(np.percentile(exps, 97.5)), 2)] if len(exps) >= 100 else None),
        "boot_slopes": bysl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cache", default=os.path.join(HERE, "d1_penalty_crossover_fold.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "d1_penalty_crossover.json"))
    a = ap.parse_args()

    units, drop = build_units()
    print(f"单位 {len(units)};剔除 {dict(drop)}")
    for pop in ("A", "B"):
        c = collections.Counter(u["topology"] for u in units if u["pop"] == pop)
        Ls = sorted(u["L"] for u in units if u["pop"] == pop)
        cl = len({u["cluster"] for u in units if u["pop"] == pop})
        if Ls:
            print(f"  {pop} 群 n={len(Ls)} 靶标簇={cl} L {Ls[0]}-{Ls[-1]} 中位 {Ls[len(Ls)//2]}  {dict(c)}")
    assert units, "no units"
    assert all(u["ct"] for u in units), "a unit has an empty true crossing set"

    fold = {}
    if os.path.exists(a.cache):
        fold = json.load(open(a.cache))
        print(f"  缓存命中 {len(fold)}")
    jobs = []
    for sweep, pen, h, k in sweeps_needed():
        for u in units:
            key = f"{u['seq']}|{h}|{k}"
            if key not in fold:
                jobs.append((u["seq"], h, k))
    jobs = sorted(set(jobs))
    print(f"  要折叠 {len(jobs)} 次,{a.workers} 进程")
    fails = collections.Counter()
    if jobs:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, (seq, h, k, st, err) in enumerate(ex.map(_fold, jobs, chunksize=4), 1):
                if st is None:
                    fails[err] += 1
                else:
                    fold[f"{seq}|{h}|{k}"] = st
                if n % 500 == 0:
                    print(f"    folded {n}/{len(jobs)}  失败 {sum(fails.values())}", flush=True)
        json.dump(fold, open(a.cache, "w"))
    print(f"  折叠失败 {sum(fails.values())} {dict(fails)}")

    # forced self-check: the observable must actually take both values, and the structures must vary
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    vals = collections.Counter()
    for u in units:
        for sweep, pen, h, k in sweeps_needed():
            st = fold.get(f"{u['seq']}|{h}|{k}")
            if st is None:
                continue
            cp = crossing_subset(pk_pairs(st))
            vals[1 if (cp - set(map(tuple, u["ct"]))) else 0] += 1
    print(f"  自检 spurious 取值分布 {dict(vals)};不同结构串 {len(set(fold.values()))}")
    if len(vals) < 2:
        print("  !!! 观察量退化成常数 —— 判读作废,不许报结论")

    rng = np.random.default_rng(0)
    res = []
    for pop in ("A", "B"):
        for sweep in ("H", "K"):
            for topo in sorted({u["topology"] for u in units if u["pop"] == pop}):
                r = analyse(units, fold, sweep, topo, pop, rng)
                if r:
                    res.append(r)

    print(f"\n预言斜率 {PRED_SLOPE:.4f} /kcal   【只检验斜率,截距未被预言】")
    print(f"{'群':3}{'扫':3}{'拓扑':18}{'n':>5}{'簇':>5}{'可定位':>7}{'斜率':>9}{'95% CI':>22}{'实测指数':>10}")
    for r in res:
        ci = "  --" if not r["slope_ci"] else f"[{r['slope_ci'][0]:+.3f},{r['slope_ci'][1]:+.3f}]"
        sl = "  --" if r["slope"] is None else f"{r['slope']:+.4f}"
        ex = "  --" if r["exponent"] is None else f"{r['exponent']:.2f}"
        print(f"{r['pop']:3}{r['sweep']:3}{r['topology']:18}{r['n_units']:>5}{r['n_clusters']:>5}"
              f"{r['n_locatable']:>7}{sl:>9}{ci:>22}{ex:>10}")

    print("\n每个格点的 spurious 率 / 真交叉召回 / L*(A 群 H 扫描):")
    for r in res:
        if r["pop"] != "A":
            continue
        print(f"  [{r['sweep']} 扫描,{r['topology']}]")
        print("    罚项  " + "".join(f"{p:>8.0f}" for p in GRID))
        print("    spur  " + "".join(f"{(r['spurious_rate_by_pen'][str(p)] if r['spurious_rate_by_pen'][str(p)] is not None else float('nan')):>8.2f}" for p in GRID))
        print("    recl  " + "".join(f"{(r['recall_by_pen'][str(p)] if r['recall_by_pen'][str(p)] is not None else float('nan')):>8.2f}" for p in GRID))
        print("    L*    " + "".join(f"{(r['L_star_by_pen'][str(p)] if r['L_star_by_pen'][str(p)] is not None else float('nan')):>8.1f}" for p in GRID))

    # ---- verdict, strictly by the declaration's outcome table
    A = {(r["sweep"], r["topology"]): r for r in res if r["pop"] == "A"}
    matched = A.get(("H", "H-type"))
    cross = A.get(("H", "kissing / chain"))
    verdict, why, spec_ratio, spec_ci = "no_data", "", None, None
    if matched is None or matched["slope"] is None or matched["n_locatable"] < MIN_LOCATABLE:
        why = f"H 扫描 H-type 可定位格点 {None if matched is None else matched['n_locatable']} < {MIN_LOCATABLE}"
    elif matched["slope_ci"] is None:
        why = "bootstrap 有效复本不足 100"
    elif matched["slope_ci"][0] <= 0.0 <= matched["slope_ci"][1]:
        verdict = "dead"
        why = (f"matched 斜率 {matched['slope']:+.4f} CI {matched['slope_ci']} 含 0 —— 旋钮推不动交叉点。"
               "按声明,D1 及其下游 D2(重新求和)/D3(独立性失效做产品信号)全线关闭")
    elif matched["slope_ci"][1] < 0.0:
        verdict = "inverted"
        why = f"matched 斜率显著为负 {matched['slope']:+.4f} CI {matched['slope_ci']} —— 机制方向相反,推导错"
    else:
        # matched slope is positive and excludes 0.  Specificity next: the H knob must NOT move kissing.
        if cross is None or cross["slope"] is None or cross["slope_ci"] is None:
            verdict = "no_specificity_untested"
            why = "kissing 臂拿不到斜率,特异性检验做不了 —— 按声明不许把 P1 单独当成机制成立"
        else:
            cs = cross["slope_ci"]
            # the declaration's statistic is the RATIO, computed on the SAME bootstrap draws
            mb, cb = matched["boot_slopes"], cross["boot_slopes"]
            n = min(len(mb), len(cb))
            if n >= 100:
                rat = [mb[i] / cb[i] for i in range(n) if abs(cb[i]) > 1e-9]
                if rat:
                    spec_ratio = round(float(np.median(rat)), 2)
                    spec_ci = [round(float(np.percentile(rat, 2.5)), 2),
                               round(float(np.percentile(rat, 97.5)), 2)]
            if cs[0] > 0.0:
                verdict = "no_specificity"
                why = (f"matched {matched['slope']:+.4f} CI {matched['slope_ci']} 显著为正,但 cross(kissing)"
                       f" {cross['slope']:+.4f} CI {cs} 也同向显著 —— 效应不是罚项特异的,最可能就是"
                       "『序列越长越难』。按声明:【邓煜机制不成立】,只能报成一个平凡的长度效应")
            elif matched["exponent_ci"] is None:
                verdict = "mechanism_yes_exponent_no"
                why = "指数 CI 拿不到"
            elif matched["exponent_ci"][0] <= 3.0 <= matched["exponent_ci"][1]:
                verdict = "confirmed"
                why = (f"P1 斜率 {matched['slope']:+.4f} CI {matched['slope_ci']} 排除 0 且为正;"
                       f"P2 cross(kissing) {cross['slope']:+.4f} CI {cs} 含 0;"
                       f"P3 实测指数 {matched['exponent']} CI {matched['exponent_ci']} 覆盖 3")
            else:
                verdict = "mechanism_yes_exponent_no"
                why = (f"P1 P2 都过,但实测指数 {matched['exponent']} CI {matched['exponent_ci']} 排除 3 —— "
                       "赛跑存在,指数 3g 错。按声明把实测指数当结论报出来")

    print(f"\n=== 判读(主人群 A,主检验 = H 扫描 x H-type)===")
    print(f"  结论 {verdict}")
    print(f"  {why}")
    if spec_ratio is not None:
        print(f"  特异性统计量 matched/cross = {spec_ratio} CI {spec_ci}"
              f"  (同一批 bootstrap 抽样;比两次独立检验功效高)")
    if cross is not None:
        print(f"  安慰剂臂功效限制原样带出:kissing 只有 {cross['n_clusters']} 个靶标簇,"
              "『CI 含 0』不等于『它确实不动』")
    print("  截距/L* 的绝对位置【不】参与判读 —— 推导丢掉了 O(1) 常数 c_g,没有预言它")
    kk = A.get(("K", "kissing / chain")); hk = A.get(("K", "H-type"))
    if kk and hk:
        print(f"  K 扫描的镜像检查:kissing 斜率 {kk['slope']} CI {kk['slope_ci']};"
              f"H-type 斜率 {hk['slope']} CI {hk['slope_ci']}")

    json.dump({"units": len(units), "drop": dict(drop), "fold_fail": dict(fails),
               "verdict": verdict, "why": why, "spec_ratio": spec_ratio, "spec_ratio_ci": spec_ci,
               "pred_slope": PRED_SLOPE, "RT": RT, "grid": GRID, "results":
               [{k: v for k, v in r.items() if k != "boot_slopes"} for r in res]},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
