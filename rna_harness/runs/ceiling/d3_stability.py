#!/usr/bin/env python
"""D3 -- is the shipped verdict stable under the uncertainty pKiss admits it has?

Pre-registration: d3_stability_declaration.json (commit 60abe4d4).

D1's law is dead: the rank correlation between drop point and log length came out at -0.302, the wrong
sign, and that measurement was confounded by holding one penalty inside the other's swept range anyway
(d1_audit.md).  What survives is Deng's TECHNIQUE rather than his conclusion.  His proof does not evaluate
the individually uncontrollable cycle terms one at a time; it establishes whether their sum is stable.  The
transplant: stop reading the oracle's point prediction and read its sensitivity.  pKiss's own help text
concedes the pseudoknot energy parameters were never measured in a wet lab, so the uncertainty being
propagated here is the tool authors', not a device invented for the occasion.

Four things in the declaration have teeth and are enforced below:

  * A HARD IDENTITY GATE runs before anything else.  The whole stencil rests on my refold of the shipped
    point reproducing the shipped column, so that assumption is a gate, not a comment.  A15 is what happens
    when a judge is fed the wrong thing and the run completes, prints a verdict and writes JSON regardless.
  * The bar is F2 union f1c = 0.196, not the shipped 0.047 and not the panel's 0.103.  R2's gap column only
    revealed its true size (+0.0012) once it was finally held to that bar.
  * A REQUIRED CONTROL decides whether the column is new at all.  If the panel's own disagreement absorbs
    the increment, the verdict is `redundant` and may not be written up as a new column.
  * The void gate is a direct observation -- how many molecules change their crossing set anywhere in the
    stencil -- not a statistic standing in for one, which is the mistake A16 made.
"""
from __future__ import annotations

import argparse, collections, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)

from arbitration_confirm import load_fresh                                       # noqa: E402
from r1_info_ladder import ORACLES, FINE, r2_within, demean_by_cell             # noqa: E402

PKISS = "/home/ubuntu/miniconda3/envs/pk/bin/pKiss"
SHIP = (9.0, 12.0)
NEIGH = [(8.0, 12.0), (10.0, 12.0), (9.0, 11.0), (9.0, 13.0)]
STENCIL = [SHIP] + NEIGH
FLOOR_CELLS, NBOOT, NPERM_POINT, NPERM_BOOT = 20, 2000, 200, 8
L2_FLOOR = 30                                # declared: k>=1 & unstable subgroup below this -> no data
POPS = [("第四批", "panel_f1_fourth.json"), ("样本3", "panel_f1_third.json")]
KNOWN_BAR = {"第四批": 0.196, "样本3": 0.154}


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
    o = (r.stdout or "").strip().split()
    if r.returncode != 0 or not o or len(o[-1]) != len(seq):
        return (seq, h, k, None, f"rc={r.returncode}")
    return (seq, h, k, o[-1], None)


def pk_cols(struct, cross):
    """pKiss's columns under panel_f1_designs.py's definitions, verbatim."""
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    ps = pk_pairs(struct); pc = crossing_subset(ps)
    hit = len(pc & cross)
    rec = hit / len(cross); pre = (hit / len(pc)) if pc else 0.0
    return {"recall_product": len(ps & cross) / len(cross), "recall": rec, "precision": pre,
            "f1": (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else 0.0,
            "n_pred_cross": len(pc), "n_pred_total": len(ps), "_cross": pc}


def base_blocks(r):
    """F2 (21) + f1c (1) = the 22 columns of the bar the declaration froze."""
    po = r["per_oracle"]
    rp = [po[o]["recall_product"] for o in ORACLES]
    k = sum(1 for x in rp if x >= 1.0)
    f0 = [1.0 if k == j else 0.0 for j in (1, 2, 3)]
    fine = [float(po[o][c]) for o in ORACLES for c in FINE]
    return f0 + [float(x) for x in rp] + fine + [float(r["f1c"])], k


def disagree_cols(r):
    """The panel's OWN disagreement, 4 columns, from existing JSON -- the required control."""
    po = r["per_oracle"]
    rec = [po[o]["recall"] for o in ORACLES]
    npc = [float(po[o]["n_pred_cross"]) for o in ORACLES]
    f1 = [po[o]["f1"] for o in ORACLES]
    ind = [1 if po[o]["recall_product"] >= 1.0 else 0 for o in ORACLES]
    maj = 1 if sum(ind) >= 2 else 0
    return [float(np.std(rec)), float(np.std(npc)), float(max(f1) - min(f1)),
            float(sum(1 for x in ind if x != maj))]


def net_r2(y, X, nperm, seed, ci):
    obs = r2_within(y, X)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(nperm):
        yp = y.copy()
        for c in range(ci.max() + 1):
            m = ci == c
            yp[m] = rng.permutation(yp[m])
        vals.append(r2_within(yp, X))
    return obs - float(np.mean(vals)), obs


def run_pop(name, path, T, fold, out):
    from rna_harness.benchmarks.rnainvbench import pk_pairs, _round_only
    rec, _s, _m = load_fresh(os.path.join(HERE, path), "panel_recall_product")
    keep, idgate = [], []
    for r in rec:
        po = r.get("per_oracle") or {}
        if sorted(po) != sorted(ORACLES) or not r.get("seq") or r.get("f1c") is None:
            continue
        if any(po[o] is None or po[o].get("recall_product") is None for o in ORACLES):
            continue
        t = T.get(r["puzzle"])
        if not t:
            continue
        cross = pk_pairs(t["target"]) - pk_pairs(_round_only(t["target"]))
        if not cross:
            continue
        sts = {p: fold.get(f"{r['seq']}|{p[0]}|{p[1]}") for p in STENCIL}
        if any(v is None for v in sts.values()):
            continue
        cols = {p: pk_cols(sts[p], cross) for p in STENCIL}
        idgate.append(abs(cols[SHIP]["recall_product"] - po["pkiss"]["recall_product"]))
        r["_st"] = cols
        keep.append(r)
    print(f"\n=== {name} ===\n  模板全齐的分子 {len(keep)} / {len(rec)}")

    # ---------------- HARD IDENTITY GATE, before anything else
    bad = sum(1 for d in idgate if d > 1e-9)
    print(f"  【硬身份闸门】出厂点重折 vs 面板已存 pkiss.recall_product:比对 {len(idgate)} 条,"
          f"不一致 {bad} 条  {'✓' if bad == 0 else '✗'}")
    if bad:
        worst = max(idgate)
        print(f"  最大偏差 {worst:.6g} —— 按声明,整个 D3 不可用,不报任何结果")
        return {"pop": name, "identity_fail": bad, "max_dev": worst}

    # ---------------- VOID GATE: a direct observation
    for r in keep:
        c = r["_st"]; b = c[SHIP]["_cross"]
        r["_nflip"] = sum(1 for p in NEIGH if c[p]["_cross"] != b)
        r["_unstable"] = 1.0 if r["_nflip"] else 0.0
        recs = [c[p]["recall"] for p in STENCIL]
        r["_recrange"] = max(recs) - min(recs)
        k0 = 1 if c[SHIP]["recall_product"] >= 1.0 else 0
        r["_kflip"] = 1.0 if any((1 if c[p]["recall_product"] >= 1.0 else 0) != k0 for p in NEIGH) else 0.0
    dist = collections.Counter(r["_nflip"] for r in keep)
    nun = sum(1 for r in keep if r["_unstable"])
    print("\n  【作废闸门 = 直接观察】模板内交叉集改变次数分布:")
    for k in sorted(dist):
        print(f"    变了 {k}/4 次:{dist[k]:5d} 条 ({dist[k]/len(keep):6.1%})")
    print(f"    >>> 不稳定 {nun}/{len(keep)} = {nun/len(keep):.1%};"
          f"k_flip {sum(1 for r in keep if r['_kflip'])}/{len(keep)};"
          f"召回极差中位 {np.median([r['_recrange'] for r in keep]):.3f}")
    if nun == 0:
        print("  没有任何分子改变 -> 什么都没测,D3 在这个人群上作废")
        return {"pop": name, "void": True}

    # ---------------- feature sets
    C = {}
    for r in keep:
        b, k = base_blocks(r)
        C[id(r)] = {"base": b, "stab": [r["_unstable"], float(r["_nflip"]), r["_recrange"], r["_kflip"]],
                    "dis": disagree_cols(r), "k": k}
    cells = collections.defaultdict(list)
    for r in keep:
        cells[(r["puzzle"], r["method"])].append(r)
    rows = [(c, g) for c, g in cells.items() if len(g) >= 3]
    print(f"\n  格 {len(rows)}(每格 >=3 条)")
    if len(rows) < FLOOR_CELLS:
        print(f"  低于声明下限 {FLOOR_CELLS} -> 无数据"); return {"pop": name, "no_data": True}

    SETS = {"bar": ["base"], "bar+stab": ["base", "stab"], "bar+dis": ["base", "dis"],
            "bar+dis+stab": ["base", "dis", "stab"]}
    def build(parts):
        return {id(r): sum((C[id(r)][p] for p in parts), []) for r in keep}
    y, _X, ci = demean_by_cell(rows, build(["base"]))
    XS = {nm: demean_by_cell(rows, build(parts))[1] for nm, parts in SETS.items()}
    nmax = max(X.shape[1] for X in XS.values())
    dof = len(y) - nmax - len(rows)
    print(f"  参与拟合 {len(y)};最大特征数 {nmax};自由度 {len(y)} − {nmax} − {len(rows)} = {dof}")
    if dof <= 0:
        return {"pop": name, "no_data": True, "dof": dof}

    pt = {}
    for nm, X in XS.items():
        nr, obs = net_r2(y, X, NPERM_POINT, 12345, ci)
        pt[nm] = {"net": nr, "obs": obs, "ncol": X.shape[1]}
    print(f"\n  点估计(置换 {NPERM_POINT} 次,所有特征集共用同一批置换):"
          f"  【杆 = F2∪f1c,已知 {KNOWN_BAR.get(name)}】")
    print(f"    {'特征集':16}{'列数':>5}{'样本内 R^2':>12}{'净 R^2':>10}{'Δ净 vs 杆':>12}")
    for nm in SETS:
        d = pt[nm]["net"] - pt["bar"]["net"]
        ds = "" if nm == "bar" else f"{d:+.4f}"
        print(f"    {nm:16}{pt[nm]['ncol']:>5}{pt[nm]['obs']:12.4f}{pt[nm]['net']:10.4f}{ds:>12}")
    viol = [(a, b) for a, b in (("bar", "bar+stab"), ("bar", "bar+dis"), ("bar+dis", "bar+dis+stab"))
            if pt[b]["obs"] < pt[a]["obs"] - 1e-12]
    print(f"  自检(嵌套 ⇒ 样本内 R^2 单调不减,精确代数):违反 {len(viol)} 处"
          + ("  ✓" if not viol else f"  ✗ {viol} —— 估计量坏了,不报结果"))
    if viol:
        return {"pop": name, "estimator_broken": str(viol)}

    # ---------------- cluster bootstrap, common random numbers
    print(f"\n  cluster bootstrap {NBOOT} 次(按格重抽):")
    B = collections.defaultdict(list)
    rng = np.random.default_rng(7)
    for _ in range(NBOOT):
        sub = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        yb, _z, cib = demean_by_cell(sub, build(["base"]))
        if len(yb) < nmax + len(sub) + 1:
            continue
        sd = int(rng.integers(0, 2 ** 31 - 1))
        v = {}
        for nm, parts in SETS.items():
            Xb = demean_by_cell(sub, build(parts))[1]
            v[nm] = net_r2(yb, Xb, NPERM_BOOT, sd, cib)[0]
        B["stab_over_bar"].append(v["bar+stab"] - v["bar"])
        B["stab_over_bar_dis"].append(v["bar+dis+stab"] - v["bar+dis"])
        B["dis_over_bar"].append(v["bar+dis"] - v["bar"])
        B["dis_over_bar_stab"].append(v["bar+dis+stab"] - v["bar+stab"])
    res = {}
    print(f"    {'增量':22}{'中位':>10}{'95% CI':>24}{'排除 0':>8}")
    for k, vv in B.items():
        a = np.array(vv)
        if len(a) < 100:
            continue
        lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
        res[k] = {"median": float(np.median(a)), "ci": [lo, hi], "excl": bool(lo > 0 or hi < 0), "n": len(a)}
        print(f"    {k:22}{res[k]['median']:+10.4f}   [{lo:+.4f},{hi:+.4f}]{'是' if res[k]['excl'] else '否':>8}")

    # ---------------- level 2: the product question
    conf = [r for r in keep if C[id(r)]["k"] >= 1]
    cl2 = collections.defaultdict(list)
    for r in conf:
        cl2[(r["puzzle"], r["method"])].append(r)
    use = [(c, g) for c, g in cl2.items() if len({r["_unstable"] for r in g}) == 2]
    n_un = sum(1 for r in conf if r["_unstable"])
    print(f"\n  === level 2(产品问题)===\n  k>=1 的分子 {len(conf)};其中不稳定 {n_un};"
          f"两种都有的格 {len(use)}")
    l2 = None
    if n_un < L2_FLOOR or len(use) < 5:
        print(f"  不稳定子群 {n_un} < 声明下限 {L2_FLOOR} 或可比格 < 5 -> level 2 无数据,不许用它讲故事")
    else:
        d = []
        for _c, g in use:
            s = [r["wet"] for r in g if not r["_unstable"]]; u = [r["wet"] for r in g if r["_unstable"]]
            d.append(float(np.mean(s) - np.mean(u)))
        d = np.array(d)
        bs = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(NBOOT)]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        l2 = {"n_cells": len(d), "mean": float(d.mean()), "ci": [lo, hi], "excl": bool(lo > 0 or hi < 0),
              "n_unstable": n_un}
        print(f"  wet(稳定) − wet(不稳定),格内对比:格 {len(d)}  均值 {d.mean():+.3f}"
              f"  CI [{lo:+.3f},{hi:+.3f}]  {'排除 0' if l2['excl'] else '跨过 0'}")
    out.append({"pop": name, "n": len(keep), "n_cells": len(rows), "identity_ok": True,
                "nflip_dist": {str(k): v for k, v in dist.items()}, "unstable_rate": nun / len(keep),
                "point": pt, "boot": res, "level2": l2})
    return out[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache", default=os.path.join(HERE, "d3_fold.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "d3_stability.json"))
    a = ap.parse_args()
    from leaderboard_design import targets
    T = targets()
    seqs = set()
    for _n, p in POPS:
        for r in json.load(open(os.path.join(HERE, p)))["designs"]:
            if r.get("seq") and r["puzzle"] in T:
                seqs.add(r["seq"])
    print(f"唯一序列 {len(seqs)};模板 {STENCIL}")
    fold = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
    jobs = sorted({(s, h, k) for s in seqs for (h, k) in STENCIL if f"{s}|{h}|{k}" not in fold})
    print(f"  待折叠 {len(jobs)},{a.workers} 进程")
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
        run_pop(nm, p, T, fold, out)

    pri = out[0] if out else None
    print("\n=== 判读(按声明:主人群 = 第四批,杆 = F2∪f1c)===")
    v, why = "no_data", "主人群没有可用结果"
    if pri and pri.get("identity_ok"):
        so = pri["boot"].get("stab_over_bar"); sod = pri["boot"].get("stab_over_bar_dis")
        if so is None or sod is None:
            v, why = "no_data", "bootstrap 有效复本不足"
        elif not so["excl"] or so["median"] <= 0:
            v, why = "null", (f"稳定性在杆之上的增量 {so['median']:+.4f} CI {so['ci']} 未排除 0 "
                              "-> 这一列在 F2∪f1c 之上不含信息")
        elif not sod["excl"] or sod["median"] <= 0:
            v = "redundant"
            why = (f"稳定性在杆之上加 {so['median']:+.4f} CI {so['ci']},但在【加入面板自己的分歧之后】"
                   f"只剩 {sod['median']:+.4f} CI {sod['ci']} -> 它只是『判官们意见不一致』的换皮,"
                   "按声明【不许】写成新列")
        else:
            v = "new_column"
            why = (f"稳定性在杆之上加 {so['median']:+.4f} CI {so['ci']},且在加入分歧对照之后仍加 "
                   f"{sod['median']:+.4f} CI {sod['ci']} -> 正交的新列")
        l2 = pri.get("level2")
        if l2 and l2["excl"] and l2["mean"] > 0:
            why += (f";【level 2 成立】k>=1 里稳定的比不稳定的高 {l2['mean']:+.3f} CI {l2['ci']}"
                    f"(格 {l2['n_cells']},不稳定 {l2['n_unstable']} 条) -> 一条可接进出厂的弃权规则")
        elif l2:
            why += f";level 2 跨过 0({l2['mean']:+.3f} CI {l2['ci']}) -> 弃权规则没有依据"
        else:
            why += ";level 2 无数据"
    elif pri:
        v, why = "identity_fail", f"出厂点重折与面板已存列不一致 {pri.get('identity_fail')} 条"
    print(f"  结论 {v}\n  {why}")
    print("  自设天花板原样带出:不稳定率约一成,池化增量有界,本实验不会把 0.196 变成 0.4;")
    print("  它的价值在【正交性】—— 面板 21 列全是同一参数点上的点预测,RibonanzaNet 那列没有罚项。")
    json.dump({"verdict": v, "why": why, "stencil": STENCIL, "pops": out},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
