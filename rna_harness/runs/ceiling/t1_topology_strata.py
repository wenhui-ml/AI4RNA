#!/usr/bin/env python3
"""T1: split the POPULATION by topology, not the energy -- Deng move two, used the way it was meant.

Pre-registered in t1_topology_strata_declaration.json (rule 38/39/40 spec).

THIS IS A MEASUREMENT, NOT A DESIGN. It answers whether the topology classes behave differently enough to
justify treating them separately. It designs no decision procedure -- that is design work, and it must not
start before this returns, because if the classes behave alike there is nothing to split.

WHY THE SPLIT IS ON THE POPULATION. Today I used move two on the energy (gamma* = skeleton + stacks +
unknown) and it failed. Deng splits the INDEX SET: tree diagrams compute exactly, cycles only get bounded.
Transposed here, our index set is the target's topology class, not one molecule's energy terms.

NOT A FISHING TRIP. certify.py's TOPOLOGY_OPERATING_POINTS already records external agreement of 0.972
[0.855, 0.999] for H-type against 0.800 and 0.608 for the others, and H-type's row already says "the
calibrated rule is NOT used here" -- the product already treats it specially. What is unmeasured is how the
criterion's INFORMATION is distributed across classes.

DECLARED LIMITATION. H-type has only 9 targets, so a cluster bootstrap over targets on 9 clusters would be
unreliable. The unit is therefore the CELL (target x method), as in R1, and a class below the 20-cell floor
is reported as no-data rather than estimated. H-type's interval will be the widest of the three and any
"H-type is better" reading must carry that width.

COST: $0, offline, minutes. Columns already on disk; nothing is folded.
"""
from __future__ import annotations

import argparse, collections, json, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from arbitration_confirm import load_fresh                                       # noqa: E402
from r1_info_ladder import r2_within, demean_by_cell                             # noqa: E402
from e1_ensemble_marginals import base_block, load_pop                           # noqa: E402

TUBE_PASS = 80.0
FLOOR_CELLS = 20
KNOWN = ("H-type", "kissing / chain", "higher-order")


class Undefined(ValueError):
    """classify_topology returned a class this run does not know how to place (rule 38, second sentence)."""


def topo_of(target):
    from rna_harness.structure_utils import classify_topology
    t = classify_topology(target)["topology"]
    if t not in KNOWN:
        raise Undefined(f"unknown topology {t!r}; refusing to fold it into 'other'")
    return t


def net_r2(rows, cols_of, nperm, seed, draws):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    if len(rws) < FLOOR_CELLS:
        return {"no_data": True, "n_cells": len(rws), "n_mol": len(rows)}
    y, X, ci = demean_by_cell(rws, cols_of)
    NT = X.shape[1]
    dof = len(y) - NT - len(rws)
    if dof <= 0:
        return {"no_data": True, "dof": dof, "n_cells": len(rws)}
    obs = r2_within(y, X)
    rng = np.random.default_rng(seed)
    vv = []
    for _ in range(nperm):
        yp = y.copy()
        for c in range(ci.max() + 1):
            m = ci == c
            yp[m] = rng.permutation(yp[m])
        vv.append(r2_within(yp, X))
    nul = float(np.mean(vv))
    prng = random.Random(seed)
    bs = []
    for _ in range(draws):
        sel = [rws[prng.randrange(len(rws))] for _ in range(len(rws))]
        yy, XX, _ = demean_by_cell(sel, cols_of)
        v = r2_within(yy, XX)
        if v is not None:
            bs.append(v - nul)
    bs.sort()
    return {"n_cells": len(rws), "n_mol": int(len(y)), "dof": int(dof), "r2": obs, "null": nul,
            "net": obs - nul, "ci": [bs[int(0.025 * len(bs))], bs[min(len(bs) - 1, int(0.975 * len(bs)))]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "t1_topology_strata.json"))
    a = ap.parse_args()
    from leaderboard_design import targets
    T = targets()
    POPS = (("基准4560(主)", "panel_f1_designs.json"), ("样本3", "panel_f1_third.json"),
            ("第四批", "panel_f1_fourth.json"), ("样本2", "panel_f1_fresh.json"))
    res = {}
    for nm, fn in POPS:
        print(f"\n=== {nm} ===", flush=True)
        rec, _s, _m = load_pop(fn)
        cols, by, undef = {}, collections.defaultdict(list), 0
        for r in rec:
            B = base_block(r)
            if B is None:
                continue
            try:
                t = topo_of(T[r["puzzle"]]["target"])
            except Undefined:
                undef += 1
                continue
            cols[id(r)] = B
            by[t].append(r)
        print(f"  域外(未知拓扑,已抛出不归入 other):{undef}")
        pop = {"undefined_topology": undef}
        allrows = [r for g in by.values() for r in g]
        whole = net_r2(allrows, cols, a.nperm, a.seed, a.draws)
        print(f"  {'类别':16}{'靶标':>5}{'设计':>6}{'格':>5}{'净R2':>9}   95%CI"
              f"{'试管失败':>10}{'k=3占比':>9}{'k=3失败':>9}{'k 单调':>8}")
        pop["ALL"] = whole
        print(f"  {'(全体)':16}{len({r['puzzle'] for r in allrows}):5d}{len(allrows):6d}"
              f"{whole.get('n_cells', 0):5d}{whole.get('net', float('nan')):9.4f}"
              f"   [{whole['ci'][0]:+.4f},{whole['ci'][1]:+.4f}]" if not whole.get("no_data") else "  (全体)无数据")
        for t in KNOWN:
            g = by.get(t) or []
            if not g:
                print(f"  {t:16}  —— 本人群无此类"); continue
            d = net_r2(g, cols, a.nperm, a.seed, a.draws)
            fr = sum(1 for r in g if r["wet"] < TUBE_PASS) / len(g)
            ks = collections.defaultdict(list)
            for r in g:
                po = r["per_oracle"]
                k = sum(1 for o in ("pkiss", "probknot", "shapeknots")
                        if (po[o] or {}).get("recall_product") is not None and po[o]["recall_product"] >= 1.0)
                ks[k].append(r["wet"])
            f3 = (sum(1 for w in ks[3] if w < TUBE_PASS) / len(ks[3])) if ks[3] else None
            rates = {k: (sum(1 for w in v if w < TUBE_PASS) / len(v)) for k, v in ks.items() if v}
            mono = all(rates.get(k + 1, 1e9) <= rates.get(k, -1e9) for k in range(3) if k in rates and k + 1 in rates)
            d.update({"tube_fail": fr, "k3_frac": len(ks[3]) / len(g), "k3_fail": f3,
                      "k_rates": rates, "k_monotone": bool(mono),
                      "n_targets": len({r["puzzle"] for r in g})})
            pop[t] = d
            if d.get("no_data"):
                print(f"  {t:16}{d['n_targets']:5d}{len(g):6d}{d.get('n_cells',0):5d}"
                      f"   格数不足 {FLOOR_CELLS} -> 无数据(只报计数)")
                continue
            print(f"  {t:16}{d['n_targets']:5d}{len(g):6d}{d['n_cells']:5d}{d['net']:9.4f}"
                  f"   [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]{fr:10.3f}{d['k3_frac']:9.3f}"
                  f"{(f3 if f3 is not None else float('nan')):9.3f}{'成立' if mono else '破坏':>8}", flush=True)
        res[fn] = pop
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)

    print("\n=== 判定(按冻结的打死条件,主人群)===")
    P = res["panel_f1_designs.json"]
    live = {t: P[t] for t in KNOWN if t in P and not P[t].get("no_data")}
    if len(live) < 2:
        print("  K2 触发:可用类别少于 2 -> 无数据,B 这条线不进入设计阶段")
        v = "no_data"
    else:
        cis = {t: d["ci"] for t, d in live.items()}
        # K1: do all pairs of intervals overlap?
        def ov(a_, b_):
            return not (a_[1] < b_[0] or b_[1] < a_[0])
        ts = list(live)
        allov = all(ov(cis[x], cis[y]) for i, x in enumerate(ts) for y in ts[i + 1:])
        K1 = allov
        best = max(live, key=lambda t: live[t]["net"])
        worst_tube = max(live, key=lambda t: live[t]["tube_fail"])
        K3 = (best == worst_tube)
        print(f"  各类净R2: " + "  ".join(f"{t}={live[t]['net']:+.4f}{cis[t]}" for t in ts))
        print(f"  K1 所有区间彼此重叠(无分层依据) :{'FIRES' if K1 else '不触发'}")
        print(f"  K3 净R2最高的那类恰是试管最难的那类(高R2只是方差大):{'FIRES' if K3 else '不触发'}"
              f"   (净R2最高={best}, 试管最难={worst_tube})")
        v = "no_basis_for_stratifying" if K1 else "r2_is_a_difficulty_artefact" if K3 else "stratifying_is_justified"
        P["kills"] = {"K1": K1, "K3": K3}
        P["candidate_class_for_exact_decision"] = best
    print(f"  裁定:{v}")
    P["verdict"] = v
    P["note"] = ("本实验只判『是否值得分类别处理』。任何精确判定程序的设计都在本实验之后,"
                 "且必须另开声明。H 型只有 9 个靶标,其区间必然最宽,任何『H 型更好』的读法都要带区间宽度。")
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
