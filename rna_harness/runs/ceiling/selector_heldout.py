#!/usr/bin/env python3
"""Stage B of executing decision A: let RibonanzaNet choose, and let judges that were never in the loop score.

RUNS IN THE `rna` env (ViennaRNA + the oracle binaries). Stage A (`selector_heldout_score.py`, rnet_env)
wrote the RibonanzaNet score of every dumped candidate; this file uses those scores only to CHOOSE.

THE ONE THING THIS RUN EXISTS TO AVOID. Every previous comparison of selection rules on our own pools was
scored by RibonanzaNet -- including the 0.826 -> 0.955 that decision A is built on. That is fine while
RibonanzaNet is outside the loop. The moment it becomes the chooser, scoring its choices with itself is
selecting on the outcome, which this project has already measured: max-of-RNet inflates by about +3.67 in its
own synthetic null, and `pool_rerank`'s +0.250 headline was retracted for exactly this. So the arms here are
judged by three predictors that have never participated in any selection rule:

    IPknot      /data/wenhui/pk_judges/ipknot          integer programming, different codebase
    Knotty      /data/wenhui/pk_judges/Knotty          sparse CKY, different codebase
    RNAPKplex   envs/vpk/bin/RNAPKplex                 ViennaRNA's own pseudoknot heuristic, separate env

    Quantity per arm: crossed-pair F1 of the DELIVERED sequence against the target's crossing pairs, the
    same statistic the release publishes for RibonanzaNet, so the arms are on one scale.

    A judge that fails to run returns None and the target is dropped from that judge's pairing. It is never
    counted as zero -- "the tool broke" and "the design is wrong" are different facts.

WHAT ALREADY LIMITS WHAT THIS CAN SHOW, stated before the numbers exist:

  * The judges are computational, not a tube. They are independent of our selection, not of thermodynamics --
    IPknot, Knotty and RNAPKplex all rest on nearest-neighbour energy models, as do the three in-loop
    oracles. Agreement between them is consistency, not truth.
  * Our own designs have no wet-lab score and never will without a laboratory. The only tube-anchored
    estimate of what RibonanzaNet-argmax buys comes from OTHER people's molecules: +0.55 OpenKnot points on
    unsaturated pools (21/35, p = 0.022, below this project's own pre-registered 1.0-point bar) and -0.30 on
    saturated ones (`rerank_audit.json`). Our pools sit between: a median of 10 candidates tie at the top.
  * Both rulers saturate. The panel's correlation with quality decays as its own score rises
    (+0.279 / +0.158 / +0.040) and RibonanzaNet's does too (+0.387 / +0.118 / +0.084 by its own terciles,
    36.1% of measured designs sitting at exactly 1.000). Selection is argmax, and argmax reads precisely the
    part of a ruler that has no gradient left. That is the reason to expect a small effect here, and the
    reason `screen_first` -- which uses the SOLVER's unsaturated ordering -- is the arm to beat rather than
    `shipped`.

ARMS (fixed before running; the primary comparison is rnet_gated vs screen_first, because shipped is already
known to be the worst of the three and beating it proves nothing new):

    shipped        what the product returned: argmax |confirmed_by|, then the entropy tie-breaker
    screen_first   first candidate clearing the loose screen = the solver's own top-ranked survivor
    rnet_gated     the panel still decides eligibility; RibonanzaNet picks among the eligible.  <- decision A
    rnet_screen    the panel is used only as the loose screen; RibonanzaNet picks among survivors
    random         control, mean over draws inside the same eligible set
    oracle_rnet    max RNet over the whole pool -- an UPPER BOUND under a metric we then do not judge with,
                   printed only so the scale is visible

COST: $0. Judges run on the delivered sequences only (a few hundred), not on the pools.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

ARMS = ["shipped", "screen_first", "rnet_gated", "rnet_screen", "rnet_top10", "rnet_top20",
        "random"]


def crossing_subset(pairs):
    ps = sorted(pairs)
    out = set()
    for a, (i, j) in enumerate(ps):
        for k, l in ps[a + 1:]:
            if i < k < j < l or k < i < l < j:
                out.add((i, j))
                out.add((k, l))
    return out


def judge_one(item):
    """Three never-in-loop judges on one delivered sequence. None means the tool failed, never a rejection."""
    puz, seq, target = item
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rnet_reproduce import f1
    from ipknot_judge import ipknot_pairs
    from knotty_judge import knotty_pairs
    from external_judge import pkplex_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    out = {}
    for name, fn in (("ipknot", ipknot_pairs), ("knotty", knotty_pairs), ("pkplex", pkplex_pairs)):
        try:
            ps = fn(seq)
        except Exception:                                                      # noqa: BLE001
            ps = None
        out[name] = None if ps is None else f1(crossing_subset(ps), cross)
    return puz, seq, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "pool_dump"))
    ap.add_argument("--run", default=os.path.join(HERE, "leaderboard_replay.json"))
    ap.add_argument("--rnet", default=os.path.join(HERE, "selector_heldout_rnet.json"))
    ap.add_argument("--pool-precision", default=None,
                    help="pool_precision.json -- adds panel_f1 arms (the panel choosing on its own "
                         "precision-aware score instead of the confirm count)")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "selector_heldout.json"))
    a = ap.parse_args()

    import random
    from m2r_pairwise import wilcoxon
    from selector_replay import load_dump, pick, KNOT_VERDICTS, SCREEN
    from rna_harness.cognitive.certify import calibrated_rule_applies

    R = json.load(open(a.run))
    T = {p: v["target"] for p, v in R["targets"].items()}
    rows = {r["puzzle"]: r for r in R["rows"]}
    P = load_dump(a.dump, T)
    G = {p: not calibrated_rule_applies(t) for p, t in T.items()}
    F = json.load(open(a.rnet))
    f1c = F["f1c"]
    PF = {}
    if a.pool_precision:
        d = json.load(open(a.pool_precision))
        for r in d["candidates"]:
            PF.setdefault(r["puzzle"], {})[r["seq"]] = r["f1"]
        print(f"面板 F1 列已载入:{sum(len(v) for v in PF.values())} 条候选"
              f"(重裁决与落盘召回的一致性不符 {d['identity_mismatches']} 条)")
        if d["identity_mismatches"]:
            print("  -> 重裁决与落盘的召回对不上,不加 panel_f1 臂。")
            PF = {}
    if PF:
        for extra in ("panel_f1", "panel_f1_screen"):
            if extra not in ARMS:
                ARMS.insert(ARMS.index("random"), extra)
    print(f"落盘池 {len(P)}/{len(T)} 个靶标,候选 {sum(len(v) for v in P.values())};"
          f"RNet 打分 {F['n_scored']}/{F['n_candidates']},耗时 {F['minutes']:.1f} 分钟"
          f"({F['minutes']*60/max(1,F['n_scored']):.2f} 秒/条,{F['workers']} 进程)")

    # ---- integrity gate, unchanged from selector_replay ------------------------------------------------
    elig = [p for p in P if rows.get(p, {}).get("verdict") in KNOT_VERDICTS]
    bad = [p for p in elig
           if (pick(P[p], "shipped", gated=G[p]) or {}).get("seq") != rows[p]["sequence"]]
    print(f"\n=== 完整性闸门:重实现的 shipped 能否复现真实交付的序列 ===")
    print(f"  可比靶标 {len(elig)},不一致 {len(bad)} {bad[:6]}")
    if bad:
        print("  -> 重放没有复刻产品行为,不打印任何对比数字。")
        json.dump({"integrity_pass": False, "mismatched": bad}, open(a.out, "w"))
        return 2

    # ---- choose ---------------------------------------------------------------------------------------
    rng = random.Random(a.seed)
    chosen = collections.defaultdict(dict)
    for p in elig:
        rowsp, sc = P[p], f1c.get(p, {})
        conf = [] if G[p] else [r for r in rowsp if r["knot_confirmed"]]
        surv = [r for r in rowsp if r["mean_recall"] >= SCREEN]
        elig_set = conf if conf else surv                # what the product's gate leaves on the table
        for arm in ARMS:
            if arm == "rnet_gated":
                pool = [r for r in elig_set if r["seq"] in sc]
                chosen[arm][p] = max(pool, key=lambda r: sc[r["seq"]])["seq"] if pool else None
            elif arm == "rnet_screen":
                pool = [r for r in surv if r["seq"] in sc]
                chosen[arm][p] = max(pool, key=lambda r: sc[r["seq"]])["seq"] if pool else None
            elif arm in ("panel_f1", "panel_f1_screen"):
                # The panel choosing on ITS OWN precision-aware score. Worth a row because the recall-vs-F1
                # defect was the whole of the panel's apparent loss to RibonanzaNet at matched coverage, so
                # the panel may be a usable chooser after all -- but note this puts the auditor back in the
                # selection seat, which is the configuration decision A was taken to end. A win here is a
                # measurement, not a licence to ship it.
                pf = PF.get(p, {})
                base = surv if arm.endswith("screen") else elig_set
                pool = [r for r in base if r["seq"] in pf]
                chosen[arm][p] = max(pool, key=lambda r: pf[r["seq"]])["seq"] if pool else None
            elif arm.startswith("rnet_top"):
                # THE BOUNDED-COST VERSION, and the reason it is worth a row: RibonanzaNet costs
                # 1.5e-5 * n^2.39 seconds per candidate (measured: 0.158 s at 48 nt, 1.09 at 100, 7.41 at
                # 240), so scoring a whole eligible set costs a median of 72 s and up to 6.5 min per target
                # on top of a ladder that currently takes 180-284 s. Scoring only the first k survivors in
                # the SOLVER's order caps that at k * sec(n) -- 74 s at k=10 even on the worst 240 nt
                # target -- and the solver's order is exactly the signal `screen_first` already exploits.
                k = int(arm[len("rnet_top"):])
                pool = [r for r in surv[:k] if r["seq"] in sc]
                chosen[arm][p] = max(pool, key=lambda r: sc[r["seq"]])["seq"] if pool else None
            elif arm == "random":
                chosen[arm][p] = [(pick(rowsp, "random", rng, gated=G[p]) or {}).get("seq")
                                  for _ in range(a.draws)]
            else:
                chosen[arm][p] = (pick(rowsp, arm, gated=G[p]) or {}).get("seq")
    orac = {p: (max(f1c[p].values()) if f1c.get(p) else None) for p in elig}

    # ---- judge, on unique delivered sequences only ----------------------------------------------------
    need = set()
    for arm in ARMS:
        for p, v in chosen[arm].items():
            for s in (v if isinstance(v, list) else [v]):
                if s:
                    need.add((p, s))
    print(f"\n=== 三个从未进过选择环的判官给交付序列打分 ===")
    print(f"  需要评判的 (靶标,序列) 组合 {len(need)} 个,{a.workers} 进程", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    J = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, (puz, seq, out) in enumerate(
                ex.map(judge_one, [(p, s, T[p]) for p, s in sorted(need)], chunksize=2), 1):
            J[(puz, seq)] = out
            if n % 50 == 0 or n == len(need):
                print(f"    {n}/{len(need)}", flush=True)
    for name in ("ipknot", "knotty", "pkplex"):
        nf = sum(1 for v in J.values() if v[name] is None)
        print(f"  {name:9s} 失败 {nf}/{len(J)}")

    def score(arm, p, judge):
        v = chosen[arm][p]
        if isinstance(v, list):
            xs = [J[(p, s)][judge] for s in v if s and J[(p, s)][judge] is not None]
            return st.mean(xs) if xs else None
        if not v or J[(p, v)][judge] is None:
            return None
        return J[(p, v)][judge]

    # ---- compare --------------------------------------------------------------------------------------
    res = {"integrity_pass": True, "n_targets": len(elig), "rnet_minutes": F["minutes"],
           "rnet_seconds_per_candidate": F["minutes"] * 60 / max(1, F["n_scored"]), "judges": {}}
    JUDGES = ["ipknot", "knotty", "pkplex", "mean3"]
    for judge in JUDGES:
        def val(arm, p):
            if judge != "mean3":
                return score(arm, p, judge)
            xs = [score(arm, p, j) for j in ("ipknot", "knotty", "pkplex")]
            xs = [x for x in xs if x is not None]
            return st.mean(xs) if xs else None
        med = {}
        for arm in ARMS:
            xs = [val(arm, p) for p in elig]
            xs = [x for x in xs if x is not None]
            med[arm] = {"n": len(xs), "median": st.median(xs) if xs else None,
                        "mean": st.mean(xs) if xs else None}
        print(f"\n=== 判官 {judge} ===")
        for arm in ARMS:
            m = med[arm]
            print(f"  {arm:13s} n={m['n']:2d}  中位 {m['median']:.3f}  均值 {m['mean']:.3f}"
                  if m["median"] is not None else f"  {arm:13s} 无数据")
        cmps = {}
        for hi, lo in (("rnet_gated", "screen_first"), ("rnet_gated", "shipped"),
                       ("screen_first", "shipped"), ("rnet_screen", "screen_first"),
                       ("rnet_top10", "screen_first"), ("rnet_top20", "screen_first"),
                       ("rnet_top10", "rnet_gated"), ("panel_f1", "screen_first"),
                       ("panel_f1", "rnet_gated"), ("random", "shipped")):
            d = []
            for p in elig:
                x, y = val(hi, p), val(lo, p)
                if x is not None and y is not None:
                    d.append(x - y)
            if len(d) < 6:
                continue
            w, p2, nz = wilcoxon(d, "two-sided")
            better = sum(1 for v in d if v > 0)
            worse = sum(1 for v in d if v < 0)
            cmps[f"{hi}-{lo}"] = {"n": len(d), "median": st.median(d), "mean": st.mean(d),
                                  "better": better, "worse": worse, "tie": len(d) - better - worse,
                                  "p_two_sided": p2}
            star = "  <- 主比较" if (hi, lo) == ("rnet_gated", "screen_first") else ""
            print(f"    {hi:13s} − {lo:13s} 中位 {st.median(d):+.3f}  均值 {st.mean(d):+.3f}"
                  f"  好/差/平 {better}/{worse}/{len(d)-better-worse}"
                  + (f"  双侧 p = {p2:.4g}" if p2 is not None else "") + star)
        res["judges"][judge] = {"per_arm": med, "comparisons": cmps}

    ob = [v for v in orac.values() if v is not None]
    print(f"\n  参照(不作判据):池内 RNet 最大值中位 {st.median(ob):.3f} —— 这是在 RNet 自己的尺子上取最大,"
          f"本项目自己的置换零显示这种取法能凭空抬高约 +3.67,所以它只是刻度,不是目标。")
    res["oracle_rnet_median_NOT_A_TARGET"] = st.median(ob)
    res["chosen"] = {arm: {p: v for p, v in chosen[arm].items()} for arm in ARMS if arm != "random"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n写出 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
