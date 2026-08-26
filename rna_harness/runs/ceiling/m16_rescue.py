#!/usr/bin/env python3
"""M16: does a base pair's TYPE predict whether it actually exists?

Declaration: m16_rescue_declaration.json, frozen and pushed at b4785451 BEFORE this file existed
(the seventeenth). Blind on the main arm -- only the pooled outcome distribution was inspected, for
the power calculation.

rescue_factor is the fraction of SHAPE-profile disruption a COMPENSATORY double mutation restores.
Swap both ends of a pair: if the pair is really there, the structure comes back. It is not the
OpenKnot score, so the score's thresholds and base-composition sensitivity cannot reach it, and it
interrogates one specific pair, so structural degeneracy cannot rescue. First readout on this line
immune to both -- which is what F8 needs.

Matched inside each design on helix depth and crossed status, because terminal pairs are looser and
designers put G-C inside helices. K5 names the outcome where matching kills it.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
OPEN, CLOSE = "([{<", ")]}>"
NPERM, R_NOISE, DRAWS, P_GATE, PASS = 2000, 12, 20, 0.005, 0.8
DELTAS = (0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60)
SELF_MAX, MIN_N = 0.15, 100


def parse_pairs(db):
    st = {c: [] for c in OPEN}; out = {}
    for i, c in enumerate(db):
        if c in OPEN:
            st[c].append(i)
        elif c in CLOSE:
            s = st[OPEN[CLOSE.index(c)]]
            if s:
                j = s.pop(); out[j] = i; out[i] = j
    return out


def load():
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(SRC) as f:
        rd = csv.DictReader(f)
        rc = sorted(k for k in rd.fieldnames
                    if k.startswith("reactivity_") and not k.startswith("reactivity_error"))
        for d in rd:
            rows.append({"puzzle": d["puzzle"], "method": d["method"], "seq": d["sequence"],
                         "db": d["target_structure"], "mutA": d["mutA"], "mutB": d["mutB"],
                         "rf": d["rescue_factor"], "sn": d["SN_filter"],
                         "ss": d["sub_start"], "react": [None if d[k] in ("", "NA", "nan")
                                                         else float(d[k]) for k in rc]})
    return rows


def rmsd(a, b):
    v = [(x - y) ** 2 for x, y in zip(a, b) if x is not None and y is not None]
    return math.sqrt(sum(v) / len(v)) if v else None


def build():
    rows = load()
    wt, singles = {}, collections.defaultdict(dict)
    for d in rows:
        k = (d["puzzle"], d["method"])
        if d["mutA"] in ("", "NA", "nan") and d["mutB"] in ("", "NA", "nan"):
            wt[k] = d
        elif d["mutB"] in ("", "NA", "nan") and d["mutA"] not in ("", "NA", "nan"):
            singles[k][int(float(d["mutA"]))] = d
    recs = []
    for d in rows:
        if d["rf"] in ("", "NA", "nan") or d["sn"] not in ("1", "1.0"):
            continue
        k = (d["puzzle"], d["method"]); p = wt.get(k)
        if p is None:
            continue
        ss = int(float(d["ss"])); P = parse_pairs(d["db"])
        try:
            ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
        except ValueError:
            continue
        a, b = ma + ss - 2, mb + ss - 2
        if P.get(a) != b:
            continue
        s = p["seq"]; pt = "".join(sorted({s[a], s[b]}))
        dep = 0
        while P.get(a - dep - 1) == b + dep + 1:
            dep += 1
        up = 0
        while P.get(a + up + 1) == b - up - 1:
            up += 1
        crossed = any(m < a < n < b or a < m < b < n for m, n in P.items() if m < n)
        sa, sb = singles[k].get(ma), singles[k].get(mb)
        r1 = rmsd(p["react"], sa["react"]) if sa else None
        r2 = rmsd(p["react"], sb["react"]) if sb else None
        recs.append({"puzzle": d["puzzle"], "method": d["method"], "pt": pt,
                     "rf": float(d["rf"]), "depth": min(dep, up), "crossed": bool(crossed),
                     "design": k, "single_rmsd": (r1, r2)})
    return recs


def stratum(r):
    return (r["design"], min(r["depth"], 3), r["crossed"])


def matched_diff(recs, A, B):
    """within-stratum difference of means, stratum-size weighted; strata needing both types."""
    by = collections.defaultdict(lambda: {A: [], B: []})
    for r in recs:
        if r["pt"] in (A, B):
            by[stratum(r)][r["pt"]].append(r["rf"])
    num = den = 0.0; used = []
    for st, d in by.items():
        if not d[A] or not d[B]:
            continue
        w = min(len(d[A]), len(d[B]))
        num += w * (float(np.mean(d[A])) - float(np.mean(d[B]))); den += w
        used.append((st, w))
    return (num / den if den else None), den, used


def perm_diff(recs, A, B, rng):
    sub = [r for r in recs if r["pt"] in (A, B)]
    by = collections.defaultdict(list)
    for r in sub:
        by[stratum(r)].append(r)
    out = []
    for st, rs in by.items():
        lab = [r["pt"] for r in rs]
        rng.shuffle(lab)
        for r, L in zip(rs, lab):
            out.append({**r, "pt": L})
    return matched_diff(out, A, B)[0]


def cluster_ci(recs, A, B, seed, nb=NPERM):
    pz = sorted({r["puzzle"] for r in recs})
    byp = collections.defaultdict(list)
    for r in recs:
        byp[r["puzzle"]].append(r)
    rng = np.random.default_rng(seed); out = []
    for _ in range(nb):
        idx = rng.integers(0, len(pz), len(pz))
        samp = [r for q in idx for r in byp[pz[q]]]
        d, den, _u = matched_diff(samp, A, B)
        if d is not None and den > 0:
            out.append(d)
    out = np.sort(np.array(out))
    return [round(float(out[int(.025 * len(out))]), 4), round(float(out[int(.975 * len(out))]), 4)]


def ladder(recs, A, B, seed):
    """rule 79: built on the NULL. Scramble labels first, then inject delta into A."""
    import random as _r
    rng0 = _r.Random(seed)
    null = np.sort(np.array([perm_diff(recs, A, B, _r.Random(seed + i)) for i in range(400)]))
    thr = float(np.quantile(null, .995))
    out = {}
    for d in DELTAS:
        hit = tot = 0
        for t in range(DRAWS):
            rr = _r.Random(int(d * 1e5) + t + seed)
            sub = [r for r in recs if r["pt"] in (A, B)]
            by = collections.defaultdict(list)
            for r in sub:
                by[stratum(r)].append(r)
            sc = []
            for st, rs in by.items():
                lab = [r["pt"] for r in rs]; rr.shuffle(lab)
                for r, L in zip(rs, lab):
                    sc.append({**r, "pt": L, "rf": r["rf"] + (d if L == A else 0.0)})
            v, den, _u = matched_diff(sc, A, B)
            if v is not None:
                hit += (v > thr); tot += 1
        out[d] = round(hit / tot, 3) if tot else None
    return out, thr, float(null.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m16_rescue.json"))
    a = ap.parse_args()
    print("=== M16:碱基对的【种类】能不能预测它是不是真的存在 ===")
    print("  声明 b4785451(代码存在之前已推送)。**主臂对我是盲的。** 离线 $0。")
    print("  读数 = rescue_factor(补偿双突变恢复的剖面 RMSD 比例)。")
    print("  **它不是 OpenKnot Score ⇒ 避开探测化学;它直查那一个对 ⇒ 避开结构简并。**\n")
    import random as _r
    recs = build()
    cnt = collections.Counter(r["pt"] for r in recs)
    print(f"可用碱基对 {len(recs)}  类型: " + "  ".join(f"{k}={v}" for k, v in cnt.most_common(6)))
    print(f"设计数 {len({r['design'] for r in recs})}  puzzle {len({r['puzzle'] for r in recs})}\n")

    OUT = {"counts": dict(cnt)}
    for A, B in (("CG", "AU"), ("CG", "GU"), ("AU", "GU")):
        if cnt[A] < 30 or cnt[B] < 30:
            continue
        print(f"=== {A} vs {B} ===")
        mA = float(np.mean([r["rf"] for r in recs if r["pt"] == A]))
        mB = float(np.mean([r["rf"] for r in recs if r["pt"] == B]))
        print(f"  边际(未配平):{A} {mA:+.4f}   {B} {mB:+.4f}   差 {mA - mB:+.4f}")
        d, den, used = matched_diff(recs, A, B)
        if d is None:
            print("  配平后无可用分层\n"); continue
        rng = _r.Random(a.seed)
        null = np.sort(np.array([perm_diff(recs, A, B, _r.Random(a.seed + i))
                                 for i in range(NPERM)]))
        p = (float((null >= d).sum()) + 1) / (NPERM + 1)
        ci = cluster_ci(recs, A, B, a.seed, nb=500)
        print(f"  **配平后差 {d:+.4f}**  置换 p {p:.5f}  零SD {null.std():.4f}  "
              f"分层权重和 {den:.0f}  可用分层 {len(used)}")
        print(f"  puzzle cluster bootstrap {ci}  半宽 {(ci[1]-ci[0])/2:.4f}")
        ent = {"marginal_A": round(mA, 4), "marginal_B": round(mB, 4),
               "marginal_diff": round(mA - mB, 4), "matched_diff": round(d, 4),
               "p_perm": round(p, 5), "null_sd": round(float(null.std()), 4),
               "boot_ci": ci, "weight": den, "n_strata": len(used)}
        # shape_4: crossed vs nested separately
        for cx in (True, False):
            sub = [r for r in recs if r["crossed"] == cx]
            dv, dn, _u = matched_diff(sub, A, B)
            ent[f"crossed_{cx}"] = {"diff": round(dv, 4) if dv is not None else None, "w": dn}
            print(f"  {'被交叉' if cx else '嵌套  '} 配平后差 "
                  f"{('%+.4f' % dv) if dv is not None else '—':>8s}  权重 {dn:.0f}")
        # shape_3: leave-one-puzzle-out
        pz = sorted({r["puzzle"] for r in recs})
        loo = [matched_diff([r for r in recs if r["puzzle"] != q], A, B)[0] for q in pz]
        loo = [x for x in loo if x is not None]
        ent["loo_range"] = [round(min(loo), 4), round(max(loo), 4)] if loo else None
        print(f"  留一 puzzle 极差 {ent['loo_range']}")
        # probes
        pos = 0
        for k in range(R_NOISE):
            v = perm_diff(recs, A, B, _r.Random(7000 + a.seed + k))
            pos += ((float((null >= v).sum()) + 1) / (NPERM + 1)) < P_GATE
        ent["P1_pos"] = pos
        lad, thr, nsd = ladder(recs, A, B, a.seed)
        ent["ladder"] = {str(k): v for k, v in lad.items()}
        ent["ladder_delta0"] = lad.get(0.00)
        det = next((x for x in DELTAS if x > 0 and (lad.get(x) or 0) >= PASS), None)
        ent["ladder_threshold"] = det
        d0 = lad.get(0.00)
        # NOT `(lad.get(0.00) or 1)`: 0.0 is falsy, so a PASSING self-check of 0.0 printed as a
        # failure. The verdict logic never used that expression, so only the log line was wrong.
        d0_ok = (d0 is not None and d0 <= SELF_MAX)
        print(f"  P1 随机标签 {pos}/{R_NOISE}(须<=1)   "
              f"P2 梯级 δ=0 自检 {d0} {'通过' if d0_ok else '**不通过**'}")
        print(f"     " + "  ".join(f"{x:.3f}:{lad[x]}" for x in DELTAS if x > 0))
        print(f"     检出阈值 {det}  实测差 {d:+.4f}  "
              f"{'有功率' if (det is not None and abs(d) >= det) else '**功率不足**'}")
        OUT[f"{A}_vs_{B}"] = ent
        print()

    # diagnostic: single-mutant RMSD by pair type (declared, not gated)
    print("--- 诊断(声明要求报出,不进闸门):单突变 RMSD 按对类型 ---")
    dg = {}
    for t in ("CG", "AU", "GU"):
        v = [x for r in recs if r["pt"] == t for x in r["single_rmsd"] if x is not None]
        if v:
            dg[t] = round(float(np.mean(v)), 4)
            print(f"  {t}: 单突变 RMSD 均值 {dg[t]:.4f}  (n={len(v)})")
    OUT["single_mutant_rmsd"] = dg
    if "CG" in dg and "AU" in dg:
        print(f"  ⇒ G-C 的分母比 A-U {'更大 ⇒ 本臂【低估】真实差(保守)' if dg['CG'] > dg['AU'] else '更小 ⇒ 本臂可能【高估】,必须写出来'}")

    # verdict
    e = OUT.get("CG_vs_AU", {})
    k5 = bool(e and e["marginal_diff"] > 0 and abs(e["matched_diff"]) < 0.2 * abs(e["marginal_diff"]))
    sc_ok = all(v.get("ladder_delta0", 1) <= SELF_MAX for k, v in OUT.items()
                if isinstance(v, dict) and "ladder_delta0" in v)
    p1_ok = all(v.get("P1_pos", 9) <= 1 for k, v in OUT.items()
                if isinstance(v, dict) and "P1_pos" in v)
    det = e.get("ladder_threshold")
    if not (sc_ok and p1_ok):
        vd = "void_probe_failed"
    elif not e:
        vd = "no_data"
    elif det is None or abs(e["matched_diff"]) < det:
        vd = "no_adjudicating_power"
    elif k5:
        vd = "position_not_type"
    elif e["matched_diff"] > 0 and e["p_perm"] < P_GATE:
        vd = "pair_type_predicts_existence"
    elif e["matched_diff"] <= 0:
        vd = "no_pair_type_effect"
    else:
        vd = "under_replicated"
    print(f"\n裁定: {vd}")
    OUT["verdict"] = vd
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
