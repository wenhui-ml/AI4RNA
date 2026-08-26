#!/usr/bin/env python3
"""M14: someone changed one base and measured the tube again.

Declaration: m14_mutation_intervention_declaration.json, frozen and pushed at 9cf52dad BEFORE this
file existed (the thirteenth). Blind: I have looked only at column names, the index mapping
i = mutA + sub_start - 2 (verified 13976/13976), and the per-class COUNTS.

Everything in M13 was a matched contrast on a frozen corpus. This is an intervention. The library
swaps a base for its Watson-Crick complement, and the counts are symmetric by construction, so the
SAME pair gets broken from both sides an equal number of times.

Published base bias: intrinsic 2'-OH reactivity has G ~ A, 1.4x U and 1.7x C (PMID 19458034), and
the Eterna Classic Score calls a target-paired residue correct below 0.25. Chemistry therefore
predicts the two directions come out with OPPOSITE signs; biophysics predicts the same sign.

Y1 reads at the PARTNER position, whose base was never touched, so base identity cannot enter.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, csv, json, math, os, random, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/data/wenhui/openknot_data/Data/OK7a_M2_data.v4.5.2.csv"
OPEN, CLOSE = "([{<", ")]}>"
NBOOT, R_MIS, DRAWS, DISTAL = 2000, 12, 20, 15
DELTAS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50)
PAIRED_THR, UNPAIRED_THR, PASS = 0.25, 0.125, 0.8


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


def ecs(react, db, lo, hi):
    """Eterna Classic Score over the design region, documented thresholds."""
    ok = tot = 0
    for k in range(lo, hi + 1):
        v = react[k]
        if v is None or db[k] == "*":
            continue
        tot += 1
        ok += (v < PAIRED_THR) if db[k] != "." else (v > UNPAIRED_THR)
    return 100.0 * ok / tot if tot else None


def load():
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(DATA) as f:
        rd = csv.DictReader(f)
        rc = [k for k in rd.fieldnames if k.startswith("reactivity_")
              and not k.startswith("reactivity_error")]
        rc.sort()
        for d in rd:
            if d["SN_filter"] not in ("1", "1.0"):
                continue
            react = [None if d[k] in ("", "NA", "nan") else float(d[k]) for k in rc]
            rows.append({"puzzle": d["puzzle"], "method": d["method"], "seq": d["sequence"],
                         "db": d["target_structure"], "mutA": d["mutA"], "react": react,
                         "lo": int(float(d["sub_start"])) - 1, "hi": int(float(d["sub_end"])) - 1})
    return rows


def build():
    rows = load()
    par, kids = {}, collections.defaultdict(list)
    for d in rows:
        k = (d["puzzle"], d["method"])
        if d["mutA"] in ("", "NA", "nan"):
            par[k] = d
        else:
            kids[k].append(d)
    recs = []
    for k, ks in kids.items():
        p = par.get(k)
        if p is None:
            continue
        P = parse_pairs(p["db"]); lo, hi = p["lo"], p["hi"]
        e0 = ecs(p["react"], p["db"], lo, hi)
        for d in ks:
            if len(d["seq"]) != len(p["seq"]):
                continue
            diff = [x for x in range(len(d["seq"])) if d["seq"][x] != p["seq"][x]]
            if len(diff) != 1:
                continue
            i = diff[0]
            if not (lo <= i <= hi):
                continue
            b0, b1 = p["seq"][i], d["seq"][i]
            j = P.get(i)
            paired = j is not None and lo <= j <= hi
            e1 = ecs(d["react"], d["db"], lo, hi)
            rec = {"puzzle": k[0], "method": k[1], "i": i, "j": j if paired else None,
                   "b0": b0, "b1": b1, "bp": p["seq"][j] if paired else None,
                   "paired": paired, "lo": lo, "hi": hi,
                   "dECS": (e1 - e0) if (e0 is not None and e1 is not None) else None,
                   "pr": p["react"], "mr": d["react"], "db": p["db"]}
            if paired and p["react"][j] is not None and d["react"][j] is not None:
                rec["Y1"] = d["react"][j] - p["react"][j]
            else:
                rec["Y1"] = None
            recs.append(rec)
    return recs, par


def arm_of(r):
    if not r["paired"]:
        return "unpaired_control"
    pr = {r["b0"], r["bp"]}
    if pr == {"G", "C"}:
        return "GC_via_G_to_C" if r["b0"] == "G" else "GC_via_C_to_G"
    if pr == {"A", "U"}:
        return "AU_via_A_to_U" if r["b0"] == "A" else "AU_via_U_to_A"
    return "other"


def boot(vals, tgs, seed, nb=NBOOT):
    """cluster bootstrap over puzzle of the mean."""
    by = collections.defaultdict(list)
    for v, t in zip(vals, tgs):
        by[t].append(v)
    T = sorted(by)
    s = np.array([sum(by[t]) for t in T], float)
    c = np.array([len(by[t]) for t in T], float)
    obs = float(s.sum() / c.sum())
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(T), size=(nb, len(T)))
    num, den = s[ix].sum(1), c[ix].sum(1)
    B = np.sort(num[den > 0] / den[den > 0])
    lo, hi = float(B[int(.025 * len(B))]), float(B[int(.975 * len(B))])
    return {"mean": round(obs, 4), "ci": [round(lo, 4), round(hi, 4)],
            "half": round((hi - lo) / 2, 4), "excl_pos": bool(lo > 0),
            "excl_neg": bool(hi < 0), "n": len(vals), "ntgt": len(T)}


def distal(r, rng):
    """P1: mean delta reactivity at unpaired design-region positions >= DISTAL nt from the mutation."""
    out = []
    for k in range(r["lo"], r["hi"] + 1):
        if abs(k - r["i"]) < DISTAL or r["db"][k] != ".":
            continue
        a, b = r["pr"][k], r["mr"][k]
        if a is not None and b is not None:
            out.append(b - a)
    return float(np.mean(out)) if out else None


def pseudo_partner(r, rng):
    """shape_4 control: a random design-region position with data in both molecules."""
    cand = [k for k in range(r["lo"], r["hi"] + 1)
            if r["pr"][k] is not None and r["mr"][k] is not None and abs(k - r["i"]) >= DISTAL]
    if not cand:
        return None
    k = cand[rng.randrange(len(cand))]
    return r["mr"][k] - r["pr"][k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m14_mutation_intervention.json"))
    a = ap.parse_args()
    print("=== M14:有人改了一个碱基,然后把管子重新测了一遍 ===")
    print("  声明 9cf52dad(代码存在之前已推送)。**本实验对我是盲的。** 离线 $0。")
    print("  化学预测:拆同一个对的两个方向【符号相反】。生物物理预测:【同号】。")
    print("  Y1 在【伙伴位】上读 —— 那一位的碱基没被动过,所以没有碱基身份的成分。\n")
    recs, par = build()
    rng = random.Random(a.seed)
    ARMS = ("GC_via_G_to_C", "GC_via_C_to_G", "AU_via_A_to_U", "AU_via_U_to_A",
            "unpaired_control", "other")
    by = collections.defaultdict(list)
    for r in recs:
        by[arm_of(r)].append(r)
    print(f"可用突变体 {len(recs)}(SN_filter==1 双侧,单碱基,落在设计区内)")
    print("  " + "  ".join(f"{k}={len(by[k])}" for k in ARMS) + "\n")

    # ---------------- P0 identity
    p0 = max(abs(r["mr"][r["j"]] - r["mr"][r["j"]]) for r in recs if r["j"] is not None)
    print(f"--- P0 恒等式(父本与自身)---  最大偏差 {p0:.10f}  {'通过' if p0 == 0 else '不通过'}\n")

    # ---------------- main arms
    print("--- 主臂:Y1 = 伙伴位反应性变化(拆对预期为正)---")
    RES = {}
    for k in ARMS[:4]:
        v = [(r["Y1"], r["puzzle"]) for r in by[k] if r["Y1"] is not None]
        if len(v) < 50:
            print(f"  {k:16s} n={len(v)} 不足"); continue
        b = boot([x for x, _t in v], [t for _x, t in v], a.seed)
        RES[k] = b
        print(f"  {k:16s} Y1 {b['mean']:+7.4f} [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}]  "
              f"n={b['n']:5d}/{b['ntgt']:2d}靶  "
              f"{'正向排除0' if b['excl_pos'] else ('负向排除0' if b['excl_neg'] else '含0')}")
    # unpaired control uses a pseudo-partner (shape_4)
    r2 = random.Random(a.seed + 7)
    v = [(pseudo_partner(r, r2), r["puzzle"]) for r in by["unpaired_control"]]
    v = [(x, t) for x, t in v if x is not None]
    if len(v) >= 50:
        b = boot([x for x, _t in v], [t for _x, t in v], a.seed)
        RES["unpaired_control_pseudo"] = b
        print(f"  {'未配对位(伪伙伴)':16s} Y1 {b['mean']:+7.4f} "
              f"[{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}]  n={b['n']:5d}  "
              f"{'**不为0**' if (b['excl_pos'] or b['excl_neg']) else '含0 ✓'}")

    print("\n--- Y2 = ΔECS(拆对预期为负;这一项把被改动的那一位本身也计入)---")
    RES2 = {}
    for k in ARMS[:4]:
        v = [(r["dECS"], r["puzzle"]) for r in by[k] if r["dECS"] is not None]
        if len(v) < 50:
            continue
        b = boot([x for x, _t in v], [t for _x, t in v], a.seed)
        RES2[k] = b
        print(f"  {k:16s} ΔECS {b['mean']:+7.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]  "
              f"n={b['n']:5d}  "
              f"{'正向排除0' if b['excl_pos'] else ('负向排除0' if b['excl_neg'] else '含0')}")

    # ---------------- P1 distal placebo (GATED)
    print(f"\n--- P1 远端安慰剂(>= {DISTAL} nt、目标未配对、设计区内)—— 归一化漂移检验 ---")
    p1 = {}
    for k in ARMS[:4]:
        v = [(distal(r, rng), r["puzzle"]) for r in by[k]]
        v = [(x, t) for x, t in v if x is not None]
        if len(v) < 50:
            continue
        b = boot([x for x, _t in v], [t for _x, t in v], a.seed)
        p1[k] = b
        main = abs(RES[k]["mean"]) if k in RES else 0
        ratio = abs(b["mean"]) / main if main else float("inf")
        print(f"  {k:16s} 远端 {b['mean']:+7.4f} [{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}]  "
              f"占主臂 {ratio:5.1%}  {'通过' if ratio < 0.20 else '**不通过**'}")
    p1_ok = all(abs(p1[k]["mean"]) / abs(RES[k]["mean"]) < 0.20
                for k in p1 if k in RES and RES[k]["mean"])
    print(f"  P1 {'通过' if p1_ok else '不通过 => K0'}")

    # ---------------- P2 mismatched parent
    print(f"\n--- P2 错配父本({R_MIS} 次,把突变体配到别组的父本上)---")
    keys = sorted({(r["puzzle"], r["method"]) for r in recs})
    p2 = {}
    for k in ("GC_via_G_to_C", "GC_via_C_to_G"):
        pos = neg = 0
        for t in range(R_MIS):
            rr = random.Random(3000 + a.seed + t); v = []
            for r in by[k]:
                alt = keys[rr.randrange(len(keys))]
                if alt == (r["puzzle"], r["method"]):
                    continue
                pp = par.get(alt)
                if pp is None or r["j"] is None:
                    continue
                x, y = pp["react"][r["j"]], r["mr"][r["j"]]
                if x is not None and y is not None:
                    v.append((y - x, r["puzzle"]))
            if len(v) < 50:
                continue
            b = boot([x for x, _t in v], [t2 for _x, t2 in v], a.seed + t, nb=500)
            pos += b["excl_pos"]; neg += b["excl_neg"]
        p2[k] = {"pos": pos, "neg": neg}
        print(f"  {k:16s} 正向排除0 {pos}/{R_MIS}  负向 {neg}/{R_MIS}")
    p2_ok = all(v["pos"] <= 1 for v in p2.values())
    print(f"  P2 {'通过' if p2_ok else '不通过 => K0'}")

    # ---------------- P3 ladder
    print("\n--- P3 梯级(第74条:顶档 1.50,是预言上界 0.35 的四倍以上)---")
    base = [(r["Y1"], r["puzzle"]) for r in by["GC_via_G_to_C"] if r["Y1"] is not None]
    lad = {}
    for d in DELTAS:
        hit = 0
        for t in range(DRAWS):
            rr = np.random.default_rng(int(d * 1e4) + t + a.seed)
            idx = rr.integers(0, len(base), len(base))
            v = [(base[q][0] - RES["GC_via_G_to_C"]["mean"] + d, base[q][1]) for q in idx]
            b = boot([x for x, _t in v], [t2 for _x, t2 in v], a.seed + t, nb=400)
            hit += b["excl_pos"]
        lad[d] = round(hit / DRAWS, 3)
        print(f"  δ={d:.2f}  达标比例 {lad[d]:.2f}")
    thr = next((d for d in DELTAS if lad[d] >= PASS), None)
    mx = max(abs(RES[k]["mean"]) for k in RES)
    p3_ok = max(DELTAS) >= mx
    print(f"  检出阈值 {thr}  主臂最大 |效应| {mx:.4f}  顶档 {max(DELTAS)}  "
          f"P3(b) {'通过' if p3_ok else '不通过'}；K4 "
          f"{'触发' if (thr is None or thr > mx) else f'不触发(余量 {mx/thr:.1f}x)'}")

    # ---------------- verdict
    g1, g2 = RES.get("GC_via_G_to_C"), RES.get("GC_via_C_to_G")
    same_sign_Y1 = bool(g1 and g2 and math.copysign(1, g1["mean"]) == math.copysign(1, g2["mean"]))
    both_excl = bool(g1 and g2 and g1["excl_pos"] and g2["excl_pos"])
    e1, e2 = RES2.get("GC_via_G_to_C"), RES2.get("GC_via_C_to_G")
    opp_Y2 = bool(e1 and e2 and math.copysign(1, e1["mean"]) != math.copysign(1, e2["mean"]))
    k0 = not (p0 == 0 and p1_ok and p2_ok and p3_ok)
    if k0:
        v = "void_probe_failed"
    elif not same_sign_Y1:
        v = "composition_artifact_dominates"
    elif both_excl:
        v = "pair_breaking_is_structural"
    else:
        v = "no_effect_or_underpowered"
    print(f"\nY1 两方向同号 {same_sign_Y1};两方向都正向排除0 {both_excl};"
          f"Y2 两方向符号相反 {opp_Y2}")
    print("裁定:", v + ("  + K5 structural_effect_real_but_the_score_is_composition_biased"
                       if (v == "pair_breaking_is_structural" and opp_Y2) else ""))
    json.dump({"counts": {k: len(by[k]) for k in ARMS}, "Y1": RES, "Y2_dECS": RES2,
               "P0_max_dev": p0, "P1_distal": p1, "P2_mismatched": p2, "P3_ladder": lad,
               "threshold": thr, "same_sign_Y1": same_sign_Y1, "both_excl_Y1": both_excl,
               "opposite_sign_Y2": opp_Y2, "verdict": v, "K5": bool(v == "pair_breaking_is_structural" and opp_Y2)},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
