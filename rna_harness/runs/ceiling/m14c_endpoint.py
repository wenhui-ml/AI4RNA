#!/usr/bin/env python3
"""M14c: lower start, higher finish -- and a ladder that is finally built on zero.

Declaration: m14c_endpoint_declaration.json, frozen and pushed at dbb58eed BEFORE this file existed
(the fifteenth). NOT blind: M14b's two strata are already known to me, and the declaration says the
only clause that counts as a prediction is partner_U's endpoint difference.

Three repairs:
  primary statistic -> the ENDPOINT (post-mutation partner reactivity), immune to the headroom
                       reading that shape_4 raised, because G-C starts LOWER and finishes HIGHER
  inference         -> permutation (rule 76), bootstrap reported beside it
  ladder            -> built on the NULL with a delta=0 self-check (rule 79), after five consecutive
                       rule-71 violations

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from m14_mutation_intervention import build, DISTAL                                  # noqa: E402
from m14b_partner_matched import baselines, pairtype, STRATA                         # noqa: E402

NPERM, R_SCRAM, DRAWS, PASS, P_GATE = 2000, 12, 20, 0.8, 0.005
DELTAS = (0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.50)
SELF_CHECK_MAX = 0.15


def cluster_mean_diff(A, B):
    """A, B are lists of (value, puzzle). Cluster-balanced difference of means."""
    da, db = collections.defaultdict(float), collections.defaultdict(int)
    ea, eb = collections.defaultdict(float), collections.defaultdict(int)
    for v, t in A:
        da[t] += v; db[t] += 1
    for v, t in B:
        ea[t] += v; eb[t] += 1
    sa, ca = sum(da.values()), sum(db.values())
    sb, cb = sum(ea.values()), sum(eb.values())
    return (sa / ca if ca else 0.0) - (sb / cb if cb else 0.0)


def boot_diff(A, B, seed, nb=NPERM):
    T = sorted({t for _v, t in A} | {t for _v, t in B})
    ia = collections.defaultdict(list); ib = collections.defaultdict(list)
    for v, t in A:
        ia[t].append(v)
    for v, t in B:
        ib[t].append(v)
    sa = np.array([sum(ia[t]) for t in T]); ca = np.array([len(ia[t]) for t in T], float)
    sb = np.array([sum(ib[t]) for t in T]); cb = np.array([len(ib[t]) for t in T], float)
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(T), size=(nb, len(T)))
    na, d1 = sa[ix].sum(1), ca[ix].sum(1); nbb, d2 = sb[ix].sum(1), cb[ix].sum(1)
    ok = (d1 > 0) & (d2 > 0)
    D = np.sort(na[ok] / d1[ok] - nbb[ok] / d2[ok])
    return [round(float(D[int(.025 * len(D))]), 4), round(float(D[int(.975 * len(D))]), 4)]


def perm_null(A, B, seed, nperm=NPERM):
    """rule 72: scramble ONLY the strong/weak label; molecule and partner base untouched."""
    allv = A + B; nA = len(A)
    rng = np.random.default_rng(seed); out = []
    for _ in range(nperm):
        p = rng.permutation(len(allv))
        out.append(cluster_mean_diff([allv[q] for q in p[:nA]], [allv[q] for q in p[nA:]]))
    return np.sort(np.array(out))


def pval(obs, null):
    return (float((null >= obs).sum()) + 1) / (len(null) + 1)


def ladder(A, B, seed):
    """rule 79: build the dial on the NULL. Scramble labels first (true difference = 0), then
    inject delta into one side. delta=0 must fall back to nominal or the ladder itself is void."""
    allv = A + B; nA = len(A)
    null = perm_null(A, B, seed, nperm=800)
    thr = float(np.quantile(null, .995))
    out = {}
    for d in DELTAS:
        hit = 0
        for t in range(DRAWS):
            rng = np.random.default_rng(int(d * 1e4) + t + seed)
            p = rng.permutation(len(allv))
            A2 = [(allv[q][0] + d, allv[q][1]) for q in p[:nA]]
            B2 = [allv[q] for q in p[nA:]]
            hit += (cluster_mean_diff(A2, B2) > thr)
        out[d] = round(hit / DRAWS, 3)
    return out, thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m14c_endpoint.json"))
    a = ap.parse_args()
    print("=== M14c:起点更低而终点更高 —— 主统计量换成【终点】,梯级建在零上 ===")
    print("  声明 dbb58eed(代码存在之前已推送)。**M14b 的结果我已看过,本轮不是独立验证。**")
    print("  唯一计分的盲预言:partner_U 层触发 K3(功率不足,非零效应)。离线 $0。\n")
    recs, _p = build()
    base = baselines(recs)
    rows = []
    for r in recs:
        if not r["paired"] or r["Y1"] is None:
            continue
        b, nb = base.get(((r["puzzle"], r["method"]), r["j"]), (None, 0))
        rows.append({"puzzle": r["puzzle"], "bp": r["bp"], "pt": pairtype(r),
                     "start": r["pr"][r["j"]], "end": r["mr"][r["j"]],
                     "rise": r["Y1"], "rise_adj": (r["Y1"] - b) if b is not None else None})
    OUT = {"strata": {}}
    for name, bp, strong, weak in STRATA:
        S = [r for r in rows if r["bp"] == bp and r["pt"] == strong]
        W = [r for r in rows if r["bp"] == bp and r["pt"] == weak]
        print(f"=== {name}(伙伴碱基 = {bp})===  拆 {strong} n={len(S)}   拆 {weak} n={len(W)}")
        if len(W) < 30:
            print("  弱对臂不足\n"); continue
        st_s = float(np.mean([r["start"] for r in S])); st_w = float(np.mean([r["start"] for r in W]))
        print(f"  起点(父本 j 位):拆{strong} {st_s:+.4f}   拆{weak} {st_w:+.4f}   "
              f"差 {st_s - st_w:+.4f}   {'(强对起点更低)' if st_s < st_w else ''}")
        ent = {"n_strong": len(S), "n_weak": len(W),
               "start_strong": round(st_s, 4), "start_weak": round(st_w, 4)}
        for key, lab in (("end", "**终点(主)**"), ("rise", "上升幅度"), ("rise_adj", "上升(扣基线)")):
            A = [(r[key], r["puzzle"]) for r in S if r[key] is not None]
            B = [(r[key], r["puzzle"]) for r in W if r[key] is not None]
            if len(B) < 30:
                continue
            obs = cluster_mean_diff(A, B)
            null = perm_null(A, B, a.seed)
            ci = boot_diff(A, B, a.seed)
            ma = float(np.mean([x for x, _t in A])); mb = float(np.mean([x for x, _t in B]))
            ent[key] = {"strong": round(ma, 4), "weak": round(mb, 4), "diff": round(obs, 4),
                        "p_perm": round(pval(obs, null), 5), "null_sd": round(float(null.std()), 4),
                        "boot_ci": ci, "half": round((ci[1] - ci[0]) / 2, 4)}
            print(f"  {lab:14s} 拆{strong} {ma:+.4f}  拆{weak} {mb:+.4f}  "
                  f"差 {obs:+.4f}  置换 p {ent[key]['p_perm']:.5f}  零SD {ent[key]['null_sd']:.4f}  "
                  f"bootstrap {ci} 半宽 {ent[key]['half']:.4f}")
        # P2
        A = [(r["end"], r["puzzle"]) for r in S]; B = [(r["end"], r["puzzle"]) for r in W]
        null = perm_null(A, B, a.seed)
        pos = 0
        for t in range(R_SCRAM):
            rng = np.random.default_rng(9000 + a.seed + t)
            allv = A + B; p = rng.permutation(len(allv))
            d = cluster_mean_diff([allv[q] for q in p[:len(A)]], [allv[q] for q in p[len(A):]])
            pos += (pval(d, null) < P_GATE)
        ent["P2_scramble_pos"] = pos
        print(f"  P2 标签对调 {R_SCRAM} 次(以置换 p<{P_GATE} 计):{pos}  {'通过' if pos <= 1 else '**不通过**'}")
        # P3 ladder on the null with delta=0 self-check
        lad, thr = ladder(A, B, a.seed)
        ent["ladder"] = {str(k): v for k, v in lad.items()}
        sc = lad[0.00]
        det = next((d for d in DELTAS if d > 0 and lad[d] >= PASS), None)
        ent["ladder_threshold"] = det; ent["ladder_delta0"] = sc
        print(f"  P3 梯级(建在零上,n={len(B)}):" + "  ".join(f"{d:.2f}:{lad[d]:.2f}" for d in DELTAS))
        print(f"     **δ=0 自检 {sc:.2f}** {'通过' if sc <= SELF_CHECK_MAX else '**不通过 => 梯级作废**'}"
              f"   检出阈值 {det}")
        # leave-one-puzzle-out on the endpoint difference
        pz = sorted({t for _v, t in B})
        loo = [cluster_mean_diff([x for x in A if x[1] != q], [x for x in B if x[1] != q])
               for q in pz if sum(1 for x in B if x[1] != q) >= 20]
        ent["loo_range"] = [round(min(loo), 4), round(max(loo), 4)] if loo else None
        print(f"  留一 puzzle 极差(终点差){ent['loo_range']}\n")
        OUT["strata"][name] = ent

    G = OUT["strata"].get("partner_G", {}); U = OUT["strata"].get("partner_U", {})
    sc_ok = all(v.get("ladder_delta0", 1) <= SELF_CHECK_MAX for v in OUT["strata"].values())
    p2_ok = all(v.get("P2_scramble_pos", 9) <= 1 for v in OUT["strata"].values())
    ge = G.get("end", {})
    k4 = G.get("ladder_threshold") is None or (G.get("ladder_threshold") or 9) > abs(ge.get("diff", 0))
    if not sc_ok:
        v = "void_ladder_self_check_failed"
    elif not p2_ok:
        v = "void_probe_failed"
    elif k4:
        v = "no_adjudicating_power"
    elif ge.get("diff", 0) > 0 and ge.get("p_perm", 1) < P_GATE:
        v = "stronger_pair_ends_higher"
    elif ge.get("diff", 0) <= 0:
        v = "no_endpoint_difference"
    else:
        v = "under_replicated"
    ue = U.get("end", {}); ut = U.get("ladder_threshold")
    u_verdict = ("underpowered_not_null" if (ut is None or ut > abs(ue.get("diff", 0)))
                 else ("endpoint_difference" if ue.get("p_perm", 1) < P_GATE else "null_with_power"))
    print(f"partner_G 裁定: {v}")
    print(f"partner_U 裁定: {u_verdict}  (差 {ue.get('diff')}  梯级阈值 {ut})")
    # K5
    if ge and G.get("rise"):
        k5 = math.copysign(1, ge["diff"]) != math.copysign(1, G["rise"]["diff"])
        print(f"K5 终点与上升幅度结论相反: {k5}")
        OUT["K5_statistic_dependent"] = bool(k5)
    OUT["verdict_partner_G"] = v; OUT["verdict_partner_U"] = u_verdict
    json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
