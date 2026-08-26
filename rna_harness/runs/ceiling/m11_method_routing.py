#!/usr/bin/env python3
"""M11: the layer every net R^2 on this line began by deleting.

Declaration: m11_method_routing_declaration.json, frozen and pushed at 50c4e3d2 BEFORE this file
existed (the eighth declaration done that way).

demean_by_cell subtracts the (puzzle, method) cell mean. Measured on the main population, that step
deletes targets (28.8% of variance) and methods-within-target (24.1%), leaving the within-cell layer
(47.1%, of which 15.1% is tube noise) as the only thing four weeks of experiments looked at. We occupy
20.9% of that layer's explainable part -- 6.7% of the total.

The oracle version of per-target method choice is worth +3.560 [+2.435, +4.739] against +1.25 for the
whole shipped 22-column panel. It is also exactly the shape of an artifact this project has retracted
before, so the primary quantity is CROSS-FITTED: pick the method on half the designs in each cell,
score it on the other half. A and B are independent, so max-of-k cancels by construction.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import load_pop                                        # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
MIN_TARGETS, MIN_HALF, NBOOT, R_PERM, DRAWS, PASS_FRAC = 20, 2, 4000, 12, 20, 0.8
DELTAS = (0.5, 1.0, 2.0, 3.0, 4.5)
MAIN_ARM_MAX = 3.56          # declaration: the ladder's top rung must reach this


def prep(fn, seed):
    """cells -> (A half, B half) of wet values, split by a fixed seed."""
    rec, _s, _m = load_pop(fn)
    cells = collections.defaultdict(list)
    for r in rec:
        if r.get("wet") is not None and r.get("method"):
            cells[(r["puzzle"], r["method"])].append(float(r["wet"]))
    rng = random.Random(seed)
    AB = {}
    for k, v in cells.items():
        v = v[:]; rng.shuffle(v)
        h = len(v) // 2
        if h >= MIN_HALF:
            AB[k] = (v[:h], v[h:])
    return AB


def keepsets(AB):
    tg = collections.defaultdict(dict)
    for (t, m), ab in AB.items():
        tg[t][m] = ab
    nT = len(tg)
    cnt = collections.Counter(m for t in tg for m in tg[t])
    keep = {m for m, c in cnt.items() if c >= max(8, nT // 3)}
    T = {t: {m: ab for m, ab in d.items() if m in keep} for t, d in tg.items()}
    T = {t: d for t, d in T.items() if len(d) >= 3}
    return T, sorted(keep)


def pick_fixed(T, keep):
    """global best single method, chosen on A halves only, within-target centred."""
    dev = collections.defaultdict(list)
    for t, d in T.items():
        mus = {m: st.fmean(ab[0]) for m, ab in d.items()}
        c = st.fmean(list(mus.values()))
        for m, v in mus.items():
            dev[m].append(v - c)
    sc = {m: st.fmean(v) for m, v in dev.items() if len(v) >= 5}
    return max(sc, key=sc.get) if sc else None, sc


def gains(T, m_fix, route=True):
    """G(t) = meanB(t, selected) - meanB(t, m_fix); selection uses A only."""
    out = {}
    for t, d in T.items():
        if m_fix not in d:
            continue
        sel = max(d, key=lambda m: st.fmean(d[m][0])) if route else m_fix
        out[t] = (st.fmean(d[sel][1]) - st.fmean(d[m_fix][1]), sel, len(d[sel][1]))
    return out


def boot(vals, seed, nb=NBOOT):
    if len(vals) < 5:
        return None
    rng = random.Random(seed); B = []
    for _ in range(nb):
        B.append(st.fmean([vals[rng.randrange(len(vals))] for _ in vals]))
    B.sort()
    lo, hi = B[int(.025 * nb)], B[int(.975 * nb)]
    return {"mean": round(st.fmean(vals), 4), "ci": [round(lo, 4), round(hi, 4)],
            "excl_pos": bool(lo > 0), "excl": bool(lo > 0 or hi < 0), "n": len(vals)}


def run_once(T, keep, seed):
    m_fix, sc = pick_fixed(T, keep)
    if m_fix is None:
        return None
    G = gains(T, m_fix)
    v = [g for g, _s, _n in G.values()]
    r = boot(v, seed)
    if r is None:
        return None
    r.update({"m_fix": m_fix, "scores": {k: round(x, 3) for k, x in sc.items()},
              "sel": collections.Counter(s for _g, s, _n in G.values()),
              "thin": st.median([n for _g, _s, n in G.values()]),
              "per_target": {t: round(g, 4) for t, (g, _s, _n) in G.items()}})
    return r


def permuted(T, keep, seed):
    """relabel which cell is which method, INSIDE each target: cell sizes and design membership are
    untouched; only 'method identity means the same thing across targets' is destroyed."""
    rng = random.Random(seed); P = {}
    for t, d in T.items():
        ms = list(d); sh = ms[:]; rng.shuffle(sh)
        P[t] = {a: d[b] for a, b in zip(ms, sh)}
    return P


def synth(T, keep, delta, sigma, seed):
    """same cell structure, wet replaced by pure noise plus delta on one randomly chosen best method
    per target -- injection covers EVERY counted target (rule 71b)."""
    rng = random.Random(seed); S = {}
    for t, d in T.items():
        bm = rng.choice(sorted(d))
        S[t] = {}
        for m, (a, b) in d.items():
            add = delta if m == bm else 0.0
            S[t][m] = ([rng.gauss(0, sigma) + add for _ in a], [rng.gauss(0, sigma) + add for _ in b])
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m11_method_routing.json"))
    a = ap.parse_args()
    print("=== M11:方法层 —— 交叉拟合的路由增益 ===")
    print("  声明 50c4e3d2(代码存在之前已推送)。离线 $0。选择只用 A 半,评估只用 B 半。\n")

    D, RES = {}, {}
    for nm, fn in POPS:
        AB = prep(fn, a.seed)
        T, keep = keepsets(AB)
        if len(T) < MIN_TARGETS:
            print(f"=== {nm} ===  可用靶标 {len(T)} < {MIN_TARGETS} ⇒ 跳过\n"); continue
        D[fn] = (nm, T, keep)
        sig = st.fmean([st.pstdev(ab[0] + ab[1]) for t in T for ab in T[t].values() if len(ab[0]+ab[1]) > 2])
        r = run_once(T, keep, a.seed)
        print(f"=== {nm} ===  靶标 {len(T)}  方法 {len(keep)}  格内 sd {sig:.3f}")
        print(f"  A 半选出的全局最好方法 m_fix = **{r['m_fix']}**")
        print(f"  **交叉拟合路由增益 = {r['mean']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]**"
              f"  {'**排除0**' if r['excl_pos'] else '含 0'}   (n={r['n']} 靶标)")
        print(f"  被选中格在 B 半的设计数中位 {r['thin']:.1f}"
              f"{'  **thin_cells**' if r['thin'] < 3 else ''}")
        top = r["sel"].most_common(4)
        print(f"  被路由选中的方法:{', '.join(f'{m}×{c}' for m, c in top)}")
        RES[fn] = {"pop": nm, "n_targets": len(T), "methods": keep, "sigma": round(sig, 4), **{
            k: v for k, v in r.items() if k not in ("sel", "per_target")},
            "sel": dict(r["sel"]), "per_target": r["per_target"]}
        print()

    if len(RES) < 2:
        json.dump({"verdict": "no_data", "pops": RES}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("=== 裁定 ===\n  **no_data**"); return 0

    print("=== P3 恒等式:把选择规则换成『永远选 m_fix』 ===")
    ok3 = True
    for fn, (nm, T, keep) in D.items():
        m_fix, _ = pick_fixed(T, keep)
        v = [g for g, _s, _n in gains(T, m_fix, route=False).values()]
        mx = max(abs(x) for x in v)
        print(f"  {nm:<13s} 最大绝对增益 {mx:.6f}  {'✓' if mx < 1e-12 else '✗'}")
        ok3 &= mx < 1e-12

    print("\n=== P1 置换零点(靶标内重排方法标签,12 次,单侧)===")
    perm_hits = {}
    for fn, (nm, T, keep) in D.items():
        h = 0
        for j in range(R_PERM):
            rp = run_once(permuted(T, keep, a.seed + 3167 * j), keep, a.seed + j)
            if rp and rp["excl_pos"]:
                h += 1
        perm_hits[fn] = h
        print(f"  {nm:<13s} 12 次里正向排除 0 的 **{h}/12**  {'✓' if h <= 1 else '✗ 作废'}")

    print(f"\n=== P2 构造性梯级(第 71 条:刻度实测、顶档 >= {MAIN_ARM_MAX}、注入覆盖全部靶标)===")
    fn0 = max(RES, key=lambda f: RES[f]["n_targets"])
    nm0, T0, keep0 = D[fn0]
    sig0 = RES[fn0]["sigma"]
    print(f"  在 {nm0} 的格结构上({RES[fn0]['n_targets']} 靶标,sd {sig0:.3f})")
    print(f"  {'注入 delta':>10s} {'实测 E[增益]':>13s} {'达标比例':>9s} {'反解逐靶标 p':>13s} {'解析功率':>9s}")
    LAD, thr = {}, None
    for dl in DELTAS:
        gm, hits, sds = [], [], []
        for k in range(DRAWS):
            S = synth(T0, keep0, dl, sig0, a.seed + 7919 * k + int(dl * 100))
            rs = run_once(S, keep0, a.seed + k)
            if rs is None:
                continue
            gm.append(rs["mean"]); hits.append(1 if rs["excl_pos"] else 0)
            sds.append(st.pstdev(list(rs["per_target"].values())))
        if not gm:
            continue
        E = st.fmean(gm); f = st.fmean(hits)
        n = RES[fn0]["n_targets"]; se = st.fmean(sds) / math.sqrt(n)
        # analytic power for a one-sided-at-95%-two-sided interval excluding 0
        z = E / se if se > 0 else 0.0
        ap_ = 0.5 * (1 + math.erf((z - 1.96) / math.sqrt(2)))
        LAD[str(dl)] = {"E_gain": round(E, 4), "pass_frac": round(f, 3), "analytic": round(ap_, 3)}
        print(f"  {dl:10.2f} {E:13.4f} {f:9.2f} {'':13s} {ap_:9.3f}")
        if thr is None and f >= PASS_FRAC:
            thr = E
    top_reached = max((v["E_gain"] for v in LAD.values()), default=0.0)
    print(f"  **检出阈值(以实测 E[增益] 为刻度)= {thr if thr else '> ' + f'{top_reached:.2f}'}**")
    print(f"  顶档实测 E[增益] = {top_reached:.3f}  {'✓ >= ' if top_reached >= MAIN_ARM_MAX else '✗ < '}{MAIN_ARM_MAX}")

    obs = {fn: RES[fn]["mean"] for fn in RES}
    mx_obs = max(obs.values())
    K0 = (not ok3) or any(h > 1 for h in perm_hits.values()) or top_reached < MAIN_ARM_MAX
    K1 = sum(1 for fn in RES if RES[fn]["excl_pos"]) >= 2
    K2 = all((not RES[fn]["excl_pos"]) or RES[fn]["mean"] <= 0 for fn in RES)
    K3 = len(RES) < 2
    K4 = (thr is None) or thr > mx_obs
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "P1/P2/P3 任一不合格 ⇒ 作废"),
                    ("K1", K1, f"排除 0 的人群 {sum(1 for f in RES if RES[f]['excl_pos'])}/{len(RES)} >= 2"),
                    ("K2", K2, "全部含 0 或 <=0 ⇒ max-of-k 假象"),
                    ("K3", K3, "可用人群 < 2 ⇒ 无数据"),
                    ("K4", K4, f"阈值 {thr} > 主臂最大 {mx_obs:.3f} ⇒ 无裁决力")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    verdict = ("void_probe_failed" if K0 else "no_data" if K3 else
               "no_adjudicating_power" if K4 else
               "routing_gain_survives_max_of_k" if K1 and not K2 else
               "routing_gain_is_max_of_k_artifact" if K2 and not K1 else "unresolved")
    # rule 64 shapes
    shapes = []
    for fn in RES:
        if RES[fn]["mean"] < 0:
            shapes.append(f"shape1_sign_flip:{RES[fn]['pop']}")
        if RES[fn]["thin"] < 3:
            shapes.append(f"shape3_thin_cells:{RES[fn]['pop']}")
    for fn, (nm, T, keep) in D.items():
        sel = RES[fn]["sel"]
        if sel:
            dom = max(sel, key=sel.get)
            T2 = {t: {m: v for m, v in d.items() if m != dom} for t, d in T.items()}
            T2 = {t: d for t, d in T2.items() if len(d) >= 3}
            if len(T2) >= 10:
                r2 = run_once(T2, [m for m in keep if m != dom], a.seed)
                if r2:
                    RES[fn]["drop_dominant"] = {"method": dom, "mean": r2["mean"], "ci": r2["ci"],
                                                "excl_pos": r2["excl_pos"]}
                    if RES[fn]["excl_pos"] and not r2["excl_pos"]:
                        shapes.append(f"shape2_driven_by_{dom}:{nm}")
    if shapes:
        verdict += "__" + shapes[0]
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    for s in shapes:
        print(f"  第 64 条形状:{s}")
    print("\n  **第 61 条照引(跑前写死):我预期交叉拟合把 +3.56 砍到 +1.2 ~ +2.5,主人群排除 0,样本3 不确定。**")
    for fn in RES:
        d = RES[fn].get("drop_dominant")
        if d:
            print(f"  去掉 {d['method']} 后 {RES[fn]['pop']}:{d['mean']:+.4f} "
                  f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] {'排除0' if d['excl_pos'] else '含 0'}")

    json.dump({"verdict": verdict, "pops": RES, "ladder": LAD, "threshold": thr,
               "top_rung_E": round(top_reached, 4), "perm_hits": perm_hits, "P3_exact": bool(ok3),
               "shapes": shapes,
               "kill": {k: bool(v) for k, v in (("K0", K0), ("K1", K1), ("K2", K2), ("K3", K3), ("K4", K4))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
