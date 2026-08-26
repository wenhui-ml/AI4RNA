#!/usr/bin/env python3
"""M12: the twenty-two shipped columns measure what the oracles believe. None measures the sequence.

Declaration: m12_sequence_intrinsic_declaration.json, frozen and pushed at 5e2beeb9 BEFORE this file
existed (the ninth declaration done that way).

revB said features_exhausted__new_columns_required. Reading base_block says which columns are missing:
every one of the 22 is a statistic of oracle-prediction-versus-target. The claim under test is that a
block of SEQUENCE-INTRINSIC columns still adds on top of panel PLUS GC -- GC sits in the baseline so a
known-strong column cannot be passed off as a discovery.

S1 calls no folding oracle at all. If S1 alone adds, that is the first result on this line that
predicts a wet readout without asking any predictor what it thinks.

The experiment ends in something runnable: held-out capture@10 in wet units, M4b's currency.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, statistics as st, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from e1_ensemble_marginals import base_block, load_pop                          # noqa: E402

POPS = (("样本2(主)", "panel_f1_fresh.json"), ("基准4560", "panel_f1_designs.json"),
        ("样本3", "panel_f1_third.json"), ("第四批", "panel_f1_fourth.json"))
CEILING, NFOLD, NBOOT, R_NOISE, DRAWS, PASS = 0.6956, 5, 2000, 12, 20, 0.8
RHOS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45)   # widened after the first run voided on rule 71(b): the top rung reached E[dR2]=0.0438 against a main arm of 0.0746. ONLY the probe dial changed; the main arm is untouched and must reproduce bit-for-bit.
PAIRABLE = {"A": "U", "U": "AG", "G": "CU", "C": "G"}
OPEN, CLOSE = "([{<", ")]}>"
S1 = ["spur_max", "spur_cnt", "spur_margin", "loop_max", "helix_min", "gc_paired", "gc_unpaired"]
S2 = ["egap", "ed_full", "dG_gap"]
GC = ["gc_cross", "gc_frac"]


def parse_pairs(db):
    stacks = {c: [] for c in OPEN}; out = set()
    for i, c in enumerate(db):
        if c in OPEN:
            stacks[c].append(i)
        elif c in CLOSE:
            s = stacks[OPEN[CLOSE.index(c)]]
            if s:
                out.add((s.pop(), i))
    return out


def nested_proj(db):
    return "".join(c if c in "().." else ("(" if c in OPEN[1:] else ")" if c in CLOSE[1:] else ".")
                   for c in db).replace("(", "(", 1) if False else \
        "".join(c if c in "()." else "." for c in db)


def helices(P):
    P = sorted(P); seen = set(); out = []
    for (i, j) in P:
        if (i - 1, j + 1) in P:
            continue
        k = 0
        while (i + k, j - k) in P:
            seen.add((i + k, j - k)); k += 1
        out.append(k)
    return out or [0]


def spurious(seq, P, minloop=3, thr=4):
    n = len(seq); best = 0; cnt = 0
    for i in range(n):
        pi = PAIRABLE[seq[i]]
        for j in range(i + minloop + 1, n):
            if seq[j] not in pi:
                continue
            if i and j + 1 < n and seq[j + 1] in PAIRABLE[seq[i - 1]]:
                continue
            k, native = 0, False
            while i + k < j - k - minloop and seq[j - k] in PAIRABLE[seq[i + k]]:
                if (i + k, j - k) in P:
                    native = True
                k += 1
            if not native:
                best = max(best, k)
                cnt += (k >= thr)
    return best, cnt


def work(item):
    puz, seq, target = item
    try:
        import RNA
        n = len(seq); P = parse_pairs(target)
        if not P or len(seq) != len(target):
            return {"puzzle": puz, "error": "no pairs / length mismatch"}
        H = helices(P)
        pos = set()
        for (i, j) in P:
            pos.add(i); pos.add(j)
        unp = [k for k in range(n) if k not in pos]
        lm, run = 0, 0
        for k in range(n):
            run = run + 1 if k in unp else 0
            lm = max(lm, run)
        sm, sc = spurious(seq, P)
        gcp = sum(1 for k in pos if seq[k] in "GC") / max(1, len(pos))
        gcu = sum(1 for k in unp if seq[k] in "GC") / max(1, len(unp))
        nest = "".join(c if c in "()." else "." for c in target)
        fc = RNA.fold_compound(seq)
        ms, me = fc.mfe()
        fc.exp_params_rescale(me)
        _s, efe = fc.pf()
        bpp = fc.bpp()
        pm = {}
        for (i, j) in P:
            a, b = (i + 1, j + 1) if i < j else (j + 1, i + 1)
            pm[(i, j)] = float(bpp[a][b])
        qu = []
        for k in unp:
            tot = 0.0
            for m in range(n):
                a, b = (k + 1, m + 1) if k < m else (m + 1, k + 1)
                if a != b:
                    tot += float(bpp[a][b])
            qu.append(1.0 - tot)
        ed = (sum(1 - v for v in pm.values()) + sum(1 - v for v in qu)) / n
        dg = float(fc.eval_structure(nest)) - float(me)
        gcx = 0; S = set(parse_pairs(target))
        try:
            from gamma_star import crossing_set
            S = set(crossing_set(target))
        except Exception:                                                        # noqa: BLE001
            S = set()
        for (i, j) in sorted(S):
            if (i + 1, j - 1) in S:
                for (x, y) in ((i, j), (i + 1, j - 1)):
                    if {seq[x], seq[y]} == {"G", "C"}:
                        gcx += 1
        return {"puzzle": puz, "spur_max": float(sm), "spur_cnt": float(sc),
                "spur_margin": float(sm - max(H)), "loop_max": float(lm),
                "helix_min": float(min(H)), "gc_paired": gcp, "gc_unpaired": gcu,
                "egap": float(me - efe) / n, "ed_full": ed, "dG_gap": dg,
                "gc_cross": float(gcx), "gc_frac": (seq.count("G") + seq.count("C")) / n}
    except Exception as exc:                                                     # noqa: BLE001
        return {"puzzle": puz, "error": repr(exc)[:150]}


def build(fn, workers):
    from leaderboard_design import targets as load_targets
    T = load_targets()
    rec, _s, _m = load_pop(fn)
    keep = [r for r in rec if base_block(r) is not None and r.get("seq")
            and r.get("wet") is not None and r["puzzle"] in T]
    jobs = [(r["puzzle"], r["seq"], T[r["puzzle"]]["target"]) for r in keep]
    from concurrent.futures import ProcessPoolExecutor
    M = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m in ex.map(work, jobs, chunksize=16):
            M.append(m)
    rows = []
    for r, m in zip(keep, M):
        if "error" in m:
            continue
        rows.append({"puzzle": r["puzzle"], "method": r["method"], "wet": float(r["wet"]),
                     "B": base_block(r), "F": m})
    return rows


def matrices(rows, names):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["puzzle"], r["method"])].append(r)
    rws = [(c, g) for c, g in cells.items() if len(g) >= 3]
    ys, Xs, tg, ci, wets = [], [], [], [], []
    for k, (cell, grp) in enumerate(rws):
        y = np.array([r["wet"] for r in grp], float)
        X = np.array([r["B"] + [r["F"][n] for n in names] for r in grp], float)
        ys.append(y - y.mean()); Xs.append(X - X.mean(axis=0))
        tg += [cell[0]] * len(grp); ci += [k] * len(grp); wets.append(y)
    return (np.concatenate(ys), np.vstack(Xs), np.array(tg), np.array(ci),
            len(rws), [g for _c, g in rws])


def folds(tg, k, seed):
    u = sorted(set(tg.tolist())); random.Random(seed).shuffle(u)
    a = {t: i % k for i, t in enumerate(u)}
    return np.array([a[t] for t in tg])


def cv_pred(y, X, fold, cols):
    p = np.zeros(len(y))
    for f in sorted(set(fold.tolist())):
        te, tr = fold == f, fold != f
        if tr.sum() <= len(cols) + 10 or te.sum() == 0:
            continue
        b, *_ = np.linalg.lstsq(X[tr][:, cols], y[tr], rcond=None)
        p[te] = X[te][:, cols] @ b
    return p


def dr2_boot(y, pa, pb, tg, seed, nb=NBOOT):
    """paired cluster bootstrap over targets of R2(b) - R2(a)."""
    T = sorted(set(tg.tolist())); idx = {t: np.where(tg == t)[0] for t in T}
    sa = {t: float(((y[i] - pa[i]) ** 2).sum()) for t, i in idx.items()}
    sb = {t: float(((y[i] - pb[i]) ** 2).sum()) for t, i in idx.items()}
    ss = {t: float((y[i] ** 2).sum()) for t, i in idx.items()}
    def d(ts):
        S = sum(ss[t] for t in ts)
        return 0.0 if S <= 0 else (sum(sa[t] for t in ts) - sum(sb[t] for t in ts)) / S
    obs = d(T); rng = random.Random(seed); B = []
    for _ in range(nb):
        B.append(d([T[rng.randrange(len(T))] for _ in T]))
    B.sort()
    lo, hi = B[int(.025 * nb)], B[int(.975 * nb)]
    return {"dr2": round(obs, 5), "ci": [round(lo, 5), round(hi, 5)],
            "excl_pos": bool(lo > 0), "excl": bool(lo > 0 or hi < 0)}


def capture(groups, pa, pb, ci, y, seed, n=10, R=200):
    """held-out absolute gain in wet units: E[wet of top-1 of n by prediction] - E[wet of random]."""
    rng = random.Random(seed); ga, gb = [], []
    for k, grp in enumerate(groups):
        m = np.where(ci == k)[0]
        if len(m) < n:
            continue
        w = y[m]
        for _ in range(R):
            s = [rng.randrange(len(m)) for _ in range(n)]
            base = st.fmean([w[i] for i in s])
            ga.append(w[max(s, key=lambda i: pa[m][i])] - base)
            gb.append(w[max(s, key=lambda i: pb[m][i])] - base)
    if not ga:
        return None
    return {"A": round(st.fmean(ga), 4), "B": round(st.fmean(gb), 4),
            "gain": round(st.fmean(gb) - st.fmean(ga), 4), "cells": len(ga) // R}


def synth_col(y, ci, rho, rng):
    """a column with correlation rho to the within-cell demeaned y, itself demeaned within cell."""
    sd = float(np.std(y)) or 1.0
    c = rho * (y / sd) + math.sqrt(max(0.0, 1 - rho * rho)) * np.array(
        [rng.gauss(0, 1) for _ in range(len(y))])
    for k in range(int(ci.max()) + 1):
        m = ci == k
        c[m] = c[m] - c[m].mean()
    return c


def arm_dr2(y, X, fold, base, extra, tg, seed):
    """dR2 of base+extra over base, where extra is a matrix of appended columns."""
    XA = X[:, base]
    XB = np.hstack([XA, extra])
    pa = cv_pred(y, XB, fold, list(range(XA.shape[1])))
    pb = cv_pred(y, XB, fold, list(range(XB.shape[1])))
    return dr2_boot(y, pa, pb, tg, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m12_sequence_intrinsic.json"))
    a = ap.parse_args()
    print("=== M12:序列内在的那一族列,在【面板 + GC】之上还有没有留出增量 ===")
    print("  声明 5e2beeb9(代码存在之前已推送)。离线 $0。基线是 B+G,不是 B。\n")

    DATA, RES = {}, {}
    for nm, fn in POPS:
        rows = build(fn, a.workers)
        if len(rows) < 500:
            print(f"=== {nm} ===  可用分子 {len(rows)} < 500 ⇒ 跳过\n"); continue
        y, X, tg, ci, ncell, groups = matrices(rows, GC + S1 + S2)
        if len(set(tg.tolist())) < 20:
            print(f"=== {nm} ===  靶标 {len(set(tg.tolist()))} < 20 ⇒ 跳过\n"); continue
        nb = len(rows[0]["B"])
        idxB = list(range(nb)); idxG = [nb + i for i in range(len(GC))]
        idxS1 = [nb + len(GC) + i for i in range(len(S1))]
        idxS2 = [nb + len(GC) + len(S1) + i for i in range(len(S2))]
        fold = folds(tg, NFOLD, a.seed)
        base = idxB + idxG
        DATA[fn] = (nm, y, X, tg, ci, fold, groups, base, idxS1, idxS2)
        pB = cv_pred(y, X, fold, base)
        r2b = 1 - float(((y - pB) ** 2).sum()) / float((y * y).sum())
        print(f"=== {nm} ===  分子 {len(y)}  格 {ncell}  靶标 {len(set(tg.tolist()))}"
              f"  列 B={nb} G={len(GC)} S1={len(S1)} S2={len(S2)}")
        print(f"  基线 B+G 留出 R^2 = {r2b:.4f}  (占上限 {r2b/CEILING*100:.1f}%)")
        rec = {"pop": nm, "n": len(y), "cells": ncell, "targets": len(set(tg.tolist())),
               "r2_base": round(r2b, 5), "arms": {}}
        for k, cols in (("S1", idxS1), ("S2", idxS2), ("S1+S2", idxS1 + idxS2)):
            p = cv_pred(y, X, fold, base + cols)
            d = dr2_boot(y, pB, p, tg, a.seed)
            cap = capture(groups, pB, p, ci, y, a.seed)
            cg = "--" if cap is None else f"{cap['gain']:+.4f}"
            rec["arms"][k] = {**d, "capture": cap}
            print(f"  **B+G+{k:<6s} dR2 = {d['dr2']:+.5f} [{d['ci'][0]:+.5f},{d['ci'][1]:+.5f}]**"
                  f"  {'**排除0**' if d['excl_pos'] else '含 0 '}   留出 capture@10 增益 {cg}")
        # shape 2: drop each S column, does the increment survive
        best = "S1+S2"
        drop = {}
        for j, cname in enumerate(S1 + S2):
            cols = [c for c in idxS1 + idxS2 if c != (idxS1 + idxS2)[j]]
            p = cv_pred(y, X, fold, base + cols)
            drop[cname] = dr2_boot(y, pB, p, tg, a.seed)["dr2"]
        rec["drop_one"] = {k: round(v, 5) for k, v in drop.items()}
        worst = min(drop, key=drop.get)
        print(f"  去掉单列后 dR2 最低的是 **{worst}** ⇒ {drop[worst]:+.5f}"
              f"(全块 {rec['arms'][best]['dr2']:+.5f})")
        # shape 3: in-sample versus held-out
        bi, *_ = np.linalg.lstsq(X[:, base], y, rcond=None)
        fi, *_ = np.linalg.lstsq(X[:, base + idxS1 + idxS2], y, rcond=None)
        ss = float((y * y).sum())
        ins = (float(((y - X[:, base] @ bi) ** 2).sum())
               - float(((y - X[:, base + idxS1 + idxS2] @ fi) ** 2).sum())) / ss
        rec["dr2_insample"] = round(ins, 5)
        print(f"  样本内 dR2 {ins:+.5f} vs 留出 {rec['arms'][best]['dr2']:+.5f}"
              f"{'   **insample_only**' if ins > 0.005 and rec['arms'][best]['dr2'] <= 0 else ''}")
        RES[fn] = rec
        print()

    if len(RES) < 3:
        json.dump({"verdict": "no_data", "pops": RES}, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("=== 裁定 ===\n  **no_data**"); return 0

    NS = len(S1) + len(S2)
    print("=== P0 恒等式:S := B+G 里已有的一列的拷贝 ⇒ dR2 必须恰好 0 ===")
    p0ok = True
    for fn, (nm, y, X, tg, ci, fold, gr, base, i1, i2) in DATA.items():
        d = arm_dr2(y, X, fold, base, X[:, [base[0]]], tg, a.seed)
        print(f"  {nm:<13s} dR2 = {d['dr2']:+.8f}  {'✓' if abs(d['dr2']) < 1e-9 else '✗'}")
        p0ok &= abs(d["dr2"]) < 1e-9

    print(f"\n=== P1 等宽纯噪声 {R_NOISE} 根(单侧,第 60 条)===")
    p1 = {}
    for fn, (nm, y, X, tg, ci, fold, gr, base, i1, i2) in DATA.items():
        rng = random.Random(a.seed + 101); h = 0
        for _ in range(R_NOISE):
            E = np.array([[rng.gauss(0, 1) for _ in range(NS)] for _ in range(len(y))])
            for k in range(int(ci.max()) + 1):
                m = ci == k
                E[m] = E[m] - E[m].mean(axis=0)
            h += arm_dr2(y, X, fold, base, E, tg, a.seed)["excl_pos"]
        p1[fn] = h
        print(f"  {nm:<13s} 正向排除 0 的 **{h}/{R_NOISE}**  {'✓' if h <= 1 else '✗ 作废'}")

    print(f"\n=== P3 第 72 条:格内打乱 S 的行(B+G 与 y 不动)===")
    p3 = {}; p3base = True
    for fn, (nm, y, X, tg, ci, fold, gr, base, i1, i2) in DATA.items():
        pB = cv_pred(y, X, fold, base)
        r0 = 1 - float(((y - pB) ** 2).sum()) / float((y * y).sum())
        rng = random.Random(a.seed + 202); h = 0
        for _ in range(R_NOISE):
            E = X[:, i1 + i2].copy()
            for k in range(int(ci.max()) + 1):
                m = np.where(ci == k)[0]
                pm = list(m); rng.shuffle(pm)
                E[m] = X[:, i1 + i2][pm]
            h += arm_dr2(y, X, fold, base, E, tg, a.seed)["excl_pos"]
        pB2 = cv_pred(y, X, fold, base)
        r1 = 1 - float(((y - pB2) ** 2).sum()) / float((y * y).sum())
        p3base &= abs(r0 - r1) < 1e-12
        p3[fn] = h
        print(f"  {nm:<13s} 正向排除 0 的 **{h}/{R_NOISE}**  {'✓' if h <= 1 else '✗ 作废'}"
              f"   基线自检 {'逐位不变 ✓' if abs(r0-r1) < 1e-12 else '✗ 基线被动了'}")

    obs = max(RES[f]["arms"]["S1+S2"]["dr2"] for f in RES)
    print(f"\n=== P2 构造性梯级(第 71 条:刻度实测、顶档 >= 主臂最大 {obs:+.5f}、四人群全做)===")
    print(f"  {'注入 rho':>9s} {'实测 E[dR2]':>13s} {'达标(>=3/4 排除0)':>19s} {'解析功率':>9s}")
    LAD, thr = {}, None
    for rho in RHOS:
        ds, hits = [], []
        for k in range(DRAWS):
            rng = random.Random(a.seed + 7919 * k + int(rho * 1000))
            ok, dv = 0, []
            for fn, (nm, y, X, tg, ci, fold, gr, base, i1, i2) in DATA.items():
                c = synth_col(y, ci, rho, rng).reshape(-1, 1)
                d = arm_dr2(y, X, fold, base, c, tg, a.seed)
                ok += d["excl_pos"]; dv.append(d["dr2"])
            hits.append(1 if ok >= 3 else 0); ds.append(st.fmean(dv))
        E, f = st.fmean(ds), st.fmean(hits)
        sd = st.pstdev(ds) or 1e-9
        z = E / (sd / math.sqrt(max(1, DRAWS)))
        ap_ = 0.5 * (1 + math.erf((z - 1.96) / math.sqrt(2)))
        LAD[str(rho)] = {"E_dr2": round(E, 5), "pass_frac": round(f, 3), "analytic": round(ap_, 3)}
        print(f"  {rho:9.2f} {E:13.5f} {f:19.2f} {ap_:9.3f}")
        if thr is None and f >= PASS:
            thr = E
    top = max((v["E_dr2"] for v in LAD.values()), default=0.0)
    print(f"  **检出阈值(实测 E[dR2] 刻度)= {thr if thr else '> ' + f'{top:.5f}'}**;顶档 {top:+.5f} "
          f"{'✓ >=' if top >= obs else '✗ <'} 主臂最大 {obs:+.5f}")

    npos = sum(1 for f in RES if RES[f]["arms"]["S1+S2"]["excl_pos"])
    K0 = (not p0ok) or (not p3base) or any(v > 1 for v in p1.values()) \
        or any(v > 1 for v in p3.values()) or top < obs
    K1 = npos >= 3
    K2 = all((not RES[f]["arms"]["S1+S2"]["excl_pos"]) or RES[f]["arms"]["S1+S2"]["dr2"] <= 0
             for f in RES)
    K3 = len(RES) < 3
    K4 = (thr is None) or thr > obs
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "P0/P1/P2/P3 任一不合格 ⇒ 作废"),
                    ("K1", K1, f"排除 0 的人群 {npos}/{len(RES)} >= 3 ⇒ 序列内在块有增量"),
                    ("K2", K2, "全部含 0 或 <=0 ⇒ 无增量"), ("K3", K3, "可用人群 < 3"),
                    ("K4", K4, f"阈值 {thr} > 主臂最大 {obs:+.5f} ⇒ 无裁决力")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")
    verdict = ("void_probe_failed" if K0 else "no_data" if K3 else
               "no_adjudicating_power" if K4 else
               "sequence_intrinsic_block_adds" if K1 and not K2 else
               "no_increment_over_panel_plus_gc" if K2 and not K1 else "unresolved")
    s1pos = sum(1 for f in RES if RES[f]["arms"]["S1"]["excl_pos"])
    s2pos = sum(1 for f in RES if RES[f]["arms"]["S2"]["excl_pos"])
    if s2pos > s1pos:
        verdict += "__shape4_driven_by_ensemble_not_combinatorics"
    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  S1(无 oracle)排除 0 的人群 {s1pos}/{len(RES)};S2(系综){s2pos}/{len(RES)}")
    print("\n  **第 61 条照引(跑前写死):我预期 dR2 落在 +0.01 ~ +0.05,在 >=2 个人群上排除 0,"
          "且 S1 是最大的单一贡献者。**")
    json.dump({"verdict": verdict, "pops": RES, "ladder": LAD, "threshold": thr,
               "top_rung": round(top, 5), "P0_exact": bool(p0ok), "P1": p1, "P3": p3,
               "P3_baseline_unchanged": bool(p3base),
               "kill": {k: bool(v) for k, v in (("K0", K0), ("K1", K1), ("K2", K2),
                                                ("K3", K3), ("K4", K4))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
