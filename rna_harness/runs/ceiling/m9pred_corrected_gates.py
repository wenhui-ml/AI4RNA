#!/usr/bin/env python3
"""M9-pre-d: same question, gates on one scale, ladder with error bars, difficulty controlled.

Declaration: m9pred_corrected_gates_declaration.json, frozen and pushed at b213b5cf BEFORE this file
existed. Supersedes M9-pre-c, which voided on two gates that were both my own scale error.

THREE CHANGES, NOTHING ELSE.
  1 (rule 68) K1 compares the main arm against the SPEARMAN-BROWN full-sample ceiling 2r/(1+r), not the
    split-half value; the bridge threshold moves to its attenuation-corrected form. M9-pre-c compared
    0.3656 against 0.2427 when the right ceiling was 0.3906, and 0.3431 against a flat 0.4 when its
    attenuation ceiling was 0.4686.
  2 (rule 67) Twenty independent noise draws per ladder rung, pass fraction and spread reported. One
    draw per rung is what produced a ladder where 0.3 cleared the bar and 0.4 did not.
  3 (rule 64 shape 2) A target-difficulty control on the unmatched construction, plus the main arm's
    partial correlation net of difficulty. Under matching, difficulty was undetectable (+0.157); the
    unmatched construction may carry it much better, in which case the main arm is difficulty.

PARTIAL CORRELATION, THE M8C LESSON. Rank-transform, linear residualize, Pearson on the residuals, and
NEVER re-rank the residuals -- re-ranking destroys the orthogonality least squares guarantees, which is
how M8c produced a number that was neither a partial Spearman nor anything with a name. P4 asserts the
residual is orthogonal to the conditioning variable to machine precision, or the run is void.

COST: $0, offline.
"""
from __future__ import annotations

import argparse, json, math, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import numpy as np                                                            # noqa: E402
from scipy.stats import pearsonr                                              # noqa: E402
import h2_heldout_cluster_ci as H                                             # noqa: E402
from m3_gc_cancelled import build                                             # noqa: E402
from m9pre_heavytail_vs_concentration import MIN_COMMON, rho_ci, shape        # noqa: E402
from m9prec_unmatched_heavytail import (POPS, BLIND_FN, per_target_c, gv,     # noqa: E402
                                        pairs_tagged, synth, RUNGS)

DRAWS, PASS_FRAC = 20, 0.8
R_MATCHED_HALF = 0.3911          # M9-pre-b measured, matched construction


def sb(r):
    """Spearman-Brown: half-sample reliability -> full-sample reliability."""
    return 2 * r / (1 + r) if r > -1 else float("nan")


def rk(v):
    a = np.asarray(v, dtype=float)
    return np.argsort(np.argsort(a)).astype(float)


def resid(y, x):
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def partial(x, y, z):
    """standard partial correlation on ranks; returns (r, orthogonality_check)."""
    rx, ry, rz = rk(x), rk(y), rk(z)
    ex, ey = resid(rx, rz), resid(ry, rz)
    orth = max(abs(pearsonr(ex, rz)[0]), abs(pearsonr(ey, rz)[0]))
    return pearsonr(ex, ey)[0], orth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "m9pred_corrected_gates.json"))
    a = ap.parse_args()
    print("=== M9-pre-d:闸门放到同一尺度、梯级带误差条、扣掉靶标难度 ===")
    print(f"  声明 b213b5cf(代码存在之前已推送)。离线 $0。梯级每档 {DRAWS} 次抽样。\n")

    G, Gn, Gdif, Ghalf, names, SH, P2 = {}, {}, {}, {}, {}, {}, {}
    for nm, fn in POPS:
        print(f"=== {nm} ===", flush=True)
        keep, M = build(fn, a.workers)
        names[fn] = nm
        per, used, s = per_target_c(keep, M, a.seed, "none")
        if per is None or len(used) < H.MIN_TARGETS:
            print("  无数据"); continue
        G[fn] = gv(per, used)
        Gdif[fn] = {t: st.fmean(per["A"][t]) for t in used}          # 难度对照
        SH[fn] = shape(list(G[fn].values()))
        gg = [st.fmean(per["A_G"][t]) - st.fmean(per["A"][t]) for t in used]
        ci, _n = H.boot_ci(gg, a.seed + 17)
        P2[fn] = {"mean": round(st.fmean(gg), 4), "ci": ci, "excl": bool(ci[0] > 0 or ci[1] < 0)}
        print(f"  靶标 {len(used)}  P2 A_G−A {P2[fn]['mean']:+.4f} "
              f"{'✓' if P2[fn]['excl'] else '✗'}  峰度 {SH[fn]['kurtosis']}")
        pn, un, _ = per_target_c(keep, M, a.seed, "none", noise_rng=random.Random(a.seed + 991))
        if pn is not None and len(un) >= H.MIN_TARGETS:
            Gn[fn] = gv(pn, un)
        ok = [i for i, m in enumerate(M) if "error" not in m and "ens_defect_skel" in m]
        rng = random.Random(a.seed + 5); rng.shuffle(ok)
        p1, u1, _ = per_target_c(keep, M, a.seed, "none", subset=set(ok[: len(ok) // 2]))
        p2, u2, _ = per_target_c(keep, M, a.seed, "none", subset=set(ok[len(ok) // 2:]))
        if p1 is not None and p2 is not None:
            g1, g2 = gv(p1, u1), gv(p2, u2)
            cm = sorted(set(g1) & set(g2))
            if len(cm) >= MIN_COMMON:
                r, _c = rho_ci([g1[t] for t in cm], [g2[t] for t in cm], a.seed + 31)
                Ghalf[fn] = {"rho": round(r, 4), "n": len(cm)}
                print(f"  信度 对半 {r:+.4f} ⇒ **全样本(SB) {sb(r):+.4f}**")

    if len(G) < 4:
        json.dump({"verdict": "no_data"}, open(a.out, "w"), indent=1, ensure_ascii=False); return 0

    half_mean = st.fmean([v["rho"] for v in Ghalf.values()])
    CEIL = sb(half_mean)
    print(f"\n  **对半天花板均值 {half_mean:+.4f} ⇒ Spearman-Brown 全样本天花板 {CEIL:+.4f}**"
          f"  (M9-pre-c 的 K1 错在拿 {half_mean:.4f} 当门槛)")

    print(f"\n=== 构造性梯级(第 67 条:每档 {DRAWS} 次抽样)===")
    LAD, thr = {}, None
    for rho in RUNGS:
        cp, fp = [], []
        for d in range(DRAWS):
            Gs = synth(G, rho, a.seed + int(rho * 100) + 7919 * d)
            _r, c, cf, _bb = pairs_tagged(Gs, names, a.seed, "", quiet=True)
            cp.append(1 if c[2] and c[0] >= math.ceil(5 / 6 * c[2]) else 0)
            fp.append(1 if cf[2] and cf[0] >= math.ceil(7 / 9 * cf[2]) else 0)
        fc, ff = st.fmean(cp), st.fmean(fp)
        LAD[str(rho)] = {"core_pass_frac": round(fc, 3), "conf_pass_frac": round(ff, 3),
                         "core_raw": cp, "conf_raw": fp}
        print(f"  rho={rho:.1f}  发现集达标 {fc:.2f}  **确证集达标 {ff:.2f}**  "
              f"({int(ff*DRAWS)}/{DRAWS} 次)")
        if thr is None and ff >= PASS_FRAC:
            thr = rho
    print(f"  **检出阈值(达标比例 >= {PASS_FRAC}) = {thr if thr else '> 0.6'}**"
          f"  vs 全样本天花板 {CEIL:.4f}")

    _rn, cn, _cf, _b = pairs_tagged(Gn, names, a.seed, "P1 等宽纯噪声") if len(Gn) >= 4 \
        else ([], (9, 0, 0), (9, 0, 0), (0, 0, 0))

    print("\n=== 靶标难度对照(不配平构造)===")
    _rd, dc, dcf, _db = pairs_tagged(Gdif, names, a.seed, "难度 A 的跨人群持续性")

    print("\n=== 主臂 ===")
    rows, core, conf, bb = pairs_tagged(G, names, a.seed, "主臂 g_t = A_ed − A(不配平)")

    print("\n=== 主臂扣掉难度之后的偏相关(标准定义,残差不重排)===")
    prs, orths = [], []
    ks = sorted(G)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            aa, bbk = ks[i], ks[j]
            cm = sorted(set(G[aa]) & set(G[bbk]))
            if len(cm) < MIN_COMMON:
                continue
            zz = [(Gdif[aa][t] + Gdif[bbk][t]) / 2 for t in cm]
            r, o = partial([G[aa][t] for t in cm], [G[bbk][t] for t in cm], zz)
            prs.append(r); orths.append(o)
    par_mean = st.fmean(prs); raw_mean = st.fmean([r["rho"] for r in rows if not r.get("void")])
    P4 = max(orths)
    print(f"  裸相关均值 **{raw_mean:+.4f}** → 偏相关均值 **{par_mean:+.4f}**"
          f"(保留 {100*par_mean/raw_mean:.0f}%)")
    print(f"  **P4 正交性 {P4:.3e}** {'✓ 机器零' if P4 < 1e-12 else '✗ 残差被重排了 ⇒ 作废'}")

    K0 = (cn[0] > 1) or (not all(v["excl"] for v in P2.values())) or (P4 >= 1e-12)
    K1 = raw_mean > CEIL
    K2 = conf[2] > 0 and conf[0] >= math.ceil(7 / 9 * conf[2])
    K3 = conf[2] > 0 and conf[0] <= math.floor(2 / 9 * conf[2])
    K4 = sum(1 for r in rows if r.get("void")) >= 5
    K5 = (thr is None) or (thr > CEIL)
    K6 = par_mean <= raw_mean / 2
    print("\n=== 判决条件 ===")
    for k, v, d in (("K0", K0, "噪声/构造/正交性任一不合格 ⇒ 作废"),
                    ("K1", K1, f"主臂 {raw_mean:.4f} > SB 全样本天花板 {CEIL:.4f} ⇒ 作废"),
                    ("K2", K2, f"确证集 {conf[0]}/{conf[2]} >= 7/9"),
                    ("K3", K3, "确证集 <= 2/9"), ("K4", K4, "作废对 >= 5"),
                    ("K5", K5, f"检出阈值 {thr} > 天花板 {CEIL:.4f} ⇒ 无裁决力"),
                    ("K6", K6, "偏相关 <= 裸相关一半 ⇒ 主要是靶标难度")):
        print(f"  {k}: {'**触发**' if v else '未触发'}   {d}")

    if K0 or K1 or K4:
        verdict = "void_" + ("probe_failed" if K0 else ("above_ceiling" if K1 else "no_data"))
    elif K5:
        verdict = "no_adjudicating_power__even_unmatched"
    elif K2 and not K3:
        verdict = "concentrated_effect__persists_on_blind"
    elif K3 and not K2:
        verdict = "heavy_tailed_noise__no_persistence_on_blind"
    else:
        verdict = "unresolved"
    if K6:
        verdict += "__mostly_target_difficulty"

    print(f"\n=== 裁定 ===\n  **{verdict}**")
    print(f"  主臂 裸 {raw_mean:+.4f} / 偏 {par_mean:+.4f};发现集 {core[0]}/{core[2]},"
          f"确证集 {conf[0]}/{conf[2]},盲×盲 {bb[0]}/{bb[2]}")
    print(f"  难度对照 发现集 {dc[0]}/{dc[2]},确证集 {dcf[0]}/{dcf[2]}")
    print(f"\n  **跑前预测:0.3 档达标比例 << 0.8,真实阈值落在 0.4~0.6,K5 触发。**"
          f"  实测阈值 {thr},K5 {'触发 ⇒ 预测成立' if K5 else '未触发 ⇒ **预测被否,本臂有裁决力**'}")
    print("  **污染:六个人群对本问题全部已降为发现集,本次不主张任何确证。**")

    json.dump({"verdict": verdict, "threshold": thr, "ceiling_half": round(half_mean, 4),
               "ceiling_full_SB": round(CEIL, 4), "ladder": LAD, "main": rows,
               "raw_mean": round(raw_mean, 4), "partial_mean": round(par_mean, 4),
               "P4_orthogonality": float(P4), "difficulty": {"core": [int(x) for x in dc], "conf": [int(x) for x in dcf]},
               "counts": {k: [int(x) for x in v] for k, v in (("core",core),("conf",conf),("blind_blind",bb))},
               "shape": {names[k]: v for k, v in SH.items()},
               "kill": {k: bool(v) for k, v in (("K0",K0),("K1",K1),("K2",K2),("K3",K3),("K4",K4),("K5",K5),("K6",K6))}},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
