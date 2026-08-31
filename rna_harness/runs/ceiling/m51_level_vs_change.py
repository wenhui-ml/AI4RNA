#!/usr/bin/env python3
"""M51: the interval section 5.6 says in print that it does not compute.

Declaration m51_level_vs_change_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0.

Section 5.6 scores each predictor twice on the same cells -- the level it assigns the mutant, and
the change between mutant and wild type -- and then writes: "We report the level arm in the table
because it is the larger of the two, and we put no interval on the difference between them." That
is a choice between two arms made on two point estimates with the interval declined, which is the
shape this paper withdraws a metric for and grades a kill condition as failed for.

The estimator here -- the cell-size-weighted mean of within-cell Spearman correlations -- is
written in this file. M26 supplies records; M31 supplies nothing but the published numbers this
run has to reproduce before it is allowed to report anything.
"""
from __future__ import annotations

import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

DECL = os.path.join(HERE, "m51_level_vs_change_declaration.json")
OUTJ = os.path.join(HERE, "m51_level_vs_change.json")
D = json.load(open(DECL))
NMIN, NBOOT, SEED = 8, 2000, 20260831
LOG = []

# level arm, change arm, and the published value each must reproduce
ARMS = [("RibonanzaNet", "nmut", "ndp", 0.3295, 0.0491),
        ("ViennaRNA", "pmut", "dp", 0.2557, -0.0184),
        ("EternaFold", "emut", "edp", 0.2765, -0.0458)]
PUB_DEPTH = 0.3225


def log(s=""):
    print(s, flush=True); LOG.append(s)


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    log(f"\n写出 {OUTJ}")


def cz(v):
    """centred, unit-norm rank vector; None when the column is constant inside the cell."""
    v = np.asarray(v, float) - np.mean(v)
    n = float(np.sqrt(v @ v))
    return None if n == 0 else v / n


def wrho(cells, key, ry):
    """cell-size-weighted mean within-cell Spearman, as a dot product of normalised rank vectors."""
    num = den = 0.0
    for k, rs in cells:
        rx = cz(rankdata([r[key] for r in rs]))
        if rx is None or ry[k] is None:
            continue
        num += len(rs) * float(rx @ ry[k]); den += len(rs)
    return num / den if den else None


def wrho_cov(cells, key, ykey="rf"):
    """V2's equivalent form: covariance of ranks over the product of their SDs, per cell."""
    num = den = 0.0
    for k, rs in cells:
        x = rankdata([r[key] for r in rs]); y = rankdata([r[ykey] for r in rs])
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            continue
        num += len(rs) * float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)); den += len(rs)
    return num / den if den else None


def pct(v, lo=2.5, hi=97.5):
    v = np.sort(np.asarray(v, float))
    return [round(float(v[int(lo / 100 * len(v))]), 5), round(float(v[int(hi / 100 * len(v))]), 5)]


def main():
    t0 = time.time()
    log("=== M51:给 §5.6 那句“不给区间”补上区间 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL), "cost": D["cost"]}

    # M50's record cache is not usable here: the change arms need each predictor's value on the
    # WILD TYPE as well as on the mutant, and the cache carries only the mutant columns.
    zn = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {str(k): v for k, v in zip(zn["keys"], zn["mats"])}
    ze = np.load(os.path.join(HERE, "m32_eterna_cache.npz"), allow_pickle=True)
    pe = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in zip(ze["keys"], ze["maps"])}
    full, _dr = M26.build()
    miss = 0
    recs = []
    for r in full:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        if r["mut_seq"] not in pn or r["wt_seq"] not in pn or r["mut_seq"] not in pe \
           or r["wt_seq"] not in pe:
            miss += 1; continue
        r["nmut"] = float(pn[r["mut_seq"]][i, j]); r["nwt"] = float(pn[r["wt_seq"]][i, j])
        r["ndp"] = r["nmut"] - r["nwt"]
        r["emut"] = float(pe[r["mut_seq"]].get((i + 1, j + 1), 0.0))
        r["ewt"] = float(pe[r["wt_seq"]].get((i + 1, j + 1), 0.0))
        r["edp"] = r["emut"] - r["ewt"]
        recs.append(r)
    log(f"  记录 {len(recs)},四列齐备之外缺 {miss}  ({time.time()-t0:.0f}s)")
    OUT["n_records"] = len(recs); OUT["n_missing_a_column"] = miss

    by = collections.defaultdict(list)
    for r in recs:
        by[r["design"]].append(r)
    cells = [(k, v) for k, v in by.items() if len(v) >= NMIN]
    ncov = sum(len(v) for _k, v in cells)
    log(f"  至少 {NMIN} 个翻转的设计格 {len(cells)},覆盖 {ncov} 次干预")
    OUT["n_cells"] = len(cells); OUT["n_covered"] = ncov
    if len(cells) < 20:
        log("  **K3 触发:格太少 ⇒ 只报点估计**")
    ry = {k: cz(rankdata([r["rf"] for r in rs])) for k, rs in cells}

    # ================================================================ V
    log("\n=== V 验证(先于任何 arm)===")
    ok = {}
    for r in recs:
        r["_neg"] = -r["rf"]
    ok["V4_self_is_one"] = abs(wrho(cells, "rf", ry) - 1.0) < 1e-12
    ok["V4_neg_is_minus_one"] = abs(wrho(cells, "_neg", ry) + 1.0) < 1e-12
    log(f"  V4 与自身 {wrho(cells,'rf',ry):+.12f}   与自身取负 {wrho(cells,'_neg',ry):+.12f}")

    v2 = []
    for _nm, lv, ch, _pl, _pc in ARMS:
        for key in (lv, ch):
            v2.append(abs(wrho(cells, key, ry) - wrho_cov(cells, key)))
    ok["V2_equivalent_form"] = max(v2) < 1e-12
    log(f"  V2 点积路 vs 秩协方差路 max|Δ| = {max(v2):.2e}")
    OUT["V2_max_gap"] = float(max(v2))

    pt, drift = {}, {}
    for nm, lv, ch, pl, pc in ARMS:
        pt[(nm, "level")] = wrho(cells, lv, ry); pt[(nm, "change")] = wrho(cells, ch, ry)
        drift[f"{nm}_level"] = abs(pt[(nm, "level")] - pl)
        drift[f"{nm}_change"] = abs(pt[(nm, "change")] - pc)
        log(f"  V3 {nm:14s} level {pt[(nm,'level')]:+.5f}(论文 {pl:+.4f})  "
            f"change {pt[(nm,'change')]:+.5f}(论文 {pc:+.4f})")
    pt[("depth", "level")] = wrho(cells, "depth", ry)
    drift["depth"] = abs(pt[("depth", "level")] - PUB_DEPTH)
    log(f"  V3 {'depth':14s} {pt[('depth','level')]:+.5f}(论文 {PUB_DEPTH:+.4f})")
    ok["V3_reverse_control"] = max(drift.values()) < 1e-4
    OUT["A1_reverse_control"] = {k: round(v, 8) for k, v in drift.items()}
    OUT["point_estimates"] = {f"{a}_{b}": round(v, 5) for (a, b), v in pt.items()}

    OUT["V"] = ok
    if not ok["V3_reverse_control"]:
        log(f"\n  **K1 触发:复现不出已发表的秩一致性(最大偏差 {max(drift.values()):.2e})⇒ 作废**")
        OUT["verdict"] = "VOID_K1_cannot_reproduce_published_rank_agreements"; save(OUT); return 1
    if not (ok["V2_equivalent_form"] and ok["V4_self_is_one"] and ok["V4_neg_is_minus_one"]):
        log("\n  **K2 触发 ⇒ 作废**")
        OUT["verdict"] = "VOID_K2_verification_failed"; save(OUT); return 1
    log("  V 全部通过 —— 现在才读 arm")

    # ================================================================ A2/A3 one bootstrap
    log("\n=== A2/A3 配对差(同一次目标聚类抽样)===")
    pz = sorted({rs[0]["puzzle"] for _k, rs in cells})
    bypz = collections.defaultdict(list)
    for k, rs in cells:
        bypz[rs[0]["puzzle"]].append((k, rs))
    g = np.random.default_rng(SEED)
    dif = {nm: [] for nm, *_ in ARMS}
    vsd = {f"{nm}_{arm}": [] for nm, *_ in ARMS for arm in ("level", "change")}
    self_d = []
    for _ in range(NBOOT):
        ix = g.integers(0, len(pz), len(pz))
        sub = [c for q in ix for c in bypz[pz[q]]]
        ry2 = {k: cz(rankdata([r["rf"] for r in rs])) for k, rs in sub}
        dep = wrho(sub, "depth", ry2)
        vals = {}
        bad = False
        for nm, lv, ch, _pl, _pc in ARMS:
            a = wrho(sub, lv, ry2); b = wrho(sub, ch, ry2)
            if a is None or b is None or dep is None:
                bad = True; break
            vals[(nm, "level")] = a; vals[(nm, "change")] = b
        if bad:
            continue
        for nm, *_ in ARMS:
            dif[nm].append(vals[(nm, "level")] - vals[(nm, "change")])
            vsd[f"{nm}_level"].append(vals[(nm, "level")] - dep)
            vsd[f"{nm}_change"].append(vals[(nm, "change")] - dep)
        self_d.append(vals[("RibonanzaNet", "level")] - vals[("RibonanzaNet", "level")])

    ok["V1_self_difference_exactly_zero"] = bool(self_d and max(map(abs, self_d)) == 0.0)
    log(f"  V1 一臂对自己 max|.| = {max(map(abs, self_d)) if self_d else 'n/a'}")
    OUT["V"] = ok
    if not ok["V1_self_difference_exactly_zero"]:
        log("\n  **K2 触发 ⇒ 作废**"); OUT["verdict"] = "VOID_K2_self_difference"; save(OUT); return 1

    A2 = {}
    for nm, *_ in ARMS:
        p = pt[(nm, "level")] - pt[(nm, "change")]
        ci = pct(dif[nm])
        A2[nm] = {"level": round(pt[(nm, "level")], 5), "change": round(pt[(nm, "change")], 5),
                  "difference": round(p, 5), "ci95": ci,
                  "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])}
        log(f"  A2 {nm:14s} level {pt[(nm,'level')]:+.5f} − change {pt[(nm,'change')]:+.5f} "
            f"= {p:+.5f}  {ci}{'  含 0' if A2[nm]['ci_contains_zero'] else '  不含 0'}")
    OUT["A2_level_minus_change"] = A2

    A3 = {}
    log("\n  A3 两臂各自对免费深度基线的配对差:")
    for nm, *_ in ARMS:
        for arm in ("level", "change"):
            ci = pct(vsd[f"{nm}_{arm}"])
            p = pt[(nm, arm)] - pt[("depth", "level")]
            A3[f"{nm}_{arm}"] = {"difference": round(p, 5), "ci95": ci,
                                 "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])}
            log(f"    {nm:14s} {arm:7s} − depth = {p:+.5f}  {ci}"
                f"{'  含 0' if A3[f'{nm}_{arm}']['ci_contains_zero'] else '  不含 0'}")
    OUT["A3_vs_depth"] = A3

    # ================================================================ scorecard
    log("\n=== 预注册记分 ===")
    st = {}
    for i, (sid, nm) in enumerate([("S1", "RibonanzaNet"), ("S2", "ViennaRNA"), ("S3", "EternaFold")]):
        ok_i = not A2[nm]["ci_contains_zero"]
        if sid == "S3":
            biggest = max(A2, key=lambda k: A2[k]["difference"])
            ok_i = ok_i and biggest == "EternaFold"
            det = f"{A2[nm]['difference']:+.5f} {A2[nm]['ci95']};最大的是 {biggest}"
        else:
            det = f"{A2[nm]['difference']:+.5f} {A2[nm]['ci95']}"
        st[sid] = {"text": D["staked_predictions"][i]["text"], "pass": bool(ok_i), "detail": det}
    st["S4"] = {"text": D["staked_predictions"][3]["text"],
                "pass": all(A2[nm]["difference"] > 0 for nm, *_ in ARMS),
                "detail": ", ".join(f"{nm} {A2[nm]['difference']:+.5f}" for nm, *_ in ARMS)}
    for k, v in st.items():
        log(f"  {k}  {'成立' if v['pass'] else '**失败**'}   {v['detail']}")
    OUT["staked"] = st
    nres = sum(1 for nm, *_ in ARMS if not A2[nm]["ci_contains_zero"])
    OUT["n_resolving"] = nres
    OUT["verdict"] = f"level_minus_change_resolves_in_{nres}_of_{len(ARMS)}"
    log(f"\n  裁定:{OUT['verdict']}   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
