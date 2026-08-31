#!/usr/bin/env python3
"""M49: apply C8's own criterion to C7, and put an interval on the arm asymmetry.

Declaration m49_c7_did_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0 -- no API, no GPU, no download.

WHY THIS RUN EXISTS
-------------------
The ledger marks C7 (leakage) EMPTY on a difference whose interval covers zero, and marks C8
(probe chemistry) UNRESOLVED on a difference whose interval covers zero. What separated them was a
criterion -- does the interval reach far enough to permit the whole quantity being measured -- that
was stated for C8 and never applied to C7.

And the sentence C7 rests on ("exposed designs are harder, not remembered") does not explain its
own numbers. ViennaRNA cannot have memorised anything and loses ALL of its skill on the exposed
subset (0.13342 -> 0.00115). RibonanzaNet could have, and keeps some (0.21720 -> 0.13463). If
exposure were only difficulty, both would fall alike. The quantity that separates the two readings
is the difference in the drops, and this paper's own rule 69 forbids reporting two point estimates
without an interval on their difference. There was no such interval. This computes it.

WHAT IS IMPORTED AND WHAT IS NOT
--------------------------------
M26.build() supplies the RECORDS -- that is the corpus, not the estimator. The exposure labels are
read out of m39_exposure_tiers.json's published identity_by_design, so the 526 MB scan is not
repeated and its result is taken as given. The slope, the cluster bootstrap, the permutation and
the difference-in-differences are all written here and import nothing from m39 or m26d, which is
what makes A1/A2 a reverse control rather than a tautology.
"""
from __future__ import annotations

import collections, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26

DECL = os.path.join(HERE, "m49_c7_did_declaration.json")
M39J = os.path.join(HERE, "m39_exposure_tiers.json")
OUTJ = os.path.join(HERE, "m49_c7_did.json")
D = json.load(open(DECL))
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------------------- the estimator
# Written here, vectorised over a stratum index, rather than M26's dict-of-lists loop. Same
# quantity: the pooled within-stratum slope, sum_s sum_i (x-xbar_s)(y-ybar_s) / sum_s sum_i
# (x-xbar_s)^2. A stratum of size 1 contributes nothing to either sum and so drops out on its own.
def wslope(x, y, sid):
    """x, y float arrays; sid an integer stratum label per row. Returns (slope|None, n_used)."""
    k = int(sid.max()) + 1 if len(sid) else 0
    cnt = np.bincount(sid, minlength=k).astype(float)
    keep = cnt[sid] >= 2
    if not keep.any():
        return None, 0
    x, y, sid = x[keep], y[keep], sid[keep]
    cnt = np.bincount(sid, minlength=k).astype(float)
    xb = np.bincount(sid, weights=x, minlength=k) / np.maximum(cnt, 1)
    yb = np.bincount(sid, weights=y, minlength=k) / np.maximum(cnt, 1)
    dx = x - xb[sid]; dy = y - yb[sid]
    sxx = float(dx @ dx)
    return (float(dx @ dy) / sxx if sxx > 0 else None), int(len(x))


def wslope_fwl(x, y, sid):
    """V2's equivalent form: Frisch-Waugh-Lovell -- least squares of y on x with stratum dummies,
    which must equal the demeaned ratio above. Different linear algebra, same estimand."""
    k = int(sid.max()) + 1
    cnt = np.bincount(sid, minlength=k).astype(float)
    keep = cnt[sid] >= 2
    x, y, sid = x[keep], y[keep], sid[keep]
    used = np.unique(sid); remap = {s: i for i, s in enumerate(used)}
    col = np.array([remap[s] for s in sid])
    Dm = np.zeros((len(x), len(used) + 1))
    Dm[np.arange(len(x)), col] = 1.0
    Dm[:, -1] = x
    beta, *_ = np.linalg.lstsq(Dm, y, rcond=None)
    return float(beta[-1])


def pct(v, lo=2.5, hi=97.5):
    v = np.sort(np.asarray(v, float))
    return [round(float(v[int(lo / 100 * len(v))]), 5), round(float(v[int(hi / 100 * len(v))]), 5)]


def main():
    t0 = time.time()
    log("=== M49:把 C8 的判据套到 C7,并给臂间不对称一个区间 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL), "cost": D["cost"],
           "known_before_the_run": D["known_before_the_run"]}

    # ------------------------------------------------------------------ records
    recs, _dr = M26.build()
    zc = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(zc["keys"], zc["mats"])}
    miss = [r for r in recs if r["mut_seq"] not in pn]
    if miss:
        log(f"  **神经缓存缺 {len(miss)} 条 ⇒ 作废**")
        OUT["verdict"] = "void_neural_cache_incomplete"; save(OUT); return 1
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, zc
    log(f"  干预 {len(recs)} 条,载入 {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------ exposure, from M39
    m39 = json.load(open(M39J))
    THR = m39["constants"]["THRESH"]; SEED = m39["constants"]["SEED"]; NBOOT = m39["constants"]["NBOOT"]
    ident = {tuple(k.split("|", 1)): v for k, v in m39["E2"]["identity_by_design"].items()}
    exposed_cells = {k for k, v in ident.items() if v >= THR}
    log(f"  M39 的曝光标签:{len(exposed_cells)}/{len(ident)} 个设计单元 (阈 {THR})")
    if len(exposed_cells) != m39["E1"]["n_exposed_designs"]:
        log("  **曝光集合与 M39 的 E1 不符 ⇒ 作废**")
        OUT["verdict"] = "void_exposure_label_mismatch"; save(OUT); return 1

    # stratum = (design cell, capped depth, crossing) -- the paper's stratification, unchanged
    skey = {}
    for r in recs:
        k = (r["design"], min(r["depth"], 3), r["crossed"])
        skey.setdefault(k, len(skey))
        r["_s"] = skey[k]
    for r in recs:
        r["_exp"] = r["design"] in exposed_cells
    clean = [r for r in recs if not r["_exp"]]
    dirty = [r for r in recs if r["_exp"]]
    log(f"  未曝光 {len(clean)},曝光 {len(dirty)}  (M39: {m39['n_clean']}, {m39['n_dirty']})")
    if (len(clean), len(dirty)) != (m39["n_clean"], m39["n_dirty"]):
        log("  **split 与 M39 不符 ⇒ 作废**")
        OUT["verdict"] = "void_split_mismatch"; save(OUT); return 1
    OUT["split"] = {"n_clean": len(clean), "n_exposed": len(dirty),
                    "n_exposed_cells": len(exposed_cells), "n_cells": len(ident)}

    def arrs(rs, pk):
        return (np.array([r[pk] for r in rs], float),
                np.array([r["rf"] for r in rs], float),
                np.array([r["_s"] for r in rs], int))

    MODELS = (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA"))

    # ================================================================== V: verify before reading
    log("\n=== V 验证(先于任何 arm)===")
    ok = {}

    # V4 identity: slope of a column on itself is exactly 1; on a constant column, undefined
    x, y, s = arrs(recs, "pmut")
    s_self, _ = wslope(x, x, s)
    s_const, _ = wslope(np.ones_like(x), y, s)
    ok["V4_self_slope_is_one"] = (s_self is not None and abs(s_self - 1.0) < 1e-12)
    ok["V4_constant_x_is_none"] = (s_const is None)
    log(f"  V4 自身斜率 {s_self}  常数列 {s_const}  -> "
        f"{'通过' if ok['V4_self_slope_is_one'] and ok['V4_constant_x_is_none'] else '**失败**'}")

    # V2 equivalent form: demeaned ratio vs least squares with stratum dummies
    v2 = []
    for pk, nm in MODELS:
        for tag, rs in (("clean", clean), ("exposed", dirty)):
            x, y, s = arrs(rs, pk)
            a = wslope(x, y, s)[0]; b = wslope_fwl(x, y, s)
            v2.append(abs(a - b))
            log(f"  V2 {nm}/{tag:8s} demeaned {a:+.9f}  FWL {b:+.9f}  |Δ| {abs(a-b):.2e}")
    ok["V2_equivalent_form"] = max(v2) < 1e-9
    OUT["V2_max_abs_gap"] = float(max(v2))

    # V3 reverse control: the four arm slopes must reproduce M39's published values
    mine = {}
    for pk, nm in MODELS:
        for tag, rs in (("clean", clean), ("exposed", dirty)):
            x, y, s = arrs(rs, pk)
            mine[f"{nm}_{tag}"] = wslope(x, y, s)[0]
    drift = {}
    for k, v in mine.items():
        pub = m39["E3_arms"][k]["slope"]
        drift[k] = abs(v - pub)
        log(f"  V3 {k:26s} 我 {v:+.5f}  M39 {pub:+.5f}  |Δ| {drift[k]:.2e}")
    ok["V3_reverse_control"] = max(drift.values()) < 1e-4
    OUT["V3_arm_slopes"] = {k: round(v, 5) for k, v in mine.items()}
    OUT["V3_max_drift_vs_m39"] = round(float(max(drift.values())), 8)

    if not ok["V3_reverse_control"]:
        log("\n  **K1 触发:复现不出 M39 的臂斜率 ⇒ 本次运行作废**")
        OUT["verdict"] = "VOID_K1_cannot_reproduce_m39_slopes"; OUT["V"] = ok; save(OUT); return 1
    if not (ok["V2_equivalent_form"] and ok["V4_self_slope_is_one"] and ok["V4_constant_x_is_none"]):
        log("\n  **K2 触发:等价形式或零点自检失败 ⇒ 作废**")
        OUT["verdict"] = "VOID_K2_verification_failed"; OUT["V"] = ok; save(OUT); return 1

    # K4: how many exposed cells actually carry within-stratum variance
    contrib = set()
    cnt = collections.Counter(r["_s"] for r in dirty)
    for r in dirty:
        if cnt[r["_s"]] >= 2:
            contrib.add(r["design"])
    OUT["exposed_cells_contributing"] = len(contrib)
    log(f"  K4 曝光设计单元中有层内方差的 {len(contrib)}/{len(exposed_cells)}")
    k4_ok = len(contrib) >= 15

    # ================================================================== A2/A3 one bootstrap
    # M39 drew the clean resample then the exposed resample, from a generator re-seeded per model,
    # so both of its models saw identical draws. Computing all four slopes inside one iteration
    # therefore reproduces M39's two intervals AND pairs the models, which is what the DiD needs.
    log("\n=== A2 反向对照 + A3 配对差中差(同一次重抽样)===")
    pzc = collections.defaultdict(list); pzd = collections.defaultdict(list)
    for r in clean:
        pzc[r["puzzle"]].append(r)
    for r in dirty:
        pzd[r["puzzle"]].append(r)
    pc, pd = sorted(pzc), sorted(pzd)
    g = np.random.default_rng(SEED + 3)
    drop = {nm: [] for _pk, nm in MODELS}
    did, did_alt, self_did = [], [], []
    exp_slope = {nm: [] for _pk, nm in MODELS}
    for _ in range(NBOOT):
        ia = g.integers(0, len(pc), len(pc)); ib = g.integers(0, len(pd), len(pd))
        A = [r for i in ia for r in pzc[pc[i]]]
        B = [r for i in ib for r in pzd[pd[i]]]
        sl = {}
        for pk, nm in MODELS:
            xa, ya, sa = arrs(A, pk); xb, yb, sb = arrs(B, pk)
            sl[(nm, "clean")] = wslope(xa, ya, sa)[0]
            sl[(nm, "exposed")] = wslope(xb, yb, sb)[0]
        if any(v is None for v in sl.values()):
            continue
        for _pk, nm in MODELS:
            drop[nm].append(sl[(nm, "clean")] - sl[(nm, "exposed")])
            exp_slope[nm].append(sl[(nm, "exposed")])
        d1 = (sl[("ViennaRNA", "clean")] - sl[("ViennaRNA", "exposed")]) \
             - (sl[("RibonanzaNet", "clean")] - sl[("RibonanzaNet", "exposed")])
        d2 = (sl[("RibonanzaNet", "exposed")] - sl[("ViennaRNA", "exposed")]) \
             - (sl[("RibonanzaNet", "clean")] - sl[("ViennaRNA", "clean")])
        did.append(d1); did_alt.append(d2)
        self_did.append((sl[("ViennaRNA", "clean")] - sl[("ViennaRNA", "exposed")])
                        - (sl[("ViennaRNA", "clean")] - sl[("ViennaRNA", "exposed")]))

    # V1 zero point and V2's second half, checked on every draw that was kept
    gap = float(np.max(np.abs(np.asarray(did) - np.asarray(did_alt)))) if did else 1.0
    ok["V1_self_did_exactly_zero"] = bool(did and np.max(np.abs(self_did)) == 0.0)
    ok["V2_did_two_routes_agree"] = gap < 1e-12
    log(f"  V1 模型对自己的差中差 max|.| = {np.max(np.abs(self_did)) if did else 'n/a'}")
    log(f"  V2 两条路线的差中差 max|Δ| = {gap:.2e}  ({len(did)} 次重抽样)")
    OUT["V1_self_did_max_abs"] = float(np.max(np.abs(self_did))) if did else None
    OUT["V2_did_route_max_gap"] = gap
    if not (ok["V1_self_did_exactly_zero"] and ok["V2_did_two_routes_agree"]):
        log("\n  **K2 触发 ⇒ 作废**")
        OUT["verdict"] = "VOID_K2_did_identity_failed"; OUT["V"] = ok; save(OUT); return 1

    A2 = {}
    for _pk, nm in MODELS:
        ci = pct(drop[nm])
        pub = m39["E3_difference_ci"][nm]["clean_minus_exposed_ci"]
        dd = max(abs(ci[0] - pub[0]), abs(ci[1] - pub[1]))
        A2[nm] = {"mine": ci, "m39": pub, "max_abs_drift": round(dd, 6)}
        log(f"  A2 {nm:14s} 我 {ci}  M39 {pub}  |Δ|max {dd:.2e}")
    OUT["A2_reverse_control_intervals"] = A2
    ok["V3_reverse_control_intervals"] = max(v["max_abs_drift"] for v in A2.values()) < 1e-4
    if not ok["V3_reverse_control_intervals"]:
        log("\n  **K1 触发:复现不出 M39 的区间 ⇒ 作废**")
        OUT["verdict"] = "VOID_K1_cannot_reproduce_m39_intervals"; OUT["V"] = ok; save(OUT); return 1
    log("  V 全部通过 —— 现在才读 arm")

    pt_did = (mine["ViennaRNA_clean"] - mine["ViennaRNA_exposed"]) \
             - (mine["RibonanzaNet_clean"] - mine["RibonanzaNet_exposed"])
    ci_did = pct(did)
    OUT["A3_did"] = {
        "definition": "(ViennaRNA clean-exposed) - (RibonanzaNet clean-exposed), absolute units",
        "point": round(pt_did, 5), "ci95": ci_did, "n_boot": len(did),
        "ci_contains_zero": bool(ci_did[0] <= 0 <= ci_did[1]),
        "vienna_drop": round(mine["ViennaRNA_clean"] - mine["ViennaRNA_exposed"], 5),
        "neural_drop": round(mine["RibonanzaNet_clean"] - mine["RibonanzaNet_exposed"], 5)}
    log(f"\n  **A3 差中差 {pt_did:+.5f}  95% {ci_did}  "
        f"{'含 0' if OUT['A3_did']['ci_contains_zero'] else '不含 0'}**")

    # ------------------------------------------------------------------ A5 exposed-arm slopes
    A5 = {}
    for _pk, nm in MODELS:
        ci = pct(exp_slope[nm])
        A5[nm] = {"slope": round(mine[f"{nm}_exposed"], 5), "ci95": ci,
                  "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])}
        log(f"  A5 {nm:14s} 曝光子集斜率 {mine[f'{nm}_exposed']:+.5f}  95% {ci}"
            f"{'  含 0' if A5[nm]['ci_contains_zero'] else '  不含 0'}")
    OUT["A5_exposed_arm"] = A5

    # ------------------------------------------------------------------ A4 permutation null
    log("\n=== A4 曝光标签置换(设计单元层面,计数配平)===")
    cells = sorted(ident)
    gp = np.random.default_rng(SEED + 49)
    by_cell = collections.defaultdict(list)
    for r in recs:
        by_cell[r["design"]].append(r)
    null = []
    NPERM = 2000
    for _ in range(NPERM):
        pick = set(map(tuple, np.array(cells, dtype=object)[
            gp.choice(len(cells), len(exposed_cells), replace=False)]))
        A = [r for c in cells if c not in pick for r in by_cell[c]]
        B = [r for c in pick for r in by_cell[c]]
        if len(B) < 50:
            continue
        sl = {}
        bad = False
        for pk, nm in MODELS:
            xa, ya, sa = arrs(A, pk); xb, yb, sb = arrs(B, pk)
            a1 = wslope(xa, ya, sa)[0]; b1 = wslope(xb, yb, sb)[0]
            if a1 is None or b1 is None:
                bad = True; break
            sl[(nm, "clean")] = a1; sl[(nm, "exposed")] = b1
        if bad:
            continue
        null.append((sl[("ViennaRNA", "clean")] - sl[("ViennaRNA", "exposed")])
                    - (sl[("RibonanzaNet", "clean")] - sl[("RibonanzaNet", "exposed")]))
    null = np.asarray(null)
    centre = float(np.median(null)); mean = float(null.mean())
    p_one = float((1 + int((null >= pt_did).sum())) / (1 + len(null)))
    OUT["A4_permutation"] = {"n_perm": int(len(null)), "null_median": round(centre, 5),
                             "null_mean": round(mean, 5), "null_sd": round(float(null.std()), 5),
                             "null_p95": round(float(np.quantile(null, 0.95)), 5),
                             "observed": round(pt_did, 5), "p_one_sided": round(p_one, 5),
                             "centred_within_0.010": bool(abs(centre) <= 0.010)}
    log(f"  置换零分布 中位 {centre:+.5f}  均值 {mean:+.5f}  sd {null.std():.5f}  "
        f"95% 分位 {np.quantile(null,0.95):+.5f}   ({len(null)} 次)")
    log(f"  观测 {pt_did:+.5f}  单侧 p = {p_one:.5f}"
        f"{'   **K3:零分布未居中,A4 记为失败**' if abs(centre) > 0.010 else ''}")

    # ------------------------------------------------------------------ A6 reach ratios
    log("\n=== A6 触及比(C8 的判据,对两条通道用同一算法)===")
    c7_reach = abs(m39["E3_difference_ci"]["RibonanzaNet"]["clean_minus_exposed_ci"][1]) \
        / abs(mine["RibonanzaNet_clean"])
    c8_bd, c8_q = 0.2305, 0.3024          # xnBdV, xnRhoMaxV, as printed in the manuscript
    A6 = {"C7": {"interval_reach": m39["E3_difference_ci"]["RibonanzaNet"]["clean_minus_exposed_ci"][1],
                 "quantity_measured": round(mine["RibonanzaNet_clean"], 5),
                 "ratio": round(c7_reach, 4)},
          "C8": {"interval_reach": c8_bd, "quantity_measured": c8_q,
                 "ratio": round(c8_bd / c8_q, 4)}}
    OUT["A6_reach"] = A6
    log(f"  C7  {A6['C7']['interval_reach']} / {A6['C7']['quantity_measured']} = {A6['C7']['ratio']:.3f}")
    log(f"  C8  {A6['C8']['interval_reach']} / {A6['C8']['quantity_measured']} = {A6['C8']['ratio']:.3f}")

    # ================================================================== scorecard
    log("\n=== 预注册记分 ===")
    st = {}
    st["S1"] = {"text": D["staked_predictions"][0]["text"],
                "pass": not OUT["A3_did"]["ci_contains_zero"],
                "detail": f"差中差 {pt_did:+.5f} {ci_did}"}
    st["S2"] = {"text": D["staked_predictions"][1]["text"],
                "pass": bool(abs(centre) <= 0.010 and pt_did > np.quantile(null, 0.95)),
                "detail": f"零分布中位 {centre:+.5f},观测 {pt_did:+.5f} vs 95% 分位 "
                          f"{np.quantile(null,0.95):+.5f},p={p_one:.5f}"}
    st["S3"] = {"text": D["staked_predictions"][2]["text"],
                "pass": not A5["RibonanzaNet"]["ci_contains_zero"],
                "detail": f"神经曝光斜率 {A5['RibonanzaNet']['slope']} {A5['RibonanzaNet']['ci95']}"}
    st["S4"] = {"text": D["staked_predictions"][3]["text"],
                "pass": bool(c7_reach >= 0.5),
                "detail": f"C7 触及比 {c7_reach:.3f}(C8 自己 {c8_bd/c8_q:.3f})"}
    for k, v in st.items():
        log(f"  {k}  {'成立' if v['pass'] else '**失败**'}   {v['detail']}")
    OUT["staked"] = st
    OUT["V"] = ok
    OUT["K4_exposed_cells_contributing_ok"] = bool(k4_ok)
    OUT["verdict"] = ("did_resolves" if st["S1"]["pass"] else "did_unresolved") + \
                     ("_c7_must_be_reclassified" if st["S4"]["pass"] else "_c7_stands")
    log(f"\n  裁定:{OUT['verdict']}   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
