#!/usr/bin/env python3
"""M54: build the probing-derived reference, because the released one cannot be parsed.

Declaration m54_probing_reference_declaration.json, frozen and pushed BEFORE this file existed.
Offline, $0.

M53 voided at K2: the released M2_structure is not a balanced dot-bracket string on 74 of 160
wild types, 0 of which reproduce the file's own M2_F1 column, while the same parser reproduces its
RNet_F1 column to 0.00000 on all 160. The well-formed subset is not a fallback -- the malformed
rows are the more pseudoknotted ones.

So the reference is built here: ShapeKnots on the full construct with the deposit's own 2A3
reactivity as SHAPE pseudoenergy restraints, published defaults, no tuning. It shares its energy
model with two panel members; A4 measures that and K4 binds the manuscript. It is not RibonanzaNet
under any reading, which is the charge actually made.

The parsers, F1, estimator, rank agreement and key-set reconstruction are M53's, written there and
verified HERE by V3 -- M53 stopped before its own V3 could run.
"""
from __future__ import annotations

import collections, csv, json, os, subprocess, sys, tempfile, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m53_probing_reference as M53
from m53_probing_reference import (pairs_of, f1, depth_of, matched, within_sd, wrho, pct,
                                   wild_types, rnet_structures, model_structures, m25_keyset,
                                   SF, IMPL, PUB_M25, PUB_M36, PUB_AGREE, PUB_RANK, PUB_DEPTH,
                                   PROBE, NMIN)

DECL = os.path.join(HERE, "m54_probing_reference_declaration.json")
OUTJ = os.path.join(HERE, "m54_probing_reference.json")
RCACHE = os.path.join(HERE, "m54_shapeknots_cache.json")
D = json.load(open(DECL))
SEED, NBOOT, MAXPROC = 20260901, 2000, 32
RS_BIN = "/home/ubuntu/miniconda3/envs/rnastr/bin/ShapeKnots"
RS_DATA = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def _plain(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(repr(o))


def save(OUT):
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=_plain)
    log(f"\n写出 {OUTJ}")


# --------------------------------------------------------------- the reference, built here
def _first_ct(path):
    """the LOWEST-FREE-ENERGY structure only. ShapeKnots writes a set; a later one is not it."""
    out = set()
    with open(path) as f:
        head = f.readline()
        if not head.strip():
            return out
        n = int(head.split()[0])
        for _ in range(n):
            p = f.readline().split()
            if len(p) < 5:
                return out
            i, j = int(p[0]), int(p[4])
            if j and i < j:
                out.add((i - 1, j - 1))
    return out


def _fold_one(args):
    """(key, sequence, restraints or None) -> (key, pair set, ok). Restraints None = no data."""
    k, seq, react = args
    env = dict(os.environ, DATAPATH=RS_DATA)
    with tempfile.TemporaryDirectory() as d:
        sf = os.path.join(d, "s.seq"); cf = os.path.join(d, "s.ct")
        open(sf, "w").write(";\ns\n" + seq + "1\n")
        cmd = [RS_BIN, sf, cf]
        if react is not None:
            hf = os.path.join(d, "s.shape")
            with open(hf, "w") as f:
                for i in range(len(seq)):
                    x = react[i] if i < len(react) and react[i] is not None else None
                    f.write(f"{i+1}\t{-999.0 if x is None else x}\n")
            cmd += ["-sh", hf]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.exists(cf):
            return k, set(), False
        return k, _first_ct(cf), True


def reactivities(wt):
    """(puzzle, method) -> the 2A3 reactivity vector, index i aligned to sequence position i."""
    csv.field_size_limit(10 ** 9)
    rd = csv.DictReader(open(PROBE))
    rc = sorted(c for c in rd.fieldnames
                if c.startswith("reactivity_") and not c.startswith("reactivity_error"))
    out = {}
    for d in rd:
        if d.get("mutA") not in ("", "NA", "nan", None):
            continue
        k = (d["puzzle"], d["method"])
        if k in out or k not in wt:
            continue
        out[k] = [None if d[c] in ("", "NA", "nan") else float(d[c]) for c in rc]
    return out


def build_reference(wt, react, tag, use_data=True):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(k, wt[k]["sequence"], react[k] if use_data else None)
            for k in sorted(react) if k in wt]
    t0 = time.time(); got, fail = {}, []
    with ProcessPoolExecutor(max_workers=MAXPROC) as ex:
        for n, (k, prs, ok) in enumerate(ex.map(_fold_one, jobs, chunksize=1)):
            if ok:
                got[k] = prs
            else:
                fail.append(k)
            if (n + 1) % 40 == 0:
                log(f"    {tag} {n+1}/{len(jobs)}  {time.time()-t0:.0f}s")
    log(f"    {tag} 完成 {len(got)}/{len(jobs)},失败 {len(fail)},用时 {time.time()-t0:.0f}s")
    return got, fail


def main():
    t0 = time.time()
    log("=== M54:自己造探测衍生参考,因为发布的那一列解析不了 ===")
    log("  声明已冻结并推送于本文件存在之前。全离线,$0。")
    OUT = {"declaration": os.path.basename(DECL),
           "cost": "offline, $0 -- no API, no GPU, no download.",
           "why_m53_voided": "released M2_structure is unbalanced on 74/160 rows; see m53"}
    wt = wild_types()
    rnet, rnet_f1 = rnet_structures(wt)
    vi, rs, ef = model_structures(wt)
    react = reactivities(wt)
    log(f"  野生型 {len(wt)}  有 2A3 反应性 {len(react)}  ({time.time()-t0:.0f}s)")

    if os.path.exists(RCACHE):
        c = json.load(open(RCACHE))
        ref = {tuple(json.loads(k)): {tuple(p) for p in v} for k, v in c["with_data"].items()}
        nod = {tuple(json.loads(k)): {tuple(p) for p in v} for k, v in c["no_data"].items()}
        fail = [tuple(x) for x in c["failed"]]
        log(f"  ShapeKnots 缓存命中({len(ref)} 条 + {len(nod)} 条无数据对照)")
    else:
        log("  ShapeKnots(2A3 约束,默认斜率 1.8 / 截距 -0.6,不调参)")
        ref, fail = build_reference(wt, react, "ShapeKnots+2A3", True)
        log("  ShapeKnots(无数据对照,V6 用)")
        nod, _f2 = build_reference(wt, react, "ShapeKnots 无数据", False)
        json.dump({"with_data": {json.dumps(list(k)): [list(p) for p in v] for k, v in ref.items()},
                   "no_data": {json.dumps(list(k)): [list(p) for p in v] for k, v in nod.items()},
                   "failed": [list(k) for k in fail]}, open(RCACHE, "w"))
    if len(fail) > 10:
        OUT["VOID"] = f"K3: ShapeKnots 在 {len(fail)} 条上失败"; save(OUT); return 1

    def struct_for(nm, k):
        s = wt[k]["sequence"]
        return {"ViennaRNA": vi.get(s), "RNAstructure": rs.get(s),
                "EternaFold": ef.get(s), "RibonanzaNet": rnet.get(k)}[nm]

    def win(k, prs):
        d = wt[k]; ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        return {(i, j) for i, j in prs if ss - 1 <= i < se and ss - 1 <= j < se}

    def windb(k, db):
        return win(k, pairs_of(db))

    # ---------------------------------------------------------- V, before any arm
    log("\n=== V 验证(先于任何 arm)===")
    _p = next(iter(ref.values()))
    v1 = (f1(_p, _p) == 1.0 and f1(_p, set()) == 0.0)
    log(f"  V1 自身 F1 {f1(_p,_p):.1f}  对空 {f1(_p,set()):.1f}  -> {'通过' if v1 else '失败'}")

    # V2 the built reference is well formed: one partner per base, inside the construct, canonical
    CAN = {"AU", "UA", "GC", "CG", "GU", "UG"}
    bad_len = bad_deg = bad_can = 0
    for k, prs in ref.items():
        s = wt[k]["sequence"]; seen = collections.Counter()
        for i, j in prs:
            if not (0 <= i < j < len(s)):
                bad_len += 1
            seen[i] += 1; seen[j] += 1
            if s[i] + s[j] not in CAN:
                bad_can += 1
        if seen and max(seen.values()) > 1:
            bad_deg += 1
    log(f"  V2 越界配对 {bad_len}  一个碱基配两次的分子 {bad_deg}  非典型配对 {bad_can}")
    v2 = (bad_len == 0 and bad_deg == 0 and bad_can == 0)

    # V4 the shipped column still reproduces, so coordinates have not drifted since M53
    dv = max(abs(f1(windb(k, db), windb(k, wt[k]["target_structure"])) - rnet_f1[k])
             for k, db in rnet.items() if k in rnet_f1)
    log(f"  V4 RNet_F1 列复现 最大偏差 {dv:.5f}")
    v4 = dv < 1e-3

    # V6 the restraints reach the program
    diff = sum(1 for k in ref if k in nod and ref[k] != nod[k])
    log(f"  V6 加约束与不加约束给出不同结构的分子 {diff}/{len(ref)}")
    v6 = diff > len(ref) / 2
    OUT["V"] = {"V1_zero_point": bool(v1), "V2_well_formed": bool(v2),
                "V4_rnet_column": bool(v4), "V6_restraints_bind": bool(v6),
                "V6_n_changed": diff, "n_reference": len(ref), "n_failed": len(fail)}
    if not (v1 and v2 and v4 and v6):
        OUT["VOID"] = "K2: V1/V2/V4/V6 未通过"; save(OUT); return 1

    # V3a m43's agreement scores against the DECLARED target
    resc, called, keys, ndes = m25_keyset(wt, rnet)
    # The agreement axis and Section 5.1's readout are defined on DIFFERENT sets, and conflating
    # them is a real error rather than a rounding one. m43 scores agreement on M26's admitted
    # design cells (158). m25's key set is the pairs both readouts can score, whose designs are
    # 157 -- one cell short. The first run of this file used the 157 for the reverse control and
    # missed m43's four published scores by ~9e-4 against a 5e-4 tolerance, which is K1 firing on
    # a wrong cell set, not on a wrong F1. Agreement uses M26's cells; A2 uses m25's key set.
    cells = sorted({tuple(r["design"]) for r in
                    json.load(open(os.path.join(HERE, "m50_recs_cache.json")))["recs"]})
    cells = [k for k in cells if k in wt]
    log(f"  一致性轴用 M26 纳入的设计格 {len(cells)};§5.1 的键集用 m25 的共同配对 {len(keys)}"
        f"(其设计格 {len({k for (k,_a,_b) in keys})})")

    def agree(ref_of):
        sc = collections.defaultdict(list); per = {}
        for k in cells:
            tgt = ref_of(k)
            if tgt is None:
                continue
            per[k] = {}
            for nm in IMPL:
                db = struct_for(nm, k)
                if db is None:
                    continue
                v = f1(windb(k, db), tgt); sc[nm].append(v); per[k][nm] = v
        return {nm: round(float(np.mean(v)), 5) for nm, v in sc.items() if v}, per

    ag_d, per_d = agree(lambda k: windb(k, wt[k]["target_structure"]))
    d3 = {nm: abs(ag_d[nm] - PUB_AGREE[nm]) for nm in PUB_AGREE if nm in ag_d}
    for nm in IMPL:
        log(f"  V3a {nm:14s} 我 {ag_d.get(nm)}  m43 {PUB_AGREE[nm]}  |Δ| {d3.get(nm, 9):.5f}")
    v3a = len(d3) == 4 and max(d3.values()) < 5e-4

    R1 = [{**resc[k]} for k in keys]
    R2 = [{**resc[k], "y": called[k]} for k in keys]
    r1, w1, s1 = matched(R1, "y", SF); r2, w2, s2 = matched(R2, "y", SF)
    log(f"  V3b 配对 {len(keys)}({PUB_M25['n_pairs']})  权重 {w1:.0f}({PUB_M25['weight']})  "
        f"层 {s1}({PUB_M25['strata']})  介入 {r1:+.5f}({PUB_M25['R1']:+.4f})  "
        f"一致 {r2:+.5f}({PUB_M25['R2']:+.4f})")
    v3b = (len(keys) == PUB_M25["n_pairs"] and s1 == PUB_M25["strata"]
           and abs(w1 - PUB_M25["weight"]) < 0.5 and (w1, s1) == (w2, s2)
           and abs(r1 - PUB_M25["R1"]) < 5e-5 and abs(r2 - PUB_M25["R2"]) < 5e-5)
    sd1 = within_sd(R1, "y", SF); sd2 = within_sd(R2, "y", SF)
    log(f"  V3c 层内 SD {sd1:.4f}({PUB_M36['sd_rescue']}) / {sd2:.4f}({PUB_M36['sd_rnet']})  "
        f"标准化 {r1/sd1:+.4f}({PUB_M36['intv_sd']}) / {r2/sd2:+.4f}({PUB_M36['conv_sd']})")
    v3c = (abs(sd1 - PUB_M36["sd_rescue"]) < 5e-4 and abs(sd2 - PUB_M36["sd_rnet"]) < 5e-4
           and abs(r1 / sd1 - PUB_M36["intv_sd"]) < 5e-4
           and abs(r2 / sd2 - PUB_M36["conv_sd"]) < 5e-4)
    import m25_estimator_benchmark as M25
    v5d = max(abs(matched(R, "y", SF)[0] - M25.E4_matched(R, "y")) for R in (R1, R2))
    log(f"  V5 本文件的估计量 vs m25 的 E4_matched  max|Δ| {v5d:.3e}")
    v5 = v5d < 1e-12
    OUT["V"].update({"V3a_agreement": bool(v3a), "V3b_m25": bool(v3b), "V3c_m36": bool(v3c),
                     "V5_equivalent_form": bool(v5)})
    if not (v3a and v3b and v3c and v5):
        OUT["VOID"] = "K1/K2: V3 或 V5 未通过"; save(OUT); return 1
    log("  V1-V6 全部通过 —— 现在才换参考")
    OUT["reverse_control"] = {"agreement_declared": ag_d, "n_pairs": len(keys),
                              "weight": round(w1, 1), "strata": s1,
                              "intervention": round(r1, 5), "agreement_rnet": round(r2, 5),
                              "sd_rescue": round(sd1, 5), "sd_rnet": round(sd2, 5)}

    # ---------------------------------------------------------- A4 is it independent of the panel
    log("\n=== A4 built 参考与面板四个实现的距离(ShapeKnots 与两个成员共用能量模型)===")
    a4 = {}
    for nm in IMPL:
        vs = [f1(windb(k, struct_for(nm, k)), win(k, ref[k]))
              for k in cells if k in ref and struct_for(nm, k)]
        a4[nm] = round(float(np.mean(vs)), 5)
        log(f"  {nm:14s} 与 built 参考的平均 F1 {a4[nm]:.5f}")
    a4max = max(a4, key=lambda n: a4[n])
    indep = a4[a4max] <= 0.90
    log(f"  最近的是 {a4max} {a4[a4max]:.5f}  -> "
        f"{'独立于面板' if indep else '**K4:与该成员过近,必须点名**'}")
    OUT["A4_independence"] = {"f1_vs_panel": a4, "closest": a4max, "max": a4[a4max],
                              "independent": bool(indep)}

    # ---------------------------------------------------------- A1 the ranking, swapped
    log("\n=== A1 一致性排名换成 built 参考 ===")
    ag_p, per_p = agree(lambda k: win(k, ref[k]) if k in ref else None)
    for nm in IMPL:
        log(f"  {nm:14s} 声明靶标 {ag_d[nm]:.5f}  ->  built 参考 {ag_p.get(nm, float('nan')):.5f}")
    m_d = ag_d["RibonanzaNet"] - ag_d["ViennaRNA"]
    m_p = ag_p["RibonanzaNet"] - ag_p["ViennaRNA"]
    ord_d = sorted(IMPL, key=lambda n: -ag_d[n]); ord_p = sorted(IMPL, key=lambda n: -ag_p[n])
    log(f"  incumbent margin  声明 {m_d:+.5f}  ->  built {m_p:+.5f}")
    log(f"  排序  声明 {ord_d}")
    log(f"        built {ord_p}")
    pzs = sorted({wt[k]["puzzle"] for k in cells})
    bypz = collections.defaultdict(list)
    for k in cells:
        bypz[wt[k]["puzzle"]].append(k)
    rng = np.random.default_rng(SEED)
    md, mp, dm = [], [], []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(pzs), len(pzs))
        ks = [k for q in ix for k in bypz[pzs[q]]]
        def mn(per, nm):
            v = [per[k][nm] for k in ks if k in per and nm in per[k]]
            return float(np.mean(v)) if v else None
        a, b = mn(per_d, "RibonanzaNet"), mn(per_d, "ViennaRNA")
        c, e = mn(per_p, "RibonanzaNet"), mn(per_p, "ViennaRNA")
        if None in (a, b, c, e):
            continue
        md.append(a - b); mp.append(c - e); dm.append((c - e) - (a - b))
    log(f"  margin 声明 {m_d:+.5f} {pct(md)}   built {m_p:+.5f} {pct(mp)}")
    log(f"  之差(built − 声明) {m_p-m_d:+.5f} {pct(dm)}  "
        f"{'不含 0' if (pct(dm)[0] > 0 or pct(dm)[1] < 0) else '含 0'}")
    OUT["A1_ranking"] = {"agreement_declared": ag_d, "agreement_built": ag_p,
                         "order_declared": ord_d, "order_built": ord_p,
                         "margin_declared": round(m_d, 5), "margin_declared_ci": pct(md),
                         "margin_built": round(m_p, 5), "margin_built_ci": pct(mp),
                         "margin_change": round(m_p - m_d, 5), "margin_change_ci": pct(dm),
                         "n_cells": len(cells)}

    # ---------------------------------------------------------- A2 section 5.1, column swapped
    log("\n=== A2 §5.1 的一致性列换掉:五个列,同一批配对、同一套分层 ===")
    COLS = [("RibonanzaNet", "shipped structure frequency (the published column)"),
            ("ViennaRNA", "its own MFE structure"),
            ("RNAstructure", "its own MFE structure"),
            ("EternaFold", "its own MEA structure"),
            ("probing", "the built reference -- not a sequence-to-structure model")]
    prs_cache = {nm: {k: pairs_of(struct_for(nm, k)) if struct_for(nm, k) else None for k in cells}
                 for nm in ("ViennaRNA", "RNAstructure", "EternaFold")}
    recs = []
    for (k, a, b) in keys:
        r = dict(resc[(k, a, b)]); r["rf"] = r["y"]
        r["agr_RibonanzaNet"] = called[(k, a, b)]
        for nm in ("ViennaRNA", "RNAstructure", "EternaFold"):
            p = prs_cache[nm].get(k)
            r["agr_" + nm] = (1.0 if (a, b) in p else 0.0) if p is not None else None
        r["agr_probing"] = (1.0 if (a, b) in ref[k] else 0.0) if k in ref else None
        recs.append(r)
    ok = [r for r in recs if all(r["agr_" + nm] is not None for nm, _d in COLS)]
    log(f"  五列齐备的配对 {len(ok)}/{len(recs)}")
    a2 = {}
    for nm, desc in COLS:
        c, _w, _u = matched(ok, "agr_" + nm, SF); sd = within_sd(ok, "agr_" + nm, SF)
        a2[nm] = {"desc": desc, "contrast": round(c, 5), "within_sd": round(sd, 5),
                  "contrast_in_sd": round(c / sd, 5) if sd else None,
                  "hit_rate": round(float(np.mean([r["agr_" + nm] for r in ok])), 4)}
        log(f"  {nm:14s} 对比 {c:+.5f}  层内 SD {sd:.4f}  标准化 {c/sd:+.5f}  "
            f"命中率 {a2[nm]['hit_rate']:.4f}")
    ri = matched(ok, "rf", SF)[0]; sdi = within_sd(ok, "rf", SF)
    log(f"  {'intervention':14s} 对比 {ri:+.5f}  层内 SD {sdi:.4f}  标准化 {ri/sdi:+.5f}")
    byd = collections.defaultdict(list)
    for r in ok:
        byd[r["design"]].append(r)
    des = sorted(byd); rng2 = np.random.default_rng(SEED + 1)
    draws = {nm: [] for nm, _d in COLS}
    for _ in range(NBOOT):
        ix = rng2.integers(0, len(des), len(des))
        srs = [x for q in ix for x in byd[des[q]]]
        cy, _w, _u = matched(srs, "rf", SF); sy = within_sd(srs, "rf", SF)
        if cy is None or not sy:
            continue
        for nm, _d in COLS:
            cx, _w2, _u2 = matched(srs, "agr_" + nm, SF); sx = within_sd(srs, "agr_" + nm, SF)
            draws[nm].append((cy / sy - cx / sx) if (cx is not None and sx) else np.nan)
    log("\n  介入读数 − 一致读数(各自单位的 SD,配对同一次抽样):")
    for nm, _d in COLS:
        v = np.asarray(draws[nm], float); v = v[~np.isnan(v)]
        ci = pct(v); pt_ = ri / sdi - a2[nm]["contrast_in_sd"]
        a2[nm].update({"paired_diff_in_sd": round(pt_, 5), "ci95": ci,
                       "ci_contains_zero": bool(ci[0] <= 0 <= ci[1])})
        log(f"    vs {nm:14s} {pt_:+.5f}  {ci}  "
            f"{'含 0' if a2[nm]['ci_contains_zero'] else '不含 0'}")
    OUT["A2_readout"] = {"n_pairs": len(ok), "intervention_in_sd": round(ri / sdi, 5),
                         "columns": a2}

    # ---------------------------------------------------------- A3 depth, from which structure
    log("\n=== A3 helix depth 从 built 参考读 vs 从声明靶标读 ===")
    c50 = json.load(open(os.path.join(HERE, "m50_recs_cache.json")))["recs"]
    c52 = json.load(open(os.path.join(HERE, "m52_recs_cache.json")))["recs"]
    assert len(c50) == len(c52), "两个缓存长度不同"
    for x, y in zip(c50, c52):
        assert (tuple(x["design"]) == tuple(y["design"]) and x["pt"] == y["pt"]
                and x["depth"] == y["depth"] and abs(x["rf"] - y["rf"]) < 1e-12), "索引对齐失败"
    log(f"  两个记录缓存逐行一致({len(c50)} 条),按索引合并")
    R = []
    for x, y in zip(c50, c52):
        k = tuple(x["design"])
        if k not in ref:
            continue
        a, b = int(y["a"]), int(y["b"])
        P = {}
        for i, j in ref[k]:
            P[i] = j; P[j] = i
        dp, present = depth_of(P, a, b)
        R.append({"design": k, "puzzle": x["puzzle"], "rf": x["rf"], "nmut": x["nmut"],
                  "pmut": x["pmut"], "emut": x["emut"], "depth": x["depth"],
                  "depth_probe": float(dp), "present": present})
    frac = float(np.mean([r["present"] for r in R]))
    log(f"  干预 {len(R)};该配对在 built 参考里也成对的比例 {frac:.4f}")
    bc = collections.defaultdict(list)
    for r in R:
        bc[r["design"]].append(r)
    cl = [(k, v) for k, v in bc.items() if len(v) >= NMIN]
    log(f"  至少 {NMIN} 个翻转的设计格 {len(cl)},覆盖 {sum(len(v) for _k,v in cl)} 次干预")
    v3d = {n: wrho(cl, kx) for n, kx in
           (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut"))}
    dd_, dp_ = wrho(cl, "depth"), wrho(cl, "depth_probe")
    for n in PUB_RANK:
        log(f"  V3d {n:14s} 我 {v3d[n]:+.5f}  论文 {PUB_RANK[n]:+.4f}")
    log(f"  V3d depth(声明)  我 {dd_:+.5f}  论文 {PUB_DEPTH:+.4f}")
    v3ok = (all(abs(v3d[n] - PUB_RANK[n]) < 5e-4 for n in PUB_RANK)
            and abs(dd_ - PUB_DEPTH) < 5e-4)
    OUT["V"]["V3d_rank_reverse_control"] = bool(v3ok)
    if not v3ok:
        OUT["VOID"] = "K1: A3 的反向对照未复现"; save(OUT); return 1
    log(f"  depth(built 参考)  {dp_:+.5f}")
    bp = collections.defaultdict(list)
    for k, v in cl:
        bp[v[0]["puzzle"]].append((k, v))
    pz3 = sorted(bp); rng3 = np.random.default_rng(SEED + 2)
    ddr, lvl = [], {n: [] for n in PUB_RANK}
    for _ in range(NBOOT):
        ix = rng3.integers(0, len(pz3), len(pz3))
        cs = [c for q in ix for c in bp[pz3[q]]]
        p_, l_ = wrho(cs, "depth_probe"), wrho(cs, "depth")
        if p_ is None or l_ is None:
            continue
        ddr.append(p_ - l_)
        for n, kx in (("RibonanzaNet", "nmut"), ("ViennaRNA", "pmut"), ("EternaFold", "emut")):
            w = wrho(cs, kx); lvl[n].append(p_ - w if w is not None else np.nan)
    log(f"  built depth − 声明 depth  {dp_-dd_:+.5f}  {pct(ddr)}")
    a3 = {"frac_pair_present": round(frac, 4), "n_cells": len(cl),
          "depth_declared": round(dd_, 5), "depth_built": round(dp_, 5),
          "difference": round(dp_ - dd_, 5), "difference_ci": pct(ddr), "vs_level_arms": {}}
    log("  built depth − 各预测器的 level 臂:")
    for n in PUB_RANK:
        v = np.asarray(lvl[n], float); v = v[~np.isnan(v)]
        ci = pct(v); p2 = dp_ - v3d[n]
        a3["vs_level_arms"][n] = {"point": round(p2, 5), "ci95": ci,
                                  "ci_contains_zero": bool(ci[0] <= 0 <= ci[1]),
                                  "built_depth_wins": bool(p2 > 0)}
        log(f"    vs {n:14s} {p2:+.5f}  {ci}  {'含 0' if ci[0] <= 0 <= ci[1] else '不含 0'}")
    OUT["A3_depth"] = a3

    # ---------------------------------------------------------- A5 the released column, selected
    log("\n=== A5 次要且被选择过:发布列里配平的那 86 行 ===")
    probe_rel, rel_f1 = M53.probing_structures(wt)
    PAIRC = {")": "(", "]": "[", "}": "{", ">": "<"}
    csv.field_size_limit(10 ** 9)
    raw = {}
    for d in csv.DictReader(open(PROBE)):
        if d.get("mutA") in ("", "NA", "nan", None):
            raw.setdefault((d["puzzle"], d["method"]), d)
    good = [k for k in cells if k in probe_rel and k in raw
            and all(collections.Counter(raw[k]["M2_structure"])[o]
                    == collections.Counter(raw[k]["M2_structure"])[c] for c, o in PAIRC.items())]
    ag_r, _pr = agree(lambda k: windb(k, probe_rel[k]) if k in good else None)
    log(f"  可用行 {len(good)}/{len(cells)}")
    for nm in IMPL:
        log(f"  {nm:14s} {ag_r.get(nm)}")
    m_r = ag_r["RibonanzaNet"] - ag_r["ViennaRNA"]
    def pk(db):
        return sum(1 for ch in db if ch in "[]{}<>") / max(len(db), 1)
    pkg = float(np.mean([pk(wt[k]["target_structure"]) for k in good]))
    pkb = float(np.mean([pk(wt[k]["target_structure"]) for k in cells if k not in good]))
    log(f"  margin {m_r:+.5f};保留行的靶标假结字符占比 {pkg:.4f},丢弃行 {pkb:.4f} —— "
        f"这是按结构类别做的选择,只能当有界的次要对照")
    OUT["A5_released_subset"] = {"n": len(good), "agreement": ag_r, "margin": round(m_r, 5),
                                 "order": sorted(IMPL, key=lambda n: -ag_r[n]),
                                 "pk_kept": round(pkg, 4), "pk_dropped": round(pkb, 4),
                                 "selected_on_structure_class": True}

    # ---------------------------------------------------------- prereg scoring
    S1 = bool(abs(m_p) < abs(m_d))
    S2 = bool(ord_p[0] == "RibonanzaNet")
    S3 = bool(not a2["probing"]["ci_contains_zero"] and a2["probing"]["paired_diff_in_sd"] > 0)
    S4 = bool(all(a3["vs_level_arms"][n]["built_depth_wins"] for n in PUB_RANK))
    log("\n=== 预注册记分 ===")
    det = [f"margin 声明 {m_d:+.5f} -> built {m_p:+.5f}", f"built 参考下的排序 {ord_p}",
           f"介入 − built 一致列 {a2['probing']['paired_diff_in_sd']:+.5f} {a2['probing']['ci95']}",
           "built depth 对三个 level 臂 " + ", ".join(
               f"{n} {a3['vs_level_arms'][n]['point']:+.4f}" for n in PUB_RANK)]
    for i, (sid, okk) in enumerate((("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4))):
        log(f"  {sid}  {'成立' if okk else '**失败**'}   {det[i]}")
    OUT["staked"] = {f"S{i+1}": {"text": D["staked_predictions"][i]["text"],
                                 "pass": p, "detail": det[i]}
                     for i, p in enumerate([S1, S2, S3, S4])}
    OUT["verdict"] = ("agreement_axis_survives_a_built_probing_reference" if (S2 and S3 and S4)
                      else "agreement_axis_depends_on_the_reference")
    log(f"\n  裁定:{OUT['verdict']}   {sum([S1,S2,S3,S4])}/4 成立   用时 {time.time()-t0:.0f}s")
    save(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
