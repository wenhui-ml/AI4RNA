#!/usr/bin/env python3
"""M26d: re-ask M26c's question on designs that are not in the training sources.

Declaration m26d_clean_subset_declaration.json, frozen and pushed at 066f4ba1 BEFORE this file
existed. Offline, $0 -- P_neural comes from M26c's cache, training sequences stream from RMDB.

M26c's K0 fired: 6.37% of OK7a design windows are >=80% identical to something in RibonanzaNet's
source libraries and 264 are verbatim. The declaration's remedy is not to void the result, it is to
ask again on a subset that can be described, and to run ViennaRNA alongside as the negative control
that training leakage cannot reach.

The verdict is F3, not F2. If the leaked designs show the same skill as the clean ones, then
memorisation is not what produced the skill, whatever the overlap fraction happens to be.
"""
from __future__ import annotations

import argparse, collections, csv, json, os, re, sys, time, urllib.request
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m26c_leakage as LEAK

CACHE = os.path.join(HERE, "m26c_pneural_cache.npz")
NPERM, NPLACEBO, THR = 2000, 12, 0.80
K = 12                    # smaller than the audit's 20 so most queries get a candidate at all
SAMPLE = 20000            # direct-scan confirmation for queries the index misses (P1)


def max_identity(queries, train, log):
    """max identity of each query against any equal-length window of any training sequence.

    P1: a query with no k-mer candidate is NOT identity zero. Those are confirmed by a direct scan
    against a random sample, and the count is reported rather than folded into the distribution.
    """
    tl = list(train)
    inv = collections.defaultdict(list)
    for i, s in enumerate(tl):
        for st in range(0, max(1, len(s) - K + 1)):
            inv[s[st:st + K]].append(i)
    log(f"  k={K} 索引 {len(inv)} 个 k-mer,训练序列 {len(tl)}")
    rng = np.random.default_rng(0)
    samp = [tl[i] for i in rng.choice(len(tl), min(SAMPLE, len(tl)), replace=False)]
    best, nocand, nocand_best = {}, 0, []

    def scan(w, pool):
        b = 0.0
        for t in pool:
            for st in range(0, max(1, len(t) - len(w) + 1)):
                seg = t[st:st + len(w)]
                if len(seg) != len(w):
                    continue
                v = sum(x == y for x, y in zip(w, seg)) / len(w)
                if v > b:
                    b = v
                    if b >= 0.999:
                        return b
        return b

    for n, (key, w) in enumerate(queries.items()):
        cand = collections.Counter()
        for st in range(0, max(1, len(w) - K + 1)):
            for i in inv.get(w[st:st + K], ()):
                cand[i] += 1
        if cand:
            best[key] = scan(w, [tl[i] for i, _c in cand.most_common(60)])
        else:
            nocand += 1
            v = scan(w, samp)                                  # P1: confirm, do not assume zero
            best[key] = v
            nocand_best.append(v)
        if (n + 1) % 200 == 0:
            log(f"    {n+1}/{len(queries)}")
    nb = np.array(nocand_best) if nocand_best else np.array([0.0])
    log(f"  无 k-mer 候选的查询 {nocand} 条 -> 直接扫描 {len(samp)} 条确认:"
        f"同一性 均值 {nb.mean():.4f} 中位 {np.median(nb):.4f} 最大 {nb.max():.4f}")
    return best, {"n_no_candidate": nocand, "no_candidate_identity_mean": round(float(nb.mean()), 4),
                  "no_candidate_identity_median": round(float(np.median(nb)), 4),
                  "no_candidate_identity_max": round(float(nb.max()), 4)}


def arm(recs, tag, pkey, log, seed=0):
    """M26c's N1/N2 on one record set, using M26's machinery verbatim (P2)."""
    import random as _r
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    if len(recs) < 50:
        log(f"  {tag}: n={len(recs)} 太少,跳过"); return None
    s, n = M26.slope_in(recs, sb, pkey)
    if s is None:
        log(f"  {tag}: 无层内方差"); return None
    p = M26.slope_perm_p(recs, sb, pkey, s, 900 + abs(hash(tag + pkey)) % 997, nperm=400)
    v = np.array([r[pkey] for r in recs])
    edges = [float(np.quantile(v, q)) for q in np.linspace(0, 1, 6)[1:-1]]
    for r in recs:
        r["_b"] = int(np.searchsorted(edges, r[pkey], side="right"))
    sd = lambda r: (r["design"], min(r["depth"], 3), r["crossed"], r["_b"])
    cde, w1, _ = M26.matched(recs, "CG", "AU", sd)
    out = {"n": len(recs), "slope": round(s, 5), "slope_p": round(p, 5), "weight": (w1 or 0)}
    if cde is None or w1 < 100:
        log(f"  {tag:22s} 斜率 {s:+.5f} (p {p:.4f})  n={len(recs):5d}  "
            f"**K3 重叠不足(权重 {w1 or 0:.0f})⇒ 留存不报**")
        out["retention"] = None
        return out
    keep = M26.retained_recs(recs, "CG", "AU", sd)
    base, _wb, _sb2 = M26.matched(keep, "CG", "AU", sb)
    null = np.sort(np.array([M26.perm(recs, "CG", "AU", sd, _r.Random(seed + i)) or 0.0
                             for i in range(NPERM)]))
    pv = (float((null >= cde).sum()) + 1) / (NPERM + 1)
    ret = cde / base if base else None
    out.update({"cde": round(cde, 4), "baseline": round(base, 4),
                "retention": round(ret, 4) if ret else None, "p_perm": round(pv, 5)})
    log(f"  {tag:22s} 斜率 {s:+.5f} (p {p:.4f})  n={len(recs):5d}  "
        f"CDE {cde:+.4f}  基线 {base:+.4f}  **留存 {100*ret:.1f}%**  p {pv:.5f}  权重 {w1:.0f}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    LOG = []

    def log(s=""):
        print(s, flush=True); LOG.append(s)

    t0 = time.time()
    log("=== M26d:剔除泄漏设计后重问一次 ===")
    log("  声明 066f4ba1。离线 $0。\n")

    recs, _dropped = M26.build()
    z = np.load(CACHE, allow_pickle=True)
    pn = {k: v for k, v in zip(z["keys"], z["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    log(f"纳入 {len(recs)},P_neural 取自 M26c 缓存")

    log("\n=== 取训练源序列(与泄漏审计同一套)===")
    train = set()
    for rid in LEAK.TRAIN_DEPOSITS:
        try:
            u = LEAK.rdat_url(rid)
            if u:
                train.update(LEAK.stream_sequences(u, log))
        except Exception as e:                                                 # noqa: BLE001
            log(f"  {rid} 取用失败 {repr(e)[:60]}")
    log(f"  训练唯一序列 {len(train)}  ({time.time()-t0:.0f}s)")

    # ---------------- F1: which design cells are leaked
    log("\n=== F1 逐设计单元判泄漏(设计窗内同一性 >= 0.80)===")
    csv.field_size_limit(10 ** 9)
    win = {}
    for d in csv.DictReader(open(M26.SRC)):
        try:
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        except (ValueError, KeyError):
            continue
        win[(d["puzzle"], d["method"])] = (ss - 1, se)
    q = {}
    for r in recs:
        k = r["design"]
        if k in win and k not in q:
            lo, hi = win[k]
            q[k] = r["wt_seq"][lo:hi]
    log(f"  设计单元 {len(q)}")
    best, p1 = max_identity(q, train, log)
    leaked = {k for k, v in best.items() if v >= THR}
    clean = [r for r in recs if r["design"] not in leaked]
    dirty = [r for r in recs if r["design"] in leaked]
    log(f"  **泄漏设计单元 {len(leaked)}/{len(q)} = {100*len(leaked)/len(q):.1f}%**")
    log(f"  干净干预 {len(clean)}  泄漏干预 {len(dirty)}")
    log(f"  干净类型 {dict(collections.Counter(r['pt'] for r in clean))}")
    OUT = {"n_designs": len(q), "n_leaked_designs": len(leaked), "threshold": THR,
           "n_clean": len(clean), "n_dirty": len(dirty), "P1": p1,
           "identity_hist": {str(round(x, 1)): int(sum(1 for v in best.values()
                                                       if round(v, 1) == round(x, 1)))
                             for x in np.arange(0, 1.01, 0.1)}}

    # ---------------- F2 / F3, both models, all three record sets (P3, P4)
    log("\n=== F2/F3 全集 / 干净 / 泄漏,两个模型并列(P3:成对出现;P4:热力学阴性对照)===")
    res = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        log(f"\n  --- {mname} ---")
        for tag, rs in (("all", recs), ("clean", clean), ("leaked", dirty)):
            res[f"{mname}_{tag}"] = arm(rs, f"{mname}/{tag}", pkey, log, a.seed)
    OUT["arms"] = res

    # ---------------- verdict from F3
    log("\n=== F3 裁定依据:泄漏子集与干净子集的技能差 ===")
    stakes = {}
    for mname in ("RibonanzaNet", "ViennaRNA"):
        c = res.get(f"{mname}_clean"); d = res.get(f"{mname}_leaked")
        if c and d:
            diff = abs(c["slope"] - d["slope"])
            log(f"  {mname:14s} 干净 {c['slope']:+.5f}  泄漏 {d['slope']:+.5f}  "
                f"差 {diff:.5f}  {'**相当 ⇒ 记忆不是解释**' if diff < 0.05 else '**差距大 ⇒ 记忆可疑**'}")
            OUT[f"F3_{mname}_skill_gap"] = round(diff, 5)
            if mname == "RibonanzaNet":
                stakes["3_skill_gap_below_005"] = bool(diff < 0.05)
    rn_c = res.get("RibonanzaNet_clean") or {}
    vn_c = res.get("ViennaRNA_clean") or {}
    stakes["1_leaked_designs_le_10pct"] = bool(len(leaked) / max(len(q), 1) <= 0.10)
    stakes["2_clean_neural_retention_ge_50"] = bool((rn_c.get("retention") or 0) >= 0.50)
    stakes["4_clean_vienna_retention_ge_80"] = bool((vn_c.get("retention") or 0) >= 0.80)
    gap_ok = OUT.get("F3_RibonanzaNet_skill_gap")
    if (rn_c.get("retention") or 0) >= 0.50 and gap_ok is not None and gap_ok < 0.05:
        verdict = "residual_survives_leakage_removal"
    elif gap_ok is not None and gap_ok >= 0.05:
        verdict = "memorisation_explains_skill"
    else:
        verdict = "inconclusive_report_as_is"
    OUT["verdict"] = verdict
    OUT["stakes"] = stakes
    log(f"\n=== 裁定: {verdict} ===")
    log("押注逐条: " + json.dumps(stakes, ensure_ascii=False))
    log("  **界:训练源不是训练清单;『干净』只表示与我们能取到的来源不重合。**")
    json.dump(OUT, open(os.path.join(HERE, "m26d_clean_subset.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(HERE, "m26d_clean_subset.log"), "w").write("\n".join(LOG) + "\n")
    log(f"\n({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
