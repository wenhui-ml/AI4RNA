#!/usr/bin/env python3
"""M42: the leakage audit against the whole released training corpus.

Declaration `m42_full_corpus_declaration.json`, frozen and pushed at 3f5114c8 BEFORE this file
existed. Offline, $0 -- the 2.36 GB file is on disk, no API, no GPU.

Every helper is M39's, imported and not reimplemented, so the only difference between M39's
numbers and these is the input file. That is what makes the comparison attributable to the corpus
rather than to the code.

V3 is a kill and runs in the same pass: the same machinery, restricted back to QUICK_START's
167,808 sequences, must reproduce M39's exposure count of 20 of 158 exactly. Both banks are built
from one load, so the check costs a second scan and not a second run.
"""
from __future__ import annotations

import collections, csv, json, os, random, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m39_exposure_tiers as M39
import m26_model_scored as M26
import m26d_clean_subset as M26D

DECL = json.load(open(os.path.join(HERE, "m42_full_corpus_declaration.json")))
K = DECL["constants"]
THR, TIERS, SEED, NBOOT = K["THRESH"], K["TIERS"], K["SEED"], K["NBOOT"]
MIN_CLEAN = K["MIN_CLEAN_INTERVENTIONS"]
FULL = DECL["data"]["file"]
QUICK = "/data/tmp/ribonanza/train_data_QUICK_START.csv"
OUTJ = os.path.join(HERE, "m42_full_corpus.json")
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def load(path):
    csv.field_size_limit(10 ** 9)
    seq2lib = collections.defaultdict(set)
    n = 0
    for d in csv.DictReader(open(path)):
        n += 1
        lib = d["dataset_name"]
        for tag in ("_2A3", "_DMS"):
            lib = lib.replace(tag, "")
        seq2lib[d["sequence"]].add(lib.strip("_").replace("__", "_"))
    return seq2lib, n


def exposure(q, bank):
    best, who = {}, {}
    for k in sorted(q):
        best[k], who[k] = M39.max_identity(q[k], bank)
    return best, who


def main():
    t0 = time.time()
    log("=== M42:把 C7 的审计跑在完整的已发布训练语料上 ===")
    log("  声明冻结并推送于本文件存在之前。全离线,$0。")
    log("  措辞规则 K3:计数在这个语料上是精确的;**不许**写『不在训练集里』或『没有泄漏』,")
    log("  因为伪标签阶段仍未被任何臂审计。\n")
    OUT = {"declaration": os.path.basename("m42_full_corpus_declaration.json"),
           "constants": K, "code_path": "M39's helpers, unmodified; only the input file differs"}

    full, nrows = load(FULL)
    quick, _ = load(QUICK)
    fs, qs = sorted(full), sorted(quick)
    log(f"  完整语料:{nrows:,} 行,{len(fs):,} 条唯一序列,"
        f"{len({l for v in full.values() for l in v})} 个来源库  ({time.time()-t0:.0f}s)")
    log(f"  QUICK_START:{len(qs):,} 条唯一序列")
    missing = [s for s in qs if s not in full]
    log(f"  包含关系:QUICK_START 落在完整语料之外的序列 {len(missing)} 条 "
        f"-> {'嵌套,通过' if not missing else '**不嵌套,声明的前提不成立**'}")
    OUT["n_train_unique_full"] = len(fs)
    OUT["n_train_rows_full"] = nrows
    OUT["n_train_libraries_full"] = len({l for v in full.values() for l in v})
    OUT["n_train_unique_quick"] = len(qs)
    OUT["containment_ok"] = not missing
    OUT["m39_declaration_erratum"] = (
        "M39's declaration states the full training set is 'roughly two million sequences' and "
        f"rests its LOWER BOUND wording on that. The released competition file holds {len(fs)} "
        "unique sequences over " f"{nrows} rows. The figure in M39's declaration cannot be sourced "
        "to this file. M39's declaration is NOT edited. What the count is exact ON is the released "
        "labelled corpus; the pseudo-label pre-training stage remains unaudited, so the bound "
        "language is replaced by the narrower statement in K3, not removed.")

    bank_full = M39.build_bank(fs)
    bank_quick = M39.build_bank(qs)
    log(f"  两个 bank 建好(完整 {len(bank_full)} 组 / QUICK {len(bank_quick)} 组)")

    # ---- the queries, exactly M39's
    recs, _dr = M26.build()
    win = {}
    for d in csv.DictReader(open(M26.SRC)):
        try:
            ss, se = int(float(d["sub_start"])), int(float(d["sub_end"]))
        except (ValueError, KeyError):
            continue
        win[(d["puzzle"], d["method"])] = (ss - 1, se)
    zc = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {str(k): v for k, v in zip(zc["keys"], zc["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
    del pn, zc
    q = {}
    for r in recs:
        k = r["design"]
        if k in win and k not in q:
            lo, hi = win[k]
            q[k] = r["wt_seq"][lo:hi]
    log(f"  设计单元 {len(q)},窗长 {sorted({len(v) for v in q.values()})}")
    OUT["n_designs"] = len(q)

    # ---- V, before anything is read
    log("\n=== V 验证(先于任何结果)")
    ks = sorted(q)
    smp = [ks[i] for i in random.Random(SEED).sample(range(len(ks)), min(6, len(ks)))]
    self_ok = all(abs(M39.max_identity(q[k], M39.build_bank([q[k]]))[0] - 1.0) < 1e-12 for k in smp)
    log(f"  V1 零点恒等式(自己对自己 = 1.0){'通过' if self_ok else '**失败**'}")
    sh = []
    for k in ks:
        z = M39.dinuc_shuffle(q[k], random.Random(SEED + hash(k) % 9973))
        if z:
            sh.append(M39.max_identity(z, bank_full)[0])
    sh = np.asarray(sh)
    neg_ok = len(sh) and float(sh.max()) < THR
    log(f"  V2 阴性对照(双核苷酸保持打乱,{len(sh)} 条):最大 {sh.max():.4f} 中位 "
        f"{np.median(sh):.4f} < {THR}  {'通过' if neg_ok else '**失败**'}")

    bq, _wq = exposure(q, bank_quick)
    n_q = sum(1 for v in bq.values() if v >= THR)
    v3_ok = (n_q == 20)
    log(f"  V3 退回 QUICK_START 的曝光数 {n_q}/158,M39 发表 20/158  "
        f"{'通过' if v3_ok else '**失败 ⇒ 本次作废,报告为 M39 的复现失败**'}")
    OUT["V"] = {"self_identity": bool(self_ok), "shuffle_max": float(sh.max()) if len(sh) else None,
                "quick_restricted_exposed": n_q, "m39_published": 20, "V3_pass": bool(v3_ok),
                "pass": bool(self_ok and neg_ok and v3_ok)}
    if not OUT["V"]["pass"]:
        OUT["verdict"] = "void_verification_failed"
        OUT["log"] = LOG; json.dump(OUT, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 1

    # ---- E1/E2 on the full corpus
    log("\n=== E1/E2 完整语料上的曝光与分级")
    best, who = exposure(q, bank_full)
    exposed = {k for k, v in best.items() if v >= THR}
    tiers = collections.Counter(M39.tier_of(v) for v in best.values())
    names = {0: f"T0 <{TIERS[0]}", 1: f"T1 [{TIERS[0]},{TIERS[1]})",
             2: f"T2 [{TIERS[1]},{TIERS[2]})", 3: f"T3 >={TIERS[2]} 逐字"}
    log(f"  **曝光 {len(exposed)}/{len(q)} = {100*len(exposed)/len(q):.1f}%**  "
        f"(M39 子集 20/158 = 12.7%;M26d 代理 14/158 = 8.9%)")
    for t in range(4):
        log(f"    {names[t]:22s} {tiers.get(t,0):4d}")
    lib = collections.Counter()
    for k in ks:
        if best[k] >= THR and who[k] >= 0:
            for l in full[fs[who[k]]]:
                lib[l] += 1
    log("  归因到的来源库:")
    for l, c in lib.most_common(10):
        log(f"    {c:4d}  {l}")
    m39 = json.load(open(os.path.join(HERE, "m39_exposure_tiers.json")))
    prev = {k: v for k, v in m39["E2"]["identity_by_design"].items()}
    moved = sum(1 for k in ks
                if M39.tier_of(best[k]) > M39.tier_of(prev.get(f"{k[0]}|{k[1]}", 0.0)))
    t3_prev = {k for k in ks if M39.tier_of(prev.get(f"{k[0]}|{k[1]}", 0.0)) == 3}
    t3_kept = sum(1 for k in t3_prev if M39.tier_of(best[k]) == 3)
    log(f"  相对 M39:升级了 {moved} 个设计单元;M39 的 {len(t3_prev)} 个 T3 里仍是 T3 的 {t3_kept}")
    OUT["E1"] = {"n_exposed_designs": len(exposed), "n_designs": len(q),
                 "frac": round(len(exposed) / len(q), 4),
                 "m39_subset_count": 20, "m26d_proxy_count": 14}
    OUT["E2"] = {"tiers": {names[t]: tiers.get(t, 0) for t in range(4)},
                 "libraries": dict(lib.most_common()),
                 "n_designs_moved_up": moved,
                 "m39_T3": len(t3_prev), "m39_T3_still_T3": t3_kept,
                 "identity_by_design": {f"{k[0]}|{k[1]}": round(best[k], 5) for k in ks}}

    clean = [r for r in recs if r["design"] not in exposed]
    dirty = [r for r in recs if r["design"] in exposed]
    log(f"  未曝光干预 {len(clean)},曝光干预 {len(dirty)}")
    OUT["n_clean"], OUT["n_dirty"] = len(clean), len(dirty)
    if len(clean) < MIN_CLEAN:
        log(f"  **K1:未曝光 {len(clean)} < {MIN_CLEAN} ⇒ 只报 E1/E2**")
        OUT["verdict"] = "K1_clean_subset_too_small"
        OUT["log"] = LOG; json.dump(OUT, open(OUTJ, "w"), ensure_ascii=False, indent=1); return 0

    # ---- E3 / E4
    log("\n=== E3 未曝光 vs 曝光的技能对比(M26d 的 arm)")
    res = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        for tag, rs in (("all", recs), ("clean", clean), ("exposed", dirty)):
            res[f"{mname}_{tag}"] = M26D.arm(rs, f"{mname}/{tag}", pkey, log, SEED)
    OUT["E3_arms"] = res

    log("\n=== E4 技能随曝光层级(若靠记忆,应当单调上升)")
    for r in recs:
        r["_tier"] = M39.tier_of(best.get(r["design"], 0.0))
    sb = lambda r: (r["design"], min(r["depth"], 3), r["crossed"])
    E4 = {}
    for pkey, mname in (("nmut", "RibonanzaNet"), ("pmut", "ViennaRNA")):
        per = {}
        for t in range(4):
            rs = [r for r in recs if r["_tier"] == t]
            s, n = (M26.slope_in(rs, sb, pkey) if len(rs) >= 30 else (None, 0))
            per[names[t]] = {"n": len(rs), "slope": (round(s, 5) if s is not None else None)}
            log(f"  {mname:14s} {names[t]:22s} n={len(rs):5d}  斜率 "
                f"{('%+.5f' % s) if s is not None else '欠功效'}")
        vals = [v["slope"] for v in per.values() if v["slope"] is not None]
        per["monotone_rising"] = bool(len(vals) > 1 and all(
            vals[i] <= vals[i + 1] for i in range(len(vals) - 1)))
        E4[mname] = per
    OUT["E4_skill_by_tier"] = E4
    OUT["staked"] = {
        "1_exposure_above_m39": bool(len(exposed) > 20),
        "2_all_m39_T3_still_T3": bool(t3_kept == len(t3_prev)),
        "3_skill_not_monotone_in_tier": bool(
            not E4["RibonanzaNet"]["monotone_rising"] and not E4["ViennaRNA"]["monotone_rising"]),
        "4_at_least_one_moved_up": bool(moved >= 1)}
    log("\n=== 预注册记分")
    for k, v in OUT["staked"].items():
        log(f"  {k:34s} {'成立' if v else '**失败**'}")
    OUT["elapsed_s"] = round(time.time() - t0, 1); OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), ensure_ascii=False, indent=1)
    log(f"\n写出 {OUTJ}  用时 {OUT['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
