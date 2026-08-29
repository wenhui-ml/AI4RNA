#!/usr/bin/env python3
"""M40v: the two verification steps M38 and M39 had not done.

This is a REVIEW run, not a new experiment: it produces no new claim and is declared as review in
provenance.json. Both M38 and M39 shipped step (1) 对拍 and step (4) 零点/恒等式. Neither shipped
step (2) 等价形式 or step (3) 反向对照, and the project's own rule is that a boundary result is not
judged until all four are done. Offline, $0.

  M38-A  反向对照  M31's arm_A, with M31's ORIGINAL predictor columns, on the same cells, must
                   reproduce M31's committed numbers. If it does, M38's numbers differ from
                   M31's only because of the columns M38 added, and nothing else moved.
  M38-B  等价形式  the within-design agreement recomputed as Kendall tau-b -- a different rank
                   statistic, not a re-run of the same one -- and Spearman rebuilt from the
                   covariance formula instead of the normalised-rank dot product. The verdict
                   (loses to depth) must survive both.
  M39-A  反向对照  M26d's k-mer-indexed max_identity and M39's exhaustive scanner, run on the
                   SAME input. This isolates the implementation change from the change of
                   comparison set, which is the only way to know whether 12.7% vs 8.9% is about
                   the corpus or about the code.
  M39-B  等价形式  identity by counting MISMATCHES and subtracting, instead of counting matches.
"""
from __future__ import annotations

import collections, csv, json, os, random, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np
from scipy.stats import rankdata, kendalltau

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m26_model_scored as M26
import m31_which_flips_rescue as M31
import m38_ribosnitch_baseline as M38
import m39_exposure_tiers as M39
import m26d_clean_subset as M26D

OUTJ = os.path.join(HERE, "m40v_verify_m38_m39.json")
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def kendall_weighted(cells, key, ykey="rf"):
    num = den = 0.0
    for _k, rs in cells:
        x = [r[key] for r in rs]; y = [r[ykey] for r in rs]
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        t = kendalltau(x, y).statistic
        if t is None or not np.isfinite(t):
            continue
        num += len(rs) * float(t); den += len(rs)
    return num / den if den else None


def spearman_cov(cells, key, ykey="rf"):
    """Spearman rebuilt from the covariance formula rather than from centred unit vectors."""
    num = den = 0.0
    for _k, rs in cells:
        x = rankdata([r[key] for r in rs]); y = rankdata([r[ykey] for r in rs])
        sx, sy = x.std(), y.std()
        if sx == 0 or sy == 0:
            continue
        rho = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        num += len(rs) * rho; den += len(rs)
    return num / den if den else None


def identity_by_mismatch(query, bank):
    """M39-B 等价形式: (Lq - mismatches)/Lq instead of matches/Lq. Same number by construction,
    which is the point -- a disagreement would mean an indexing or edge-case bug."""
    q = M39.enc(query); Lq = len(q)
    best = 0.0
    for L, (A, _idx) in bank.items():
        if L < Lq:
            continue
        for o in range(L - Lq + 1):
            mm = (A[:, o:o + Lq] != q).sum(axis=1)
            v = (Lq - int(mm.min())) / Lq
            if v > best:
                best = v
    return best


def main():
    t0 = time.time()
    log("=== M40v:补上 M38 与 M39 缺的第 ② 步(等价形式)与第 ③ 步(反向对照)===")
    log("  这是 review,不产生新主张。四步没走完之前不下判断。离线 $0。\n")
    OUT = {"kind": "review", "purpose": "steps 2 and 3 of the four-step verification for M38, M39"}

    recs, _dr = M26.build()
    zc = np.load(os.path.join(HERE, "m26c_pneural_cache.npz"), allow_pickle=True)
    pn = {k: v for k, v in zip(zc["keys"], zc["mats"])}
    for r in recs:
        i, j = min(r["a"], r["b"]), max(r["a"], r["b"])
        r["nmut"] = float(pn[r["mut_seq"]][i, j])
        r["nwt"] = float(pn[r["wt_seq"]][i, j])
        r["ndp"] = r["nmut"] - r["nwt"]
    del pn, zc
    cells = M31.make_cells(recs, nmin=M31.NMIN)
    log(f"  {len(recs)} 次干预,{len(cells)} 个设计格 —— 与 M31/M38 同一批\n")

    # ---------------------------------------------------------------- M38-A 反向对照
    log("=== M38-A 反向对照:用 M31 的原始列跑 M31 的 arm_A,必须复现 M31 已提交的数 ===")
    m31j = json.load(open(os.path.join(HERE, "m31_which_flips_rescue.json")))["A_within_design"]
    M31.PREDS = (("P_vienna", "pmut"), ("dP_vienna", "dp"), ("P_neural", "nmut"),
                 ("dP_neural", "ndp"), ("depth_baseline", "depth"))
    back = M31.arm_A(cells, "反向", seed=310, do_perm=False)
    drift = {}
    for nm in ("P_neural", "P_vienna", "depth_baseline"):
        old = m31j[nm]["rho"]; new = back[nm]["rho"]
        drift[nm] = {"m31_committed": old, "reproduced": new, "delta": round(new - old, 8)}
        log(f"  {nm:16s} M31 已提交 {old:+.5f}   本次复现 {new:+.5f}   差 {new-old:+.2e}")
    a_ok = all(abs(v["delta"]) < 1e-6 for v in drift.values())
    log(f"  → {'通过:M38 的数与 M31 的差别只来自 M38 加的那几列' if a_ok else '**失败**'}\n")
    OUT["M38_A_reverse_control"] = {"drift": drift, "pass": bool(a_ok)}

    # ---------------------------------------------------------------- M38-B 等价形式
    log("=== M38-B 等价形式:换一个秩统计量(Kendall tau-b)与另一种 Spearman 写法 ===")
    log("  重算 M38 的分数(与 M38 同一批函数,同一个方向约定)...")
    bywt = collections.defaultdict(list)
    for r in recs:
        bywt[r["wt_seq"]].append(r)
    M38.log = log_quiet = (lambda *a, **k: None)
    M38.score_all(bywt)
    log(f"  重算完成 {time.time()-t0:.0f}s")

    m38j = json.load(open(os.path.join(HERE, "m38_ribosnitch_baseline.json")))["A_within_design"]
    rows = {}
    for nm, key in (("d_bppm", "rs_bppm"), ("d_mfe", "rs_mfe"), ("d_ens", "rs_ens"),
                    ("depth_baseline", "depth")):
        ry = M31.ry_map(cells)
        sp_dot, _ = M31.weighted_rho(cells, key, ry)
        rows[nm] = {"m38_committed": m38j[nm]["rho"],
                    "spearman_dot_recomputed": round(sp_dot, 5),
                    "spearman_covariance_form": round(spearman_cov(cells, key), 5),
                    "kendall_tau_b": round(kendall_weighted(cells, key), 5)}
        log(f"  {nm:16s} M38 {rows[nm]['m38_committed']:+.5f}  点积式 {sp_dot:+.5f}  "
            f"协方差式 {rows[nm]['spearman_covariance_form']:+.5f}  "
            f"Kendall {rows[nm]['kendall_tau_b']:+.5f}")
    same_repro = all(abs(rows[n]["spearman_dot_recomputed"] - rows[n]["m38_committed"]) < 1e-4
                     for n in rows)
    dep = rows["depth_baseline"]
    verdict_holds = all(rows[n]["kendall_tau_b"] < dep["kendall_tau_b"]
                        and rows[n]["spearman_covariance_form"] < dep["spearman_covariance_form"]
                        for n in ("d_bppm", "d_mfe", "d_ens"))
    log(f"  重算复现 M38 = {same_repro};**在两种等价写法下,三个分数仍全部低于深度 = {verdict_holds}**")
    log(f"  → {'通过' if (same_repro and verdict_holds) else '**失败**'}\n")
    OUT["M38_B_equivalent_form"] = {"rows": rows, "reproduces_m38": bool(same_repro),
                                    "verdict_survives": bool(verdict_holds),
                                    "pass": bool(same_repro and verdict_holds)}

    # ---------------------------------------------------------------- M39-A 反向对照
    log("=== M39-A 反向对照:同一份输入,M26d 的 k-mer 索引法 vs M39 的穷尽扫描 ===")
    log("  这一步分离【换了实现】与【换了比较集】。12.7% vs 8.9% 必须是后者,不能是前者。")
    csv.field_size_limit(10 ** 9)
    seqs = sorted({d["sequence"] for d in csv.DictReader(open(M39.TRAIN))})
    rng = random.Random(20260829)
    sub = [seqs[i] for i in rng.sample(range(len(seqs)), 4000)]
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
    bank_sub = M39.build_bank(sub)
    new_best = {k: M39.max_identity(v, bank_sub)[0] for k, v in q.items()}
    old_best, _p1 = M26D.max_identity(dict(q), set(sub), lambda *a, **k: None)
    worst = max(abs(new_best[k] - old_best[k]) for k in q)
    n_new = sum(v >= 0.8 for v in new_best.values())
    n_old = sum(v >= 0.8 for v in old_best.values())
    log(f"  同一 {len(sub)} 条子集上:穷尽扫描判曝光 {n_new} 个,k-mer 索引法判 {n_old} 个")
    log(f"  逐设计单元最大同一性的最大偏差 {worst:.3e}")
    same_call = (n_new == n_old)
    log(f"  → {'通过:两种实现在同一输入上给出同一个曝光判定,所以差别来自比较集' if same_call else '**两种实现判定不同 —— 差别里有实现的成分,必须报告**'}\n")
    OUT["M39_A_reverse_control"] = {"n_subset": len(sub), "exhaustive_exposed": int(n_new),
                                    "kmer_index_exposed": int(n_old),
                                    "max_identity_abs_diff": float(worst),
                                    "same_call": bool(same_call)}

    # ---------------------------------------------------------------- M39-B 等价形式
    log("=== M39-B 等价形式:数错配再相减,而不是数匹配 ===")
    ks = sorted(q)
    smp = [ks[i] for i in random.Random(7).sample(range(len(ks)), 12)]
    worst2 = max(abs(M39.max_identity(q[k], bank_sub)[0] - identity_by_mismatch(q[k], bank_sub))
                 for k in smp)
    b_ok = worst2 < 1e-12
    log(f"  {len(smp)} 个查询:最大偏差 {worst2:.3e}  {'通过' if b_ok else '**失败**'}\n")
    OUT["M39_B_equivalent_form"] = {"n_queries": len(smp), "max_abs_diff": float(worst2),
                                    "pass": bool(b_ok)}

    ok = (a_ok and same_repro and verdict_holds and same_call and b_ok)
    OUT["verdict"] = ("四步走完,M38 与 M39 的结论可以下判断"
                      if ok else "**验证未全过 —— 不许在这两个结果上下判断**")
    log(f"判决:{OUT['verdict']}")
    log(f"用时 {time.time()-t0:.0f}s")
    OUT["log"] = LOG
    json.dump(OUT, open(OUTJ, "w"), indent=1, ensure_ascii=False)
    open(OUTJ.replace(".json", ".log"), "w").write("\n".join(LOG) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
