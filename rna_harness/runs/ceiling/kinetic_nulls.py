#!/usr/bin/env python3
"""Adversarial nulls for the SECOND ruler -- the thing it does not have and the first one does.

WHY THIS EXISTS. The pseudoknot criterion carries three adversarial nulls plus a register-shift decoy. The
co-transcriptional criterion carries none. It was validated once, against Kinfold, at +15.2 points with the
length confound removed. That is one leg, and a ruler standing on one leg is not a ruler.

AND THE FIRST THING THIS FILE FOUND IS THAT THE LEG IS SHAKIER THAN REPORTED. Re-scoring the stored run inside
(target, GC) strata -- same target, same composition to the fourth decimal -- leaves +6.0 points
[-1.6, +13.7], p=0.18 over 29 strata. The trap_fraction contrast inside those strata is intact (median 0.27
against 0.31 unpinned), so the test is not empty, but 29 strata gives roughly a third of the power needed for
the observed effect. NOT ESTABLISHED, not refuted -- and pre-registration mode 3 for this whole requirement was
"it collapses to GC again", so this is exactly the outcome that must not be quietly averaged away.

THREE MODES, each able to kill the criterion on its own:

  isomer   THE DECISIVE ONE. Swap the two bases of a target pair: G-C becomes C-G, A-U becomes U-A. The base
           MULTISET is preserved exactly -- not banded, not matched, IDENTICAL -- the target is unchanged, the
           length is unchanged, and every isomer still has to certify as folding to the target before it
           counts. The only thing that differs is WHERE the bases sit, which is precisely what a pathway
           property is supposed to see and a composition property cannot. If Kinfold reach still tracks
           trap_fraction across isomers of one parent, the criterion measures the pathway. If it does not, the
           criterion is a composition proxy and the second gap needs heavy qualification.

  grow     Is the criterion tuned to one arbitrary transcription speed? --grow 20 was a choice, not a
           measurement. Re-judge the same sequences at 5, 20 and 80. Pre-registered: the sign of the
           accepted-minus-rejected gap must hold at all three. Surviving only at 20 would mean the ruler is
           calibrated to a number we made up. This project's whole thesis is that verdicts are model-relative;
           our own criterion has to face that first.

  gcpin    Makes the re-scoring above permanent and reportable rather than a note in a chat log.

Kinfold is checked before it is trusted: a positive control reaches 20/20 and poly-A reaches 0/20, and a tool
that fails to launch returns None rather than 0, because "did not run" and "did not fold" are opposite
findings.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from kinetic_calibrate import KINFOLD, TRAJ, kinfold_reach                    # noqa: E402

GROW_RATES = (5.0, 20.0, 80.0)      # declared: 4x slower and 4x faster than the calibration run
ISOMERS_PER_PARENT = 14             # how many swap patterns to keep per parent, if they certify
SWAP_TRIES = 300                    # patterns attempted before giving up on a parent


# ---------------------------------------------------------------------------------------------------
# isomer: identical base multiset, identical target, different arrangement
# ---------------------------------------------------------------------------------------------------
def swap_isomer(seq: str, pairs, rng: random.Random, p: float) -> str:
    """Transpose the two partners of a random subset of target pairs.

    A transposition cannot change the base multiset -- that is the entire point, and it is why this is a
    stronger control than any GC band or GC matching: composition is preserved by construction rather than
    by selection. G-C -> C-G and A-U -> U-A are both still valid pairs, so the target stays formable; whether
    it is still the MFE is checked, never assumed.
    """
    s = list(seq)
    for i, j in pairs:
        if rng.random() < p:
            s[i], s[j] = s[j], s[i]
    return "".join(s)


def isomer_family(job):
    tid, target, timelimit, seed = job
    from rna_harness.benchmarks.kinetic_suite import cotranscriptional, gc_percent, pair_set
    from rna_harness.cognitive.certify import certify_sequence
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=6, timelimit_s=timelimit, results_number=20).design(
            DesignTask(task_id="kn", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "error": repr(e)[:160]}
    parent = None
    for s in ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or []):
        if s and len(s) == len(target) and certify_sequence(s, target).verdict in (
                "certified", "knot_confirmed"):
            parent = s
            break
    if not parent:
        return {"task_id": tid, "error": "no folding parent"}

    pairs = sorted(pair_set(target))
    if len(pairs) < 4:
        return {"task_id": tid, "error": "too few pairs to permute"}
    rng = random.Random(hash((tid, seed)) & 0xFFFFFFFF)
    base_mult = sorted(parent)
    seen, kept = {parent}, []
    for _ in range(SWAP_TRIES):
        if len(kept) >= ISOMERS_PER_PARENT:
            break
        iso = swap_isomer(parent, pairs, rng, rng.uniform(0.10, 0.95))
        if iso in seen:
            continue
        seen.add(iso)
        # ASSERTED, NOT ASSUMED. If a transposition ever changed the multiset the whole control is void, so
        # the invariant is checked on every single isomer rather than argued for once in a comment.
        if sorted(iso) != base_mult:
            return {"task_id": tid, "error": "multiset violated -- control invalid"}
        if certify_sequence(iso, target).verdict not in ("certified", "knot_confirmed"):
            continue
        k = cotranscriptional(iso, target)
        kept.append({"seq": iso, "trap_fraction": k["trap_fraction"],
                     "accepted": k["trap_fraction"] <= 0.10})

    kp = cotranscriptional(parent, target)
    fam = [{"seq": parent, "trap_fraction": kp["trap_fraction"],
            "accepted": kp["trap_fraction"] <= 0.10, "is_parent": True}] + kept
    lo = [c for c in fam if c["accepted"]]
    hi = [c for c in fam if not c["accepted"]]
    # Judging a family with only one side costs Kinfold time and can contribute nothing to a paired test.
    if not lo or not hi:
        return {"task_id": tid, "n": len(target), "n_isomers": len(fam),
                "gc": round(gc_percent(parent), 4),
                "skipped": "one-sided family (no contrast)", "wall_s": round(time.time() - t0, 1)}
    for c in fam:
        c["kinfold"] = kinfold_reach(c["seq"], target)
    return {"task_id": tid, "n": len(target), "n_isomers": len(fam),
            "gc": round(gc_percent(parent), 4), "n_low": len(lo), "n_high": len(hi),
            "candidates": fam, "wall_s": round(time.time() - t0, 1)}


def report_isomer(R, out):
    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    fams, skipped = [], 0
    for r in R:
        if r.get("error"):
            continue
        if r.get("skipped"):
            skipped += 1
            continue
        cs = [c for c in r["candidates"] if c.get("kinfold")]
        lo = [c["kinfold"]["reach"] for c in cs if c["accepted"]]
        hi = [c["kinfold"]["reach"] for c in cs if not c["accepted"]]
        if lo and hi:
            fams.append({"task_id": r["task_id"], "n": r["n"], "gc": r["gc"],
                         "low": st.mean(lo), "high": st.mean(hi),
                         "n_low": len(lo), "n_high": len(hi)})
    print(f"\n=== 同分异构体零假设 ===")
    print(f"  可用家族 {len(fams)};单侧家族(无对比度,已跳过){skipped}")
    if len(fams) < 6:
        print("  !! 家族太少,不下结论 —— 报为样本不足,不报为 null")
        json.dump({"mode": "isomer", "families": fams, "rows": R}, open(out, "w"),
                  ensure_ascii=False, indent=1)
        return
    A = np.array([f["low"] for f in fams])
    B = np.array([f["high"] for f in fams])
    dif = A - B
    p = float(wilcoxon(A, B, alternative="greater")[1])
    sd = float(dif.std(ddof=1))
    se = sd / len(dif) ** 0.5
    print(f"  每个家族内:碱基多重集完全相同、靶标相同、全部验证折叠正确")
    print(f"  判据接受(低陷阱)减 判据拒绝(高陷阱)的 Kinfold 到达率:")
    print(f"    均值 {dif.mean():+.1%}   中位 {np.median(dif):+.1%}   "
          f"95% CI [{dif.mean()-1.96*se:+.1%}, {dif.mean()+1.96*se:+.1%}]")
    print(f"    接受更好 {int((dif>0).sum())}   更差 {int((dif<0).sum())}   持平 {int((dif==0).sum())}")
    print(f"    Wilcoxon p={p:.3g}")
    allc = [(c["trap_fraction"], c["kinfold"]["reach"])
            for r in R if not r.get("error") and not r.get("skipped")
            for c in r["candidates"] if c.get("kinfold")]
    rho, ps = spearmanr([x for x, _ in allc], [y for _, y in allc])
    print(f"  全部 {len(allc)} 条异构体上 Spearman rho={rho:+.3f} p={ps:.3g}")
    verdict = ("通过 —— 判据在【组分完全固定】下仍预测独立模拟器,它量的是路径不是组分"
               if p < 0.05 and dif.mean() > 0 else
               "未通过 —— 组分固定后判据不再预测,第二条缺口需要大幅加限定")
    print(f"\n  预注册:组分固定下差距 >0 且 Wilcoxon p<0.05  ->  {verdict}")
    json.dump({"mode": "isomer", "n_families": len(fams), "mean_gap": float(dif.mean()),
               "ci95": [float(dif.mean() - 1.96 * se), float(dif.mean() + 1.96 * se)],
               "wilcoxon_p": p, "n_better": int((dif > 0).sum()), "n_worse": int((dif < 0).sum()),
               "spearman": {"rho": float(rho), "p": float(ps), "n": len(allc)},
               "families": fams, "rows": R}, open(out, "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------------
# grow: is the criterion tuned to one arbitrary transcription speed?
# ---------------------------------------------------------------------------------------------------
def grow_job(job):
    seq, target, grow = job
    return {"seq": seq, "grow": grow, "kinfold": kinfold_reach(seq, target, grow=grow)}


def run_grow(src, n_seq, workers, out, pool=400):
    import numpy as np
    from scipy.stats import mannwhitneyu, spearmanr
    d = json.load(open(src))
    # The stored rows carry the task_id but not the target string, so it is recovered from the SAME loader
    # with the SAME filter. The length check below is the guard: a target that does not match its candidate's
    # length means the recovery is wrong, and a wrong target would silently make every reach rate meaningless.
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(pool)}
    pool = [(r["task_id"], tmap.get(r["task_id"]), c) for r in d["rows"] if not r.get("error")
            for c in r["candidates"] if c["kinfold"]]
    pool = [(t, tg, c) for t, tg, c in pool if tg and len(tg) == len(c["seq"])]
    if not pool:
        print("!! 无法从存档恢复靶标序列 —— 不猜,直接退出")
        return
    # spread across the trap_fraction range rather than taking the first n
    pool.sort(key=lambda x: x[2]["trap_fraction"])
    step = max(1, len(pool) // n_seq)
    sub = pool[::step][:n_seq]
    print(f"从存档抽 {len(sub)} 条(按 trap_fraction 等距抽,不取前 n),每条在 {len(GROW_RATES)} 个转录速度下重判\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(c["seq"], tg, g) for _, tg, c in sub for g in GROW_RATES]
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(grow_job, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res[(r["seq"], r["grow"])] = r["kinfold"]
            if n % 60 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    table = []
    for g in GROW_RATES:
        a = [res[(c["seq"], g)]["reach"] for _, _, c in sub
             if c["accepted"] and res.get((c["seq"], g))]
        b = [res[(c["seq"], g)]["reach"] for _, _, c in sub
             if not c["accepted"] and res.get((c["seq"], g))]
        xs = [c["trap_fraction"] for _, _, c in sub if res.get((c["seq"], g))]
        ys = [res[(c["seq"], g)]["reach"] for _, _, c in sub if res.get((c["seq"], g))]
        if len(a) < 3 or len(b) < 3:
            print(f"  grow={g}: 样本不足(接受 {len(a)} 拒绝 {len(b)})—— 不报")
            continue
        pv = float(mannwhitneyu(a, b, alternative="greater")[1])
        rho, ps = spearmanr(xs, ys)
        table.append({"grow": g, "n_acc": len(a), "n_rej": len(b), "acc": st.mean(a),
                      "rej": st.mean(b), "gap": st.mean(a) - st.mean(b), "p": pv,
                      "rho": float(rho), "rho_p": float(ps)})
        print(f"  grow={g:>5}  接受 {st.mean(a):.1%} (n={len(a)})  拒绝 {st.mean(b):.1%} (n={len(b)})  "
              f"差 {st.mean(a)-st.mean(b):+.1%}  p={pv:.2g}   Spearman {rho:+.3f}")
    ok = len(table) == len(GROW_RATES) and all(t["gap"] > 0 and t["p"] < 0.05 for t in table)
    print(f"\n  预注册:三个转录速度上差距同号且 p<0.05  ->  "
          f"{'通过 —— 判据不是被调到某一个转录速度上的' if ok else '未通过 —— 照实报'}")
    json.dump({"mode": "grow", "grow_rates": list(GROW_RATES), "table": table,
               "passes": ok, "n_seq": len(sub)}, open(out, "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------------
# gcpin: make the re-scoring permanent
# ---------------------------------------------------------------------------------------------------
def run_gcpin(src, out):
    import numpy as np
    from scipy.stats import wilcoxon
    d = json.load(open(src))
    rows = [r for r in d["rows"] if not r.get("error")]
    strat = defaultdict(lambda: {"a": [], "r": []})
    for r in rows:
        for x in r["candidates"]:
            if x["kinfold"]:
                strat[(r["task_id"], round(x["gc"], 4))]["a" if x["accepted"] else "r"].append(x)
    use = [(k, v) for k, v in strat.items() if v["a"] and v["r"]]
    print(f"=== GC 逐条钉死(同 target 且组分完全相同)===")
    print(f"  可用层 {len(use)},涉及 target {len({k[0] for k, _ in use})}")
    if len(use) < 6:
        print("  样本不足,不下结论")
        return
    A = np.array([st.mean([y["kinfold"]["reach"] for y in v["a"]]) for _, v in use])
    B = np.array([st.mean([y["kinfold"]["reach"] for y in v["r"]]) for _, v in use])
    dif = A - B
    sd = float(dif.std(ddof=1)); se = sd / len(dif) ** 0.5
    p = float(wilcoxon(A, B, alternative="greater")[1])
    tg = [abs(st.mean([y["trap_fraction"] for y in v["a"]])
              - st.mean([y["trap_fraction"] for y in v["r"]])) for _, v in use]
    print(f"  层内差 均值 {dif.mean():+.1%}  中位 {np.median(dif):+.1%}  "
          f"95% CI [{dif.mean()-1.96*se:+.1%}, {dif.mean()+1.96*se:+.1%}]  p={p:.3g}")
    print(f"  接受更好 {int((dif>0).sum())}  更差 {int((dif<0).sum())}  持平 {int((dif==0).sum())}")
    print(f"  层内 trap_fraction 差 中位 {np.median(tg):.3f}  <- 对比度;太小则这个检验本来就看不见东西")
    # A NULL WITHOUT A POWER STATEMENT IS NOT A RESULT. It is the difference between "we looked and it is not
    # there" and "we could not have seen it if it were".
    need = (2.8 * sd / dif.mean()) ** 2 if dif.mean() > 0 else float("inf")
    print(f"  检验力:在观测到的效应量上达到 80% 需要约 {need:.0f} 个层(现有 {len(dif)})")
    print(f"  读法:{'确立' if p < 0.05 else '未确立(不是已否定)—— 区间跨零且检验力不足'}")
    json.dump({"mode": "gcpin", "n_strata": len(use), "mean_gap": float(dif.mean()),
               "ci95": [float(dif.mean() - 1.96 * se), float(dif.mean() + 1.96 * se)],
               "wilcoxon_p": p, "median_trap_contrast": float(np.median(tg)),
               "n_strata_for_80pct_power": float(need),
               "n_better": int((dif > 0).sum()), "n_worse": int((dif < 0).sum()),
               "n_tied": int((dif == 0).sum())}, open(out, "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("isomer", "grow", "gcpin"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timelimit", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=400,
                    help="how deep to read the bpRNA loader. For --mode grow this MUST match the pool the "
                         "source json was produced with, or the recovered targets will not be its targets.")
    ap.add_argument("--src", default=os.path.join(HERE, "kinetic_calibrate.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"kinetic_null_{a.mode}.json")
    if not os.path.exists(KINFOLD):
        print(f"!! 找不到外部裁判 {KINFOLD} —— 不用我们自己的前缀 MFE 顶替,那是自证。")
        return

    if a.mode == "gcpin":
        run_gcpin(a.src, out)
        return
    if a.mode == "grow":
        run_grow(a.src, a.n, a.workers, out, a.pool)
        return

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    rows = [r for r in load_bprna_nested(a.pool) if 25 <= len(str(r["target"])) <= 70][:a.n]
    jobs = [(str(r["id"]), str(r["target"]), a.timelimit, a.seed) for r in rows]
    print(f"同分异构体零假设:{len(jobs)} 条靶标,每条最多 {ISOMERS_PER_PARENT} 个异构体\n"
          f"  交换靶标碱基对的两端(G-C -> C-G),碱基多重集【构造上】不变;每个异构体仍须验证折叠正确\n"
          f"  外部裁判 Kinfold --grow,{TRAJ} 条轨迹/序列\n", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(isomer_family, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 8 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    report_isomer(R, out)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
