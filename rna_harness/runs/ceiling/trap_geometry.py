#!/usr/bin/env python3
"""Are we counting the right objects? A structure/randomness split and a scale sweep for trap_fraction.

THE ONE ROBUST NEGATIVE WE HAVE. On natural riboswitches probed by co-transcriptional SHAPE-seq, our
criterion calls 96.4% of transcription intermediates traps. A word that applies to almost everything is not
doing work. That number survived every design flaw found in experiment_check.py, so it is the thing to attack.

trap_fraction is a POINTWISE, UNWEIGHTED COUNT:

    trap_fraction = #{k : MFE(seq[:k]) has a pair the target does not} / #{k}

Two borrowed method-shapes say the same thing about that definition, from different directions, and each one
produces a falsifiable experiment we would not otherwise have run. Borrowing that does not produce such an
experiment is decoration, so both are stated as kill conditions first.

  (1) STICKY / NON-STICKY, after the dichotomy Wang and Zahl use on Kakeya: either a configuration is highly
      structured -- tubes through nearby points pointing in nearby directions -- in which case you classify
      it and handle it separately, or it is not, in which case a counting argument wins. You must win in BOTH
      branches. Our own H-type episode only ever won the structured branch, by restriction.

      In folding the structured enemy has a name we already validated: a COMPETING REGISTER. An off-target
      pair (i,j) is STICKY if the target has a pair (a,b) with |i-a| <= D and |j-b| <= D -- the same helix,
      slipped. That is exactly the object the register-shift decoy was built around, where our pseudoknot
      criterion refused 0/386. Everything else is an incidental pairing.

      PRE-REGISTERED: sticky_trap_fraction tracks Kinfold reach MORE strongly than nonsticky_trap_fraction.
      KILL: if the two are indistinguishable, the dichotomy does not hold in RNA. Report the null and do NOT
      go looking for a third way to cut it -- that is how one manufactures a result.

  (2) INDUCTION ON SCALES, the other half of that programme: never look at a single scale. A trap that exists
      at one prefix length and is gone at the next is a blip; a trap that survives twenty nucleotides of
      elongation is a structure. The current definition cannot tell them apart because it looks at each k
      independently.

      PRE-REGISTERED: persistence at L >= 5 tracks Kinfold reach MORE strongly than L = 1, which is exactly
      today's trap_fraction. KILL: if L = 1 wins, scale is not the missing ingredient and this is refuted.

  (3) And the number that started it: does either variant stop labelling ~96% of NATURAL transcription
      intermediates as traps? A redefinition that predicts Kinfold better but still fires on everything has
      not fixed the thing that is broken.

Nothing here is a claim about mathematics. It is a claim about which objects to count, tested the same way
everything else in this project is tested.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

D_STICKY = 3              # declared before the run: how far a helix may slip and still be "the same helix"
SCALES = (1, 3, 5, 10, 20)
MIN_PREFIX = 12           # same as kinetic_suite, so the two definitions see the same prefixes


def trap_profile(seq: str, target: str):
    """Per-prefix trap detail: is it a trap, and are its off-target pairs slipped helices or incidental?

    One pass of prefix folding produces every variant below, so the pointwise count, the sticky split and
    the scale sweep are all measured on IDENTICAL folding calls. Comparing definitions that were computed
    from different foldings would confound the comparison with solver noise.
    """
    import RNA
    from rna_harness.benchmarks.kinetic_suite import pair_set
    tgt = pair_set(target)
    ks = list(range(MIN_PREFIX, len(seq) + 1))
    prof = []
    for k in ks:
        mp = pair_set(RNA.fold(seq[:k])[0])
        off = mp - tgt
        sticky = [p for p in off
                  if any(abs(p[0] - a) <= D_STICKY and abs(p[1] - b) <= D_STICKY for a, b in tgt)]
        prof.append({"k": k, "n_off": len(off), "n_sticky": len(sticky)})
    return {"ks": ks, "profile": prof}


def variants(prof):
    """Every candidate definition, from one profile. Named so a reader can see which is today's rule."""
    P = prof["profile"]
    n = max(1, len(P))
    is_trap = [p["n_off"] > 0 for p in P]
    # a prefix is sticky-trapped when the MAJORITY of its off-target pairs are slipped helices
    is_sticky = [p["n_off"] > 0 and p["n_sticky"] * 2 >= p["n_off"] for p in P]
    is_nonst = [t and not s for t, s in zip(is_trap, is_sticky)]
    out = {"trap_fraction": sum(is_trap) / n,               # <- today's rule, L=1, unweighted
           "sticky_fraction": sum(is_sticky) / n,
           "nonsticky_fraction": sum(is_nonst) / n,
           "sticky_share": (sum(is_sticky) / sum(is_trap)) if any(is_trap) else None}
    # persistence: a trap counts only if it survives L consecutive nucleotides of elongation
    for L in SCALES:
        run = sum(1 for i in range(len(is_trap) - L + 1) if all(is_trap[i:i + L]))
        out[f"persist_{L}"] = run / n
        srun = sum(1 for i in range(len(is_sticky) - L + 1) if all(is_sticky[i:i + L]))
        out[f"sticky_persist_{L}"] = srun / n
    runs, cur = [], 0
    for t in is_trap:
        cur = cur + 1 if t else 0
        runs.append(cur)
    out["max_run"] = max(runs) / n if runs else 0.0
    return out


def work(job):
    tid, target, seq, reach = job
    try:
        v = variants(trap_profile(seq, target))
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "error": repr(e)[:120]}
    v.update({"task_id": tid, "n": len(target), "reach": reach})
    return v


def cluster_bootstrap_spearman(rows, a, b, n_boot=4000, seed=0):
    """Is variant `a` more strongly related to reach than variant `b`? Resampled by TARGET, not by candidate.

    Candidates from one target are not independent draws, and a bootstrap over candidates would give an
    interval far too tight -- the same pseudo-replication that made the first Kinfold intervals look clean.
    """
    import numpy as np
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    keys = list(by)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        sub = [r for i in pick for r in by[keys[i]]]
        if len(sub) < 10:
            continue
        ra = spearmanr([r[a] for r in sub], [r["reach"] for r in sub])[0]
        rb = spearmanr([r[b] for r in sub], [r["reach"] for r in sub])[0]
        if ra == ra and rb == rb:
            diffs.append(abs(ra) - abs(rb))
    if not diffs:
        return None
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    return {"mean": float(sum(diffs) / len(diffs)), "lo95": float(lo), "hi95": float(hi),
            "p_one_sided": float(sum(1 for d in diffs if d <= 0) / len(diffs))}


def run_predict(src, workers, out):
    import numpy as np
    from scipy.stats import spearmanr
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    d = json.load(open(src))
    pool = d.get("pool", 400)
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(pool)}
    jobs = []
    for r in d["rows"]:
        if r.get("error"):
            continue
        tg = tmap.get(r["task_id"])
        for c in r["candidates"]:
            if c.get("kinfold") and tg and len(tg) == len(c["seq"]):
                jobs.append((r["task_id"], tg, c["seq"], c["kinfold"]["reach"]))
    print(f"重算 {len(jobs)} 条候选的逐前缀陷阱几何(与 Kinfold 到达率同一批,折叠调用完全共用)\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 200 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)
    rows = [r for r in R if not r.get("error")]
    print(f"\n=== {len(rows)} 条可用,来自 {len({r['task_id'] for r in rows})} 个靶标 ===")

    names = ["trap_fraction", "sticky_fraction", "nonsticky_fraction"] + \
            [f"persist_{L}" for L in SCALES] + [f"sticky_persist_{L}" for L in SCALES] + ["max_run"]
    print("\n  每个定义与 Kinfold 到达率的 Spearman(越负越好;trap_fraction 是【今天的规则】):")
    sp = {}
    for nm in names:
        xs = [r[nm] for r in rows]
        if len(set(xs)) < 3:
            print(f"    {nm:22s} 取值几乎不变,跳过")
            continue
        rho, p = spearmanr(xs, [r["reach"] for r in rows])
        sp[nm] = {"rho": float(rho), "p": float(p)}
        mark = "  <- 今天的规则" if nm == "trap_fraction" else ""
        print(f"    {nm:22s} rho={rho:+.3f}  p={p:.2g}{mark}")

    print("\n  [预注册 1] 黏性 vs 非黏性:黏性是否更强?")
    b1 = cluster_bootstrap_spearman(rows, "sticky_fraction", "nonsticky_fraction")
    if b1:
        print(f"    |rho(黏性)| - |rho(非黏性)| = {b1['mean']:+.3f}  "
              f"95% [{b1['lo95']:+.3f}, {b1['hi95']:+.3f}]  单侧 p={b1['p_one_sided']:.3g}")
        print(f"    -> {'黏性更强,二分成立' if b1['lo95'] > 0 else '分不开 —— 二分在 RNA 上不成立,报 null,不去找第三种切法'}")

    print("\n  [预注册 2] 尺度:L>=5 的持续性是否强过 L=1(= 今天的规则)?")
    best = max((L for L in SCALES if f"persist_{L}" in sp), key=lambda L: abs(sp[f"persist_{L}"]["rho"]),
               default=None)
    b2 = None
    if best is not None:
        print(f"    最强的尺度是 L={best}(rho={sp[f'persist_{best}']['rho']:+.3f})")
        if best != 1:
            b2 = cluster_bootstrap_spearman(rows, f"persist_{best}", "persist_1")
            if b2:
                print(f"    |rho(L={best})| - |rho(L=1)| = {b2['mean']:+.3f}  "
                      f"95% [{b2['lo95']:+.3f}, {b2['hi95']:+.3f}]  单侧 p={b2['p_one_sided']:.3g}")
                print(f"    -> {'尺度是缺的那味料' if b2['lo95'] > 0 else '不显著 —— 尺度不是缺的那味料'}")
        else:
            print("    -> L=1 就是最强的,尺度归纳在这里被否证")

    # what the pointwise rule was actually made of
    shares = [r["sticky_share"] for r in rows if r.get("sticky_share") is not None]
    if shares:
        print(f"\n  今天规则数进去的那些陷阱里,【黏性】占比 中位 {st.median(shares):.1%}")
    json.dump({"n": len(rows), "d_sticky": D_STICKY, "scales": list(SCALES), "spearman": sp,
               "boot_sticky_vs_nonsticky": b1, "boot_scale": b2, "best_scale": best,
               "sticky_share_median": st.median(shares) if shares else None,
               "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {out}")


def run_natives(out):
    """The 96.4%: does either redefinition stop firing on almost every natural intermediate?"""
    import glob
    import re
    from experiment_check import parse_rdat
    cache = os.path.join(HERE, "rmdb_cache")
    ec = json.load(open(os.path.join(HERE, "experiment_check.json")))
    tgt = {r["rmdb_id"]: r["target"] for r in ec["rows"] if not r.get("skipped")}
    print(f"=== 天然 RNA 上,各定义把多少比例的转录中间体判成陷阱 ===")
    print(f"  只用参照质量合格的构建体({len(tgt)} 个),与 experiment_check 同一批\n")
    agg, per = {}, []
    for p in sorted(glob.glob(os.path.join(cache, "*.rdat"))):
        rid = os.path.basename(p)[:-5]
        if rid not in tgt:
            continue
        rec = parse_rdat(open(p, encoding="utf8", errors="ignore").read())
        seq, T = rec["sequence"], tgt[rid]
        v = variants(trap_profile(seq, T))
        mol = re.match(r"[A-Z0-9]+", rid).group(0)
        per.append({"rmdb_id": rid, "molecule": mol, **{k: v[k] for k in
                    ("trap_fraction", "sticky_fraction", "nonsticky_fraction",
                     "persist_5", "persist_10", "persist_20", "max_run")}})
        print(f"  {rid:22s} 今天 {v['trap_fraction']:.1%}  黏性 {v['sticky_fraction']:.1%}  "
              f"L=5 {v['persist_5']:.1%}  L=20 {v['persist_20']:.1%}")
    for k in ("trap_fraction", "sticky_fraction", "persist_5", "persist_10", "persist_20"):
        agg[k] = st.mean([r[k] for r in per]) if per else None
    print(f"\n  构建体平均:")
    for k, v in agg.items():
        mark = "  <- 就是那个 96%" if k == "trap_fraction" else ""
        print(f"    {k:22s} {v:.1%}{mark}")
    json.dump({"per_construct": per, "mean": agg}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("predict", "natives"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--src", default=os.path.join(HERE, "kinetic_calibrate_big.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f"trap_geometry_{a.mode}.json")
    if a.mode == "predict":
        run_predict(a.src, a.workers, out)
    else:
        run_natives(out)


if __name__ == "__main__":
    main()
