#!/usr/bin/env python3
"""Does panel agreement predict outside agreement -- asked WITHIN a target, where difficulty cancels?

WHAT THIS IS FOR, and it is not one more cross on a scorecard. This project ships exactly one certificate,
`knot_confirmed`, and the attribution measurement says 97.1% of the certificates it issues would be in
question under a different oracle. Whether that matters is the single most important open question about the
thing we actually ship. Without it, "97.1%" is a frightening number and nothing else.

THE FIRST ATTEMPT COULD NOT ANSWER IT, and the reason was sample size in a specific place. Aggregated,
unanimous certificates survived RNAPKplex 7/7 against 84/236 for split ones -- +64.4 points, and worth
nothing, because unanimous certificates may simply sit on easier targets and topology alone swings outside
agreement from 29.9% to 60.8%. Stratified, every cell held 0-4 unanimous certificates against a
pre-declared minimum of 8.

MORE SAMPLES ALONE WOULD NOT FIX THE CONFOUND. Pairing does. So this run does two things at once:

    generate SEVERAL designs per target with independent solver seeds, and
    compare unanimous against split ON THE SAME TARGET

Within a target the structure, the length, the topology and the difficulty are identical by construction --
the same device that made the co-transcriptional result survive (+16.1 within-target against a raw +26.5 that
turned out to be a length artefact).

PRE-REGISTERED, before any design was generated:

    Primary: within targets that produced BOTH a unanimous and a split design, the paired difference in
    outside agreement is > 0 by Wilcoxon, over at least 8 such targets.
    KILL: not significant -> panel agreement carries no information about outside agreement, judge_share is
    bookkeeping, and the frame stays descriptive. Fewer than 8 paired targets -> still not testable, and the
    shortfall is reported as a quantity, not as a null.

    THE HONEST n IS PAIRED TARGETS, never designs. Several designs of one target are not several independent
    observations, and this is the replication level that has already been got wrong here twice.

ALSO MEASURED, free with the same data and new: is unanimity a property of the TARGET or of the SEQUENCE?
If several independent designs for one target agree on whether the panel is unanimous, then judge_share is
mostly "some targets are easy for every oracle" -- a different statement from "the panel disagrees per
sequence". Nobody has separated those two readings, and they imply different products.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from external_judge import pkplex_pairs                                        # noqa: E402
from rna_harness.instrument import Clustered, constancy                        # noqa: E402

MIN_PAIRED = 8


def work(job):
    tid, target, seed, timelimit = job
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.cognitive.certify import _knot_recall
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    t0 = time.time()
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross:
        return {"task_id": tid, "seed": seed, "skipped": "no crossing pairs"}
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=6, timelimit_s=timelimit, results_number=20).design(
            DesignTask(task_id="jp", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "seed": seed, "error": repr(e)[:140]}
    out = []
    pool = ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or [])
    seen = set()
    for s in pool:
        if not s or len(s) != len(target) or s in seen:
            continue
        seen.add(s)
        kr = _knot_recall(s, target)
        if not kr.get("knot_confirmed"):
            continue                       # only sequences we would actually certify enter the comparison
        ps = pkplex_pairs(s)
        gc = 100.0 * sum(1 for c in s if c in "GC") / len(s)
        out.append({"seq": s, "confirmed_by": sorted(kr["confirmed_by"]),
                    "k_panel": len(kr["confirmed_by"]), "gc": round(gc, 2),
                    "external_ran": ps is not None,
                    "external": None if ps is None else bool((ps & cross) == cross)})
        if len(out) >= 6:                  # cap per (target, seed): more sequences are not more targets
            break
    return {"task_id": tid, "seed": seed, "n_cross": len(cross), "designs": out,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--timelimit", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="0 = every pseudobase target")
    ap.add_argument("--out", default=os.path.join(HERE, "judge_paired.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    from rna_harness.structure_utils import classify_topology
    tasks = list(load_pseudobase())
    if a.n:
        tasks = tasks[:a.n]
    topo = {str(t.task_id): classify_topology(str(t.target))["topology"] for t in tasks}
    jobs = [(str(t.task_id), str(t.target), s, a.timelimit)
            for t in tasks for s in range(a.seeds)]
    print(f"同靶标配对:{len(tasks)} 个靶标 × {a.seeds} 个独立种子 = {len(jobs)} 次设计\n"
          f"  只有【我们会给它发证书】的序列才进入比较(knot_confirmed 为真)\n"
          f"  外部裁判 RNAPKplex,从未进入任何规则\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 50 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    from scipy.stats import wilcoxon
    by = defaultdict(list)
    for r in R:
        if r.get("error") or r.get("skipped"):
            continue
        for d in r.get("designs", []):
            if d["external_ran"]:
                by[r["task_id"]].append(d)
    n_designs = sum(len(v) for v in by.values())
    print(f"\n=== 可用 {n_designs} 条【已发证且外部裁判跑通】的设计,来自 {len(by)} 个靶标 ===")
    ks = [d["k_panel"] for v in by.values() for d in v]
    print(f"  面板确认家数分布:" + "  ".join(f"{k}家 {ks.count(k)}" for k in (1, 2, 3)))

    # --- primary: within-target pairing ----------------------------------------------------------------
    pairs, gcs = [], []
    for t, v in by.items():
        un = [d for d in v if d["k_panel"] == 3]
        sp = [d for d in v if 1 <= d["k_panel"] <= 2]
        if un and sp:
            pairs.append((t, st.mean([d["external"] for d in un]),
                          st.mean([d["external"] for d in sp]), len(un), len(sp)))
            gcs.append((st.mean([d["gc"] for d in un]), st.mean([d["gc"] for d in sp])))
    print(f"\n=== 主检验:同靶标内【一致 vs 分裂】===")
    print(f"  两种设计都有的靶标 {len(pairs)} 个(预注册门槛 {MIN_PAIRED})")
    res = None
    if len(pairs) >= MIN_PAIRED:
        A = np.array([p[1] for p in pairs]); B = np.array([p[2] for p in pairs])
        d = A - B
        p = float(wilcoxon(A, B, alternative="greater")[1])
        se = float(d.std(ddof=1)) / len(d) ** 0.5
        print(f"  配对差 均值 {d.mean():+.1%}  中位 {np.median(d):+.1%}  "
              f"95% [{d.mean()-1.96*se:+.1%}, {d.mean()+1.96*se:+.1%}]")
        print(f"  一致更好 {int((d>0).sum())}  更差 {int((d<0).sum())}  持平 {int((d==0).sum())}"
              f"   (持平被 scipy 默认丢弃,实际参与 {int((d!=0).sum())})")
        print(f"  Wilcoxon p={p:.3g}")
        gu = st.mean([g[0] for g in gcs]); gs_ = st.mean([g[1] for g in gcs])
        print(f"  组分平衡:一致组 GC {gu:.1f}% 对 分裂组 {gs_:.1f}%  "
              f"(差 {gu-gs_:+.1f} 点 —— 大则这一格仍被组分污染)")
        cl = Clustered(values=list(d), clusters=[p_[0] for p_ in pairs])
        print(f"  重复层级:{cl.n_obs} 条 / {cl.n_clusters} 个靶标 —— {cl.warn_if_inflated() or '一靶标一条,无膨胀'}")
        ok = p < 0.05 and d.mean() > 0
        res = {"n_paired": len(pairs), "gap": float(d.mean()), "p": p,
               "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()),
               "gc_unan": gu, "gc_split": gs_, "passes": bool(ok)}
        print(f"\n  预注册:配对差 >0 且 p<0.05  ->  "
              f"{'通过 —— 面板一致确实携带外部正确性信息' if ok else '未通过 —— judge_share 是记账,框架停在描述层'}")
    else:
        short = MIN_PAIRED - len(pairs)
        rate = len(pairs) / max(1, len(by))
        need = int(MIN_PAIRED / rate) if rate > 0 else None
        print(f"  !! 仍不可测,还差 {short} 个配对靶标。")
        print(f"     实测配对率 {rate:.1%};按此需要约 {need if need else '∞'} 个靶标(现有 {len(by)})")
        print(f"     报为【不可测】,不报 null。")
        res = {"n_paired": len(pairs), "insufficient": True, "pair_rate": rate, "n_targets_needed": need}

    # --- free and new: is unanimity a property of the TARGET or of the SEQUENCE? -------------------------
    print(f"\n=== 一致性是【靶标】的性质还是【序列】的性质 ===")
    multi = {t: v for t, v in by.items() if len(v) >= 2}
    same = [1.0 if len({d["k_panel"] == 3 for d in v}) == 1 else 0.0 for v in multi.values()]
    c = constancy(same)
    print(f"  {len(multi)} 个靶标有 >=2 条设计;其中所有设计【一致性判定相同】的 {int(sum(same))} 个"
          f" = {sum(same)/max(1,len(same)):.1%}")
    frac_same = sum(same) / max(1, len(same))
    reading = ("一致性主要是【靶标】的性质 —— judge_share 说的是「有些靶标对所有裁判都容易」"
               if frac_same > 0.8 else
               "一致性随【序列】变 —— 同一靶标的不同设计会得到不同的面板判决")
    print(f"  -> {reading}")
    print(f"     (这两种读法指向不同的产品:前者可以在设计前就告诉用户,后者只能逐条判)")

    json.dump({"n_targets": len(by), "n_designs": n_designs, "k_panel_hist": {k: ks.count(k) for k in (1, 2, 3)},
               "paired": res, "unanimity_is_target_property": sum(same) / max(1, len(same)),
               "n_multi_design_targets": len(multi), "constancy": c,
               "rows": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
