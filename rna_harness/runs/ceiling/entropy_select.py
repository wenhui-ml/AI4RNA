#!/usr/bin/env python3
"""If our designs are over-determined, does PICKING the less-determined one buy outside agreement?

THE HYPOTHESIS AND WHERE IT CAME FROM. k1_diagnosis.py refuted its own pre-registration in the opposite
direction: our designs are not diffuse, they are roughly three times MORE determined than natural
pseudoknots (mean positional entropy 0.087 against 0.299; MFE ensemble probability 0.88 against 0.45). What
survived is a mechanism -- within the design pool, the sequences IPknot accepts have higher entropy (0.1029)
than the ones it rejects (0.0746), and natives sit higher than both. A probability-thresholded pseudoknot
method has nothing to work with when the nested fold owns the whole ensemble.

That is a correlation found after a failed pre-registration, so it decides nothing yet. This week already
showed the cost of trusting one of those: the lottery-ticket reading of the agent ladder was wrong and would
have deleted the only rung that solves long targets. So it gets tested the only way that counts.

PROSPECTIVE, ON SEEDS NEVER USED. judge_paired.py generated its corpus with seeds 0,1,2; this run uses 10,11
on the same 251 targets, so every candidate pool here is new. The hypothesis was formed on the old pools and
is tested on new ones.

WHAT IS ACTUALLY BEING TESTED is a SELECTION RULE, not a solver change -- the same shape as this week's
grade-max change, which is why it could ship if it works. Among the candidates that already clear the
shipped floor, which one do we hand back?

    R0  current: highest panel grade k, ties keep the earliest candidate
    R1  proposed: highest panel grade k, ties broken by HIGHER ensemble entropy
    R2  entropy only: highest entropy among confirmed candidates, ignoring grade
    RC  control that must be able to fail: highest GC among confirmed candidates. Composition alone was
        already shown to carry some signal (AUC 0.569), so if picking on GC does as well as picking on
        entropy, the ensemble story adds nothing and is decoration.

PRE-REGISTERED, before any design was generated:

    P1  R1 beats R0 on IPknot acceptance, paired within target, over at least 8 targets where the two rules
        pick different sequences. One-sided sign test.
    P2  R1 does not LOSE on the other two judges: no significant drop against R0 for Knotty or RNAPKplex.
        A rule that trades two judges for one is not an improvement, it is a preference.
    P3  R2 (entropy alone) also beats R0. If R1 wins but R2 does not, entropy is a tie-breaker and nothing
        more, and the write-up says exactly that.

    KILL: P1 fails -> the entropy correlation is not actionable, it goes into the "not claimed" list, and
    the selection rule stays as it is.

    THE HONEST n IS TARGETS where the rules disagree. Several candidates of one target are not several
    observations. Ties are reported, not dropped silently.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SEEDS = (10, 11)                      # judge_paired used 0,1,2 -- these pools have never existed


def work(job):
    tid, target, seed, timelimit = job
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.cognitive.certify import _knot_recall
    from rna_harness.generators.desirna import DesiRNAGenerator
    from rna_harness.task import DesignTask, OracleConfig
    from external_judge import pkplex_pairs
    from knotty_judge import knotty_pairs
    from ipknot_judge import ipknot_pairs
    from k1_diagnosis import ensemble_features

    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross:
        return {"task_id": tid, "seed": seed, "skipped": "no crossing pairs"}
    oc = OracleConfig(param_set="turner2004", temperature=37.0, dangles=2)
    try:
        res = DesiRNAGenerator(replicas=6, timelimit_s=timelimit, results_number=20).design(
            DesignTask(task_id="es", target=target, oracle=oc), seed=seed)
    except Exception as e:                                                     # noqa: BLE001
        return {"task_id": tid, "seed": seed, "error": repr(e)[:140]}
    out, seen = [], set()
    pool = ([res.best_sequence] if res.best_sequence else []) + list(res.candidates or [])
    for s in pool:
        if not s or len(s) != len(target) or s in seen:
            continue
        seen.add(s)
        kr = _knot_recall(s, target)
        if not kr.get("knot_confirmed"):
            continue
        try:
            ef = ensemble_features(s)
        except Exception:                                                      # noqa: BLE001
            continue
        pp, kk, ii = pkplex_pairs(s), knotty_pairs(s), ipknot_pairs(s)
        out.append({"seq": s, "k": len(kr["confirmed_by"]), "entropy": ef["entropy"], "gc": ef["gc"],
                    "pk": None if pp is None else bool(pp >= cross),
                    "kn": None if kk is None else bool(kk >= cross),
                    "ip": None if ii is None else bool(ii >= cross)})
        if len(out) >= 8:
            break
    return {"task_id": tid, "seed": seed, "designs": out}


def sign_p(up, dn):
    n = up + dn
    return sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else float("nan")


RULES = {
    "R0 现行(等级最高,并列取先)": lambda C: max(range(len(C)), key=lambda i: (C[i]["k"], -i)),
    "R1 等级最高,并列取高熵":     lambda C: max(range(len(C)), key=lambda i: (C[i]["k"], C[i]["entropy"])),
    "R2 只取最高熵":               lambda C: max(range(len(C)), key=lambda i: C[i]["entropy"]),
    "RC 对照:只取最高 GC":        lambda C: max(range(len(C)), key=lambda i: C[i]["gc"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelimit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "entropy_select.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())
    if a.n:
        tasks = tasks[:a.n]
    jobs = [(str(t.task_id), str(t.target), s, a.timelimit) for t in tasks for s in SEEDS]
    print(f"系综选择规则:{len(tasks)} 个靶标 × 种子 {SEEDS} = {len(jobs)} 次设计\n"
          f"  种子从未用过(judge_paired 用的是 0,1,2)—— 假设在旧池子上形成,在新池子上检验\n"
          f"  三个外部裁判全部从未进入任何规则\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 40 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    by = {}
    for r in R:
        if r.get("error") or r.get("skipped"):
            continue
        for d in r.get("designs", []):
            if d["ip"] is not None and d["kn"] is not None and d["pk"] is not None:
                by.setdefault(r["task_id"], []).append(d)
    pools = {t: C for t, C in by.items() if len(C) >= 2}          # a rule can only differ if there is a choice
    print(f"\n=== 可用 {sum(len(v) for v in by.values())} 条已发证设计 / {len(by)} 个靶标;"
          f"其中候选 ≥2 条、规则才可能分歧的 {len(pools)} 个 ===")
    ks = [d["k"] for v in by.values() for d in v]
    print(f"  面板等级分布 " + "  ".join(f"k={k} {ks.count(k)}" for k in (1, 2, 3)))
    print(f"  平均位点熵 {st.mean([d['entropy'] for v in by.values() for d in v]):.4f}"
          f"(天然 0.2985,旧池子里的设计 0.0869)")

    pick = {name: {t: fn(C) for t, C in pools.items()} for name, fn in RULES.items()}
    base = "R0 现行(等级最高,并列取先)"
    res = {}
    for judge, jl in (("ip", "IPknot"), ("kn", "Knotty"), ("pk", "RNAPKplex")):
        print(f"\n=== 裁判 {jl} ===")
        print(f"{'规则':>26s}{'接受率':>9s}{'与 R0 分歧的靶标':>18s}{'胜':>5s}{'负':>5s}{'符号 p':>10s}")
        for name in RULES:
            acc = st.mean([pools[t][pick[name][t]][judge] for t in pools])
            diff = [t for t in pools if pick[name][t] != pick[base][t]]
            up = sum(1 for t in diff if pools[t][pick[name][t]][judge]
                     and not pools[t][pick[base][t]][judge])
            dn = sum(1 for t in diff if pools[t][pick[base][t]][judge]
                     and not pools[t][pick[name][t]][judge])
            p = sign_p(up, dn) if (up + dn) else float("nan")
            res.setdefault(judge, {})[name] = {"acc": acc, "n_diff": len(diff), "up": up, "dn": dn, "p": p}
            print(f"{name:>26s}{acc:9.1%}{len(diff):18d}{up:5d}{dn:5d}{p:10.3g}"
                  + ("   <- 基准" if name == base else ""))

    r1 = "R1 等级最高,并列取高熵"
    r2 = "R2 只取最高熵"
    rc = "RC 对照:只取最高 GC"
    ip1 = res["ip"][r1]
    P1 = bool(ip1["up"] + ip1["dn"] >= 8 and ip1["p"] < 0.05 and ip1["up"] > ip1["dn"])
    P2 = all(not (res[j][r1]["dn"] > res[j][r1]["up"] and res[j][r1]["p"] < 0.05) for j in ("kn", "pk"))
    ip2 = res["ip"][r2]
    P3 = bool(ip2["up"] + ip2["dn"] >= 8 and ip2["p"] < 0.05 and ip2["up"] > ip2["dn"])
    print(f"\n=== 预注册判决 ===")
    if ip1["up"] + ip1["dn"] < 8:
        print(f"  P1 分歧靶标只有 {ip1['up']+ip1['dn']} 个 < 8 -> 不可测,不报 null")
    else:
        print(f"  P1 R1 在 IPknot 上胜过 R0:{'通过' if P1 else '未通过'}"
              f"({ip1['up']} 胜 {ip1['dn']} 负,p={ip1['p']:.3g})")
    print(f"  P2 R1 未在另外两家裁判上显著变差:{'通过' if P2 else '未通过 —— 拿两家换一家不算改进'}")
    print(f"  P3 R2(纯熵)也胜过 R0:{'通过 —— 熵本身在起作用' if P3 else '未通过 —— 熵至多是个并列打破器'}")
    rcip = res["ip"][rc]
    if rcip["up"] > rcip["dn"] and rcip["p"] < 0.05:
        print(f"  !! 对照 RC(只取高 GC)同样显著胜出({rcip['up']}/{rcip['dn']})"
              f" —— 这可能只是组分,不是系综")
    print(f"\n  杀死条款:P1 不过 -> 系综相关性不可操作,写进【不主张】,选择规则不动。")

    json.dump({"n_targets": len(by), "n_pools": len(pools), "res": res,
               "P1": P1, "P2": P2, "P3": P3, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
