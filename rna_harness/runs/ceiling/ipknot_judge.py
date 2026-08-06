#!/usr/bin/env python3
"""A SIXTH judge, and the third distinct algorithm class -- the point is convergence, not one more number.

WHERE THIS STANDS. Two outside judges now exist, and the effect they measure is not stable across them:
ShapeKnots' within-target advantage was +42.7 points under RNAPKplex and +21.8 under Knotty, while consensus
moved the other way (+5.3 -> +10.3). Two points cannot tell a converging sequence from a diverging one. This
is the third point, and it was chosen to be as far from both as the field allows:

    RNAPKplex   heuristic helix assembly          Turner-family energies
    Knotty      exact sparsified DP               HotKnots DP09 energies
    IPknot      integer programming over          LinearPartition-C probabilities  <- LEARNED, not
                base-pairing probabilities                                            thermodynamic

So the three do not share a search strategy, and IPknot does not even share the thermodynamic scoring basis.

RUN AT DEFAULTS, DELIBERATELY. IPknot exposes -t thresholds and -e models. Tuning either while looking at
agreement with our designs would be the error already on this project's record -- an alignment was once
chosen by maximising the model-under-test's own AUROC, letting the accused pick the jury. Defaults, declared
here, and nothing is touched afterwards.

PRE-REGISTERED, before IPknot judged a single design:

    P1  IPknot is redundant with NEITHER existing judge: Cohen's kappa < 0.8 against RNAPKplex and against
        Knotty. If it is redundant with either, it is not a third point and P2/P3 prove little.
    P2  The STEP at k=3 replicates: read against IPknot's OWN native ceiling, k=3 sits above it while k=1
        and k=2 sit below. KILL: if k=1/k=2/k=3 are flat under this judge, the step is a property of the
        first two judges, not of the designs, and the grade text shipped in certify.py must be withdrawn.
    P3  The ShapeKnots within-target advantage replicates (paired difference > 0, >= 8 paired targets).
        Its SIZE across three judges is reported as a sequence -- converging or shrinking toward zero is
        the actual question, and a third significant-but-smaller effect is evidence for the latter.

WHAT THE SELF-TEST MAY AND MAY NOT ASSERT. An earlier version of this check asserted "a known pseudoknot must
come back with crossing pairs". That is a statement about the TOOL's sensitivity, not about the wrapper, and
IPknot at defaults calls that same sequence nested -- which would have been reported as a wrapper defect.
`instrument_audit.py` established the distinction (9/9 external tools clean; every failure was the analyst's).
So the self-test asserts only wrapper-level properties -- a valid matching, the right length, poly-A empty,
and a dynamic range -- and the tool's sensitivity is MEASURED as S_ext rather than gated on a threshold
somebody invented. Worse, the old case 2 took its expected label from Knotty's own output, i.e. it tested
Knotty against Knotty.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

IPKNOT = os.environ.get("IPKNOT", "/data/wenhui/pk_judges/ipknot/build/ipknot")
IPKNOT_LIB = os.environ.get("IPKNOT_LIB", "/home/ubuntu/miniconda3/envs/ipk/lib")
MIN_PAIRED = 8


def ipknot_pairs(seq: str, timeout_s: float = 120.0):
    """Fold with the sixth judge and return its pair set. None means it failed to run, never a rejection."""
    from rna_harness.benchmarks.rnainvbench import pk_pairs
    if not os.path.exists(IPKNOT):
        return None
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = IPKNOT_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        with tempfile.TemporaryDirectory() as d:
            fa = os.path.join(d, "s.fa")
            with open(fa, "w") as fh:
                fh.write(">s\n" + seq + "\n")
            r = subprocess.run([IPKNOT, fa], capture_output=True, text=True, timeout=timeout_s, env=env)
    except Exception:                                                          # noqa: BLE001
        return None
    for ln in reversed((r.stdout or "").splitlines()):
        body = ln.strip()
        if body and set(body) <= set(".()[]{}<>") and len(body) == len(seq):
            return pk_pairs(body)
    return None


def self_test() -> bool:
    """Wrapper-level only. Tool sensitivity is measured (S_ext), never asserted against an invented number."""
    from rna_harness.instrument import InstrumentError, check_pair_set
    probes = ["GGCGCGGCACCGUCCGCGGAACAAACGG", "GCAACGAUGACAUACAUCGCUAGUCGACGC", "A" * 28,
              "GGGAAACCCGGGAAACCCGGGAAACCC"]
    print("=== wrapper 自检(只断言 wrapper 级性质;工具灵敏度改为量 S_ext,不设发明出来的阈值)===")
    ok, counts = True, []
    for seq in probes:
        ps = ipknot_pairs(seq)
        if ps is None:
            print(f"  ✗ 跑不动:{seq[:24]}")
            ok = False
            continue
        try:
            check_pair_set(ps, len(seq))
            prob = ""
        except InstrumentError as e:
            prob = str(e)
        bad = bool(prob) or any(not (0 <= i < j < len(seq)) for i, j in ps)
        ok = ok and not bad
        counts.append(len(ps))
        print(f"  {'✓' if not bad else '✗'} {len(seq):3d} nt -> {len(ps):2d} 对,配对集是合法匹配"
              f"{'  !! ' + prob if prob else ''}")
    pa = ipknot_pairs("A" * 28)
    print(f"  {'✓' if pa == set() else '✗'} poly-A 给出 {len(pa or [])} 对(负对照必须【能够】通过)")
    print(f"  {'✓' if len(set(counts)) > 1 else '✗'} 动态范围:{len(set(counts))} 种不同的配对数")
    return ok and pa == set() and len(set(counts)) > 1


def _cal():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cal", os.path.join(HERE, "calibrate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def native_job(job):
    tid, target, seq = job
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross or len(seq) != len(target):
        return None
    ps = ipknot_pairs(seq)
    return (tid, None if ps is None else bool(ps >= cross))


def seq_job(seq):
    ps = ipknot_pairs(seq)
    return seq, (None if ps is None else sorted(ps))


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "ipknot_judge.json"))
    a = ap.parse_args()
    if not self_test():
        print("\n!! wrapper 自检未通过 —— 不判任何序列。")
        sys.exit(1)

    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import _round_only, load_pseudobase, pk_pairs
    from external_judge import clopper_pearson

    tgt = {str(t.task_id): str(t.target) for t in load_pseudobase()}
    natives = _cal()._native_sequences()
    njobs = [(str(k), tgt[str(k)], v) for k, v in natives.items() if str(k) in tgt]
    print(f"\n=== S_ext:IPknot 在【天然序列】上的自身天花板(n={len(njobs)})===", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        NAT = [r for r in ex.map(native_job, njobs) if r]
    v = [r[1] for r in NAT if r[1] is not None]
    lo, hi = clopper_pearson(sum(v), len(v))
    S_ip = sum(v) / len(v)
    print(f"  IPknot {sum(v)}/{len(v)} = {S_ip:.1%}  95% [{lo:.1%}, {hi:.1%}]")
    print(f"  (对比 RNAPKplex 66.7% / Knotty 73.7%,同一批天然序列)")
    if sum(v) == 0:
        print("  !! 它连一条天然假结都认不出 —— 不能用作裁判,停。")
        sys.exit(1)

    P = json.load(open(os.path.join(HERE, "judge_paired.json")))
    K = json.load(open(os.path.join(HERE, "knotty_judge.json")))
    kn = {s: (None if x is None else {tuple(y) for y in x}) for s, x in K["knotty"].items()}
    seqs = sorted({d["seq"] for r in P["rows"] if not (r.get("error") or r.get("skipped"))
                   for d in r.get("designs", []) if d["external_ran"]})
    print(f"\n=== 用 IPknot 判 {len(seqs)} 条设计序列 ===", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        ip = dict(ex.map(seq_job, seqs))
    nfail = sum(1 for x in ip.values() if x is None)
    print(f"  跑通 {len(ip)-nfail}/{len(ip)}({(time.time()-t0)/60:.1f} min;失败记为【未跑】,绝不当否决)")

    by, trip = defaultdict(list), []
    for r in P["rows"]:
        if r.get("error") or r.get("skipped"):
            continue
        tg = tgt.get(r["task_id"])
        if not tg:
            continue
        cross = pk_pairs(tg) - pk_pairs(_round_only(tg))
        for d in r.get("designs", []):
            if not d["external_ran"] or kn.get(d["seq"]) is None or ip.get(d["seq"]) is None:
                continue
            rec = {**d, "pk": bool(d["external"]), "kn": bool(kn[d["seq"]] >= cross),
                   "ip": bool({tuple(x) for x in ip[d["seq"]]} >= cross)}
            by[r["task_id"]].append(rec)
            trip.append(rec)
    print(f"  三个外部裁判都判过的设计 {len(trip)} 条,来自 {len(by)} 个靶标")

    print(f"\n=== P1 三个外部裁判两两冗余吗(Cohen's kappa,门槛 <0.8)===")
    KA = {}
    for x, y, lab in (("pk", "kn", "RNAPKplex~Knotty"), ("pk", "ip", "RNAPKplex~IPknot"),
                      ("kn", "ip", "Knotty~IPknot")):
        KA[lab] = kappa([d[x] for d in trip], [d[y] for d in trip])
        print(f"  {lab:22s} kappa = {KA[lab]:.3f}  {'冗余' if KA[lab] >= 0.8 else '不冗余'}")
    print(f"  确认率:RNAPKplex {st.mean([d['pk'] for d in trip]):.1%}  "
          f"Knotty {st.mean([d['kn'] for d in trip]):.1%}  IPknot {st.mean([d['ip'] for d in trip]):.1%}")

    CEIL = {"pk": 0.667, "kn": 0.737, "ip": S_ip}
    print(f"\n=== P2 台阶还在吗(每一级同意率 / 该裁判自身天花板)===")
    print(f"{'级别':>6s}{'n':>7s}{'PKplex':>10s}{'Knotty':>10s}{'IPknot':>10s}")
    grade = {}
    for k in (1, 2, 3):
        v2 = [d for d in trip if d["k_panel"] == k]
        row = {j: st.mean([d[j] for d in v2]) / CEIL[j] for j in ("pk", "kn", "ip")}
        grade[k] = {"n": len(v2), **{j: st.mean([d[j] for d in v2]) for j in ("pk", "kn", "ip")},
                    "vs_ceiling": row}
        print(f"{'k='+str(k):>6s}{len(v2):7d}{row['pk']:10.0%}{row['kn']:10.0%}{row['ip']:10.0%}")
    step = grade[3]["vs_ceiling"]["ip"] > 1.0 and grade[1]["vs_ceiling"]["ip"] < 1.0 \
        and grade[2]["vs_ceiling"]["ip"] < 1.0

    import numpy as np
    from scipy.stats import wilcoxon
    print(f"\n=== P3 ShapeKnots 的同靶标配对优势,三个裁判并排 ===")
    sk = lambda d: "shapeknots" in d["confirmed_by"]                            # noqa: E731
    seq_eff = {}
    for j, name in (("pk", "RNAPKplex"), ("kn", "Knotty"), ("ip", "IPknot")):
        Q = []
        for t, vv in by.items():
            A = [d[j] for d in vv if sk(d)]
            B = [d[j] for d in vv if not sk(d)]
            if A and B:
                Q.append((st.mean(A), st.mean(B)))
        if len(Q) < MIN_PAIRED:
            print(f"  {name:11s} 配对靶标 {len(Q)} < {MIN_PAIRED},不检验")
            continue
        d_ = np.array([q[0] for q in Q]) - np.array([q[1] for q in Q])
        p = float(wilcoxon([q[0] for q in Q], [q[1] for q in Q], alternative="greater")[1])
        seq_eff[name] = {"n_paired": len(Q), "gap": float(d_.mean()), "p": p}
        print(f"  {name:11s} {len(Q):3d} 靶标  差 {d_.mean():+6.1%}  p={p:.3g}")
    sizes = [seq_eff[n]["gap"] for n in ("RNAPKplex", "Knotty", "IPknot") if n in seq_eff]
    print(f"  效应量序列 {[f'{x:+.1%}' for x in sizes]} —— "
          f"{'仍全部为正,但逐个变小:这更像【随裁判而定】而非一个稳定的量' if len(sizes) == 3 and sizes == sorted(sizes, reverse=True) else '未呈单调下降'}")

    print(f"\n=== 预注册判决 ===")
    p1 = all(v < 0.8 for v in KA.values())
    p3 = seq_eff.get("IPknot", {}).get("p", 1) < 0.05 and seq_eff.get("IPknot", {}).get("gap", 0) > 0
    print(f"  P1 {'通过:三个裁判两两都不冗余,确实是三个点' if p1 else '未通过:至少有两个裁判冗余'}")
    print(f"  P2 {'通过:台阶在第三个裁判上仍在(k=3 超天花板,k=1/k=2 低于)' if step else '未通过 —— 台阶是前两个裁判的性质,certify.py 里的分级文案必须撤回'}")
    print(f"  P3 {'通过' if p3 else '未通过'}")

    json.dump({"S_ext_ipknot": S_ip, "n_native": len(v), "kappa": KA, "grade": grade,
               "step_holds": step, "paired": seq_eff, "P1": p1, "P3": p3,
               "n_designs": len(trip), "n_targets": len(by), "n_failed": nfail,
               "ipknot": ip}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
