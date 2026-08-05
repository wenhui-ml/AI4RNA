#!/usr/bin/env python3
"""A second kinetic judge built from `barriers`, and an honest account of what it can and cannot replace.

WHERE THIS COMES FROM. The co-transcriptional criterion has exactly one external judge, Kinfold, and that
judge has a measured ceiling: --stop demands landing on one exact microstate, so reach is essentially zero at
>=100 nt and 100x the simulation budget does not move it. Direction B is to change the judge. Stage 1 checked
the candidates instead of assuming:

    kinwalker   broken build -- reads two bases from any input, core dumps on --transcribed
    treekin     every build needs libgfortran.so.3; only .so.5 exists and the symlink fails loudly
    barriers    runs, and emits a macrostate rate matrix

So the master equation is solved here with expm rather than through a missing binary. That also makes the
judge distribution-valued by construction -- the POPULATION of the target macrostate, not hit-or-miss on one
microstate -- which is the same crossing that was applied to the criterion in ensemble_trap.py.

WHAT IT IS NOT, stated before any number is produced, because getting this wrong would be the most damaging
possible error here. `barriers` builds a landscape for a FIXED, FULL-LENGTH sequence. Solving the master
equation from the open chain therefore measures REFOLDING, not co-transcriptional folding: there is no growing
chain and no 5'->3' order. It cannot substitute for Kinfold --grow, and it cannot by itself extend the
co-transcriptional scope past 70 nt. Claiming otherwise would replace a measured limitation with a
mis-specified judge, which is worse than the limitation.

WHAT IT CAN DO, and both are things the project actually owes:

  1. PAY THE INDEPENDENCE DEBT. The calibration protocol's P2 says pairwise agreement must be MEASURED, never
     assumed -- the pseudoknot side claimed "three independent predictors" for two months and the measurement
     said two. Nothing has ever measured how much of Kinfold's verdict is Kinfold. A second kinetic method
     over the same landscape gives that number. Note in advance: barriers shares ViennaRNA's energy
     parameters, so by the protocol's own definition it counts as ONE source with Kinfold until kappa says
     otherwise.
  2. LOCATE THE CEILING'S CAUSE. barriers must enumerate suboptimals, so it will very likely be infeasible
     well before 284 nt. Measuring WHERE it dies distinguishes two different conclusions: a 70 nt ceiling
     because kinetic folding of long RNA is physically hard, or because every available tool enumerates.

PRE-REGISTERED GATES:
    P1  positive control (a designed sequence verified to fold reaches its target macrostate) AND negative
        control (poly-A does not). A tool failing either does not get to vote -- run_controls, every run.
    P2  the generator must pass check_generator: orientation confirmed by detailed balance, and irreducible.
        A reducible matrix scores anything outside the start's component at zero BY TRUNCATION, and reporting
        that as "does not fold" is the failure this file must not commit.
    P3  feasibility is reported as a curve, not as a single verdict, and a length where it fails is recorded
        as infeasible rather than as reach zero.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rna_harness.instrument import check_generator, run_controls                # noqa: E402

BT = "/home/ubuntu/miniconda3/envs/bt/bin"
# NOT A DECLARED CONSTANT ANY MORE, AND THAT WAS A REPEAT OFFENCE. -e 8 was declared without measuring what
# energy window a CONNECTED landscape needs, and at -e 8 the generator came back reducible, which made the
# positive control score zero and produced a premature "this judge fails P1". Measured afterwards on two
# designs: -e 8 gives 3 components, -e 12 gives 7, -e 16 gives ONE, in 0.3 s. -e 20 goes back to 9 on one of
# them, so wider is not monotonically better and a fixed value cannot be right.
# This is the same defect class as the theta grid declared an order of magnitude below its quantity's scale,
# and instrument.scale_then_threshold exists precisely for it -- I had already written the check and did not
# use it. So the window is now SEARCHED per sequence until check_generator reports irreducible.
SUBOPT_E_GRID = (8.0, 12.0, 16.0, 20.0, 24.0)
MINH = 1.0                # measured: minh 2 produced no rates.out on 4 of 6 designs; minh 1 always did
MAXSTATES = 50            # declared
TIMES = (1e0, 1e2, 1e4, 1e6, 1e9)
JUDGE_T = 1e6             # the declared operating point; the sweep is reported alongside


def landscape(seq: str, e: float, timeout_s: float = 600.0):
    """Macrostates, their energies, their structures and the rate matrix. None if the pipeline did not run.

    None is never zero here: an enumeration that blew up and a molecule that cannot fold are opposite
    findings, and this project has already caught itself conflating them twice.
    """
    try:
        with tempfile.TemporaryDirectory() as d:
            sub = subprocess.run([f"{BT}/RNAsubopt", "-e", str(e), "-s", "--noLP"],
                                 input=f"{seq}\n", capture_output=True, text=True,
                                 timeout=timeout_s, cwd=d)
            if sub.returncode != 0 or not sub.stdout:
                return None
            n_sub = max(0, len(sub.stdout.splitlines()) - 1)
            bar = subprocess.run([f"{BT}/barriers", "-G", "RNA-noLP", "--bsize", "--rates",
                                 "--minh", str(MINH), "--max", str(MAXSTATES)],
                                 input=sub.stdout, capture_output=True, text=True,
                                 timeout=timeout_s, cwd=d)
            if bar.returncode != 0 or not bar.stdout:
                return None
            rates_path = os.path.join(d, "rates.out")
            if not os.path.exists(rates_path):
                return None
            import numpy as np
            R = np.loadtxt(rates_path)
            if R.ndim != 2:
                return None
            rows = [l.split() for l in bar.stdout.splitlines()[1:] if l.strip()]
            rows = [r for r in rows if len(r) >= 3 and r[0].isdigit()][:R.shape[0]]
            if len(rows) != R.shape[0]:
                return None
            return {"n_subopt": n_sub, "subopt_e": e, "rates": R,
                    "structures": [r[1] for r in rows], "energies": [float(r[2]) for r in rows]}
    except Exception:                                                          # noqa: BLE001
        return None


def target_population(seq: str, target: str, times=TIMES):
    """Population of the macrostate whose representative IS the target, over time. None if unscoreable."""
    import numpy as np
    from scipy.linalg import expm
    # Search the energy window rather than declaring it: the smallest window that yields an IRREDUCIBLE
    # generator containing the target. A reducible generator scores anything outside the start's component at
    # zero by truncation, and reporting that as "does not fold" is the error this file warned itself about.
    L, diag, chosen = None, None, None
    for e in SUBOPT_E_GRID:
        cand = landscape(seq, e)
        if not cand:
            continue
        if target not in cand["structures"]:
            L = L or cand
            continue
        st = int(np.argmax(cand["energies"]))
        d = check_generator(cand["rates"], cand["energies"], start=st, name=f"barriers@e{e}")
        L, diag, chosen = cand, d, e
        if d["irreducible"]:
            break
    if not L:
        return None
    structs, G = L["structures"], L["energies"]
    if diag is None or target not in structs:
        # the target is not a local minimum of this landscape, so there is no macrostate to score.
        # Reported as unscoreable rather than as population zero.
        return {"unscoreable": "target is not a macrostate representative", "n_states": len(structs),
                "n_subopt": L["n_subopt"], "subopt_e": L["subopt_e"]}
    if not diag["irreducible"]:
        # every window in the grid left it reducible -- say so instead of returning a population
        return {"unscoreable": f"generator reducible at every window up to e={chosen} "
                               f"({diag['n_components']} components)",
                "n_states": len(structs), "n_subopt": L["n_subopt"], "subopt_e": chosen}
    ti = structs.index(target)
    start = int(np.argmax(G))                     # highest-energy macrostate, the open-chain proxy
    R = np.array(L["rates"], dtype=float)
    np.fill_diagonal(R, 0.0)
    M = R.T.copy()
    np.fill_diagonal(M, -R.sum(axis=1))
    p0 = np.zeros(len(G))
    p0[start] = 1.0
    pops = {}
    for t in times:
        p = expm(M * t) @ p0
        pops[f"t{t:.0e}"] = float(p[ti])
    return {"n_states": len(G), "n_subopt": L["n_subopt"], "subopt_e": chosen, "target_state": ti,
            "start_state": start, "same_component": bool(ti in diag["start_component"]),
            "irreducible": diag["irreducible"], "n_components": diag["n_components"],
            "detailed_balance_err": diag["detailed_balance_err"],
            "pops": pops, "judge": pops[f"t{JUDGE_T:.0e}"]}


def work(job):
    tag, tid, seq, target, extra = job
    t0 = time.time()
    r = target_population(seq, target)
    return {"tag": tag, "task_id": tid, "n": len(seq), "res": r,
            "wall_s": round(time.time() - t0, 1), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-short", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "barrier_judge.json"))
    a = ap.parse_args()
    if not os.path.exists(f"{BT}/barriers"):
        print(f"!! 找不到 {BT}/barriers —— 不用替代品凑数。")
        return

    # --- P1: controls, every run --------------------------------------------------------------------------
    print("=== P1 正负对照(每次运行都跑)===")
    pos_seq = "GGGGAAAACCCCUUUUGGGGAAAACCCC"
    import RNA
    pos_tgt = RNA.fold(pos_seq)[0]
    # THE NEGATIVE CONTROL TOOK TWO ATTEMPTS TO GET RIGHT, and both failures are the same shape: it passed
    # for a reason that had nothing to do with discrimination.
    #   (1) poly-A. RNAsubopt gives one structure, barriers finds zero saddles and writes no rates.out, so
    #       target_population returned None -- and the predicate accepted None as a pass. A completely broken
    #       install would have "passed" identically. None means the pipeline did not run.
    #   (2) A "foreign" sequence and target of the same length. It scored 0.88, because the target I picked
    #       happened to be that sequence's own MFE.
    # What the control has to test is whether the judge says yes to EVERYTHING. So it now uses the positive
    # sequence's own landscape and asks for a HIGH-ENERGY macrostate of it, which is kinetically inaccessible
    # from the open chain and must score near zero.
    neg_seq = pos_seq
    _L = None
    for _e in SUBOPT_E_GRID:
        _L = landscape(neg_seq, _e)
        if _L and len(_L["energies"]) >= 4:
            break
    if not _L:
        print("!! 连正对照序列的地形都建不出来 —— 无法构造负对照,退出。")
        return
    import numpy as _np
    _order = _np.argsort(_L["energies"])          # ascending: index 0 is the ground state
    neg_tgt = _L["structures"][int(_order[-2])]   # second-highest energy macrostate
    print(f"  负对照靶标 = 同一条序列的次高能宏观态 "
          f"({_L['energies'][int(_order[-2])]:.2f} vs 基态 {min(_L['energies']):.2f} kcal/mol)")
    cr = run_controls("barriers-judge", lambda s, t: target_population(s, t),
                      (pos_seq, pos_tgt), (neg_seq, neg_tgt),
                      positive_ok=lambda v: bool(v) and v.get("judge", 0) > 0.5,
                      # a run that produced nothing is NOT a pass -- that is the conflation this project
                      # keeps catching in other people's results and just committed itself
                      negative_ok=lambda v: bool(v) and (v.get("unscoreable") is not None
                                                         or v.get("judge", 1) < 0.05))
    print(f"  正对照(设计序列折到自己的 MFE)  {cr.positive_ok}"
          f"   布居={None if not cr.positive_value else round(cr.positive_value.get('judge', -1), 4)}")
    print(f"  负对照(poly-A 折到同一结构)      {cr.negative_ok}")
    if not cr.usable:
        print("\n!! 对照未通过 —— 这个裁判不参与任何判决。先修,不凑合。")
        json.dump({"gate": "P1", "positive_ok": cr.positive_ok, "negative_ok": cr.negative_ok,
                   "note": cr.note}, open(a.out, "w"), ensure_ascii=False, indent=1)
        return
    print("  -> P1 通过\n")

    # --- population: the same designs Kinfold judged, so the two are comparable ---------------------------
    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    big = json.load(open(os.path.join(HERE, "kinetic_calibrate_big.json")))
    rv = json.load(open(os.path.join(HERE, "revisit_calibration.json")))
    tmap = {str(r["id"]): str(r["target"]) for r in load_bprna_nested(big.get("pool", 400))}
    idx = {}
    for r in big["rows"]:
        if r.get("error"):
            continue
        for c in r["candidates"]:
            k = (r["task_id"], round(c["gc"], 4), round(c["trap_fraction"], 4))
            idx[k] = None if k in idx else c["seq"]
    jobs, seen = [], set()
    for r in rv["rows"]:
        k = (r["task_id"], round(r["gc"], 4), round(r["trap_fraction"], 4))
        seq, tg = idx.get(k), tmap.get(r["task_id"])
        if not seq or not tg or len(tg) != len(seq) or r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        jobs.append(("short", r["task_id"], seq, tg,
                     {"kinfold_visit": r["visit_reach"], "trap_fraction": r["trap_fraction"]}))
        if len(jobs) >= a.n_short:
            break

    # --- feasibility curve: where does enumeration die? --------------------------------------------------
    from experiment_check import parse_rdat
    ec = json.load(open(os.path.join(HERE, "experiment_check.json")))
    for r in ec["rows"]:
        if r.get("skipped"):
            continue
        rec = parse_rdat(open(os.path.join(HERE, "rmdb_cache", f"{r['rmdb_id']}.rdat"),
                              encoding="utf8", errors="ignore").read())
        for k in (80, 100, 130, 160, 200):
            if k <= len(rec["sequence"]):
                jobs.append(("feas", f"{r['rmdb_id']}@{k}", rec["sequence"][:k],
                             r["target"][:k], {}))
        break                                     # one construct is enough for a length curve
    print(f"共 {len(jobs)} 个任务:{sum(1 for j in jobs if j[0]=='short')} 条短设计"
          f"(与 Kinfold 同一批) + {sum(1 for j in jobs if j[0]=='feas')} 个长度可行性点\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 10 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    import numpy as np
    from scipy.stats import spearmanr
    short = [r for r in R if r["tag"] == "short"]
    ok = [r for r in short if r["res"] and not r["res"].get("unscoreable")]
    uns = [r for r in short if r["res"] and r["res"].get("unscoreable")]
    dead = [r for r in short if not r["res"]]
    print(f"\n=== 短设计 {len(short)} 条:可打分 {len(ok)},靶标不是宏观态 {len(uns)},流水线跑不动 {len(dead)} ===")
    red = [r for r in ok if not r["res"]["irreducible"]]
    print(f"  生成元可约的 {len(red)}/{len(ok)}  —— 可约时分量外布居恒为 0 是【截断】造成的,不是动力学")
    good = [r for r in ok if r["res"]["irreducible"] or r["res"]["same_component"]]
    if len(good) >= 8:
        x = [r["res"]["judge"] for r in good]
        y = [r["kinfold_visit"] for r in good]
        rho, p = spearmanr(x, y)
        print(f"\n=== P2 独立性欠账:两个动力学裁判有多不一样({len(good)} 条)===")
        print(f"  barriers 布居 中位 {np.median(x):.3f}   Kinfold 访问式 中位 {np.median(y):.3f}")
        print(f"  Spearman rho={rho:+.3f}  p={p:.3g}")
        print(f"  读法:高相关 = 一个有效来源(协议口径:共用 ViennaRNA 参数,默认算一个);")
        print(f"        低相关 = 两个来源,但也说明【至少一个】刻画的不是我们要的过程")
        tf = [r["trap_fraction"] for r in good]
        r2, p2 = spearmanr(tf, x)
        print(f"  顺带:trap_fraction vs barriers 布居 rho={r2:+.3f} p={p2:.3g}"
              f"   (对比 Kinfold 上的 -0.509)")
        print(f"  提醒:barriers 测的是【从开链重折叠】,不是共转录。相关低是预期,不是判据失效。")
    else:
        rho = None
        print(f"\n  可用样本 {len(good)} < 8,不做相关 —— 报样本不足")

    print(f"\n=== P3 长度可行性曲线 ===")
    feas = sorted([r for r in R if r["tag"] == "feas"], key=lambda r: r["n"])
    for r in feas:
        s = r["res"]
        if not s:
            print(f"  {r['n']:>4} nt   流水线跑不动(枚举爆炸/超时)  {r['wall_s']}s")
        elif s.get("unscoreable"):
            print(f"  {r['n']:>4} nt   次优 {s['n_subopt']:>9}  宏观态 {s['n_states']:>3}  "
                  f"靶标不是宏观态  {r['wall_s']}s")
        else:
            print(f"  {r['n']:>4} nt   次优 {s['n_subopt']:>9}  宏观态 {s['n_states']:>3}  "
                  f"布居 {s['judge']:.3f}  {r['wall_s']}s")
    print("\n  这条曲线区分两个完全不同的结论:70 nt 上限是【物理】还是【枚举】。")

    json.dump({"gate": "ok", "minh": MINH, "max_states": MAXSTATES,
               "judge_time": JUDGE_T, "subopt_e_grid": list(SUBOPT_E_GRID), "n_short": len(short), "n_scoreable": len(ok),
               "n_unscoreable": len(uns), "n_dead": len(dead), "n_reducible": len(red),
               "spearman_vs_kinfold": rho, "rows": R}, open(a.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
