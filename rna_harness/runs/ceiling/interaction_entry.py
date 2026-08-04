#!/usr/bin/env python3
"""Entry check P1-P3 for a THIRD judgment-missing domain: RNA-RNA interaction structure.

RUN STRICTLY BY docs/CALIBRATION_PROTOCOL.md, which was written before this file and before any of this data
existed. That is the whole point: two domains were built by discovering each step after needing it, so "we
did it twice" cannot separate method from improvisation. The third domain is a test OF THE PROTOCOL, and the
only question it has to answer is not whether it worked but WHICH STEP THE PROTOCOL FAILED TO SPECIFY.

The protocol's entry conditions, and every one of them is allowed to fail:

  P1  at least 3 INDEPENDENT runnable predictors. Independence is defined operationally in the protocol
      because v1 of it left the word undefined and I promptly ticked the box with four ViennaRNA binaries:
      "binaries from one package count as ONE source until kappa says otherwise". IntaRNA reports
      "using Vienna RNA package 2.7.0" on startup -- its own codebase and its own algorithm, but the same
      energy parameters -- so it does NOT get to be a third source by assertion either. Only P2 decides.

  P2  they must DISAGREE. Measured, never assumed. On pseudoknots this project claimed "three independent
      predictors" for two months and the measurement said ProbKnot~ShapeKnots = 0.74, i.e. two effective
      sources, which changed what every certificate grade meant.

  P3  the strict rule -- every source reproducing the identical interaction -- must deliver almost nothing.
      If strictness already delivers, no calibration is needed and there is no contribution to make.

FAILING IS A RESULT. If P2 says these tools mostly agree, this domain has a consensus and we swap domains
rather than lowering the bar. That decision is written here before the numbers exist.

POPULATION, chosen to need no new solver runs and to be the case the project actually has a stake in: the
1596 already-archived DesiRNA designs, each paired with a trigger strand that is the reverse complement of a
window inside it. That is exactly the object the two-state switch capability manipulates -- the most-named
requirement across 1676 design papers, and the one no external tool exposes.

The four predictors and what each is:
    RNAduplex   ViennaRNA -- hybridisation only, no intramolecular structure
    RNAcofold   ViennaRNA -- full dimer partition function, intramolecular structure allowed
    IntaRNA     own codebase and algorithm (accessibility + seed), ViennaRNA energy parameters
    bifold      RNAstructure -- separate codebase AND separate parameter set
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

from rna_harness.instrument import (check_duplex_antiparallel, check_pair_set,   # noqa: E402
                                    run_controls)

VPK = "/home/ubuntu/miniconda3/envs/vpk/bin"
INTA = "/home/ubuntu/miniconda3/envs/inta/bin/IntaRNA"
RNASTR = "/home/ubuntu/miniconda3/envs/rnastr/bin"
DATAPATH = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
WINDOW = 12               # trigger length; declared before the run
KAPPA_CONSENSUS = 0.80    # P2 kill threshold: agreement at or above this means the field HAS a verdict


def _run(cmd, inp=None, cwd=None, env=None, t=120):
    try:
        r = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=t, cwd=cwd, env=env)
        return r.stdout or ""
    except Exception:                                                          # noqa: BLE001
        return None


def pairs_rnaduplex(a: str, b: str):
    """`(((((((((.&)))))))))   1,10  :   1,9   (-16.70)` -- hybrid only, plus the two windows it used."""
    out = _run([f"{VPK}/RNAduplex"], inp=f"{a}\n{b}\n")
    if not out:
        return None
    m = re.search(r"([().&]+)\s+(\d+),(\d+)\s*:\s*(\d+),(\d+)", out)
    if not m:
        return None
    st, s1, e1, s2, _e2 = m.group(1), *map(int, m.groups()[1:])
    left, right = st.split("&") if "&" in st else (st, "")
    stack, pr = [], set()
    for i, ch in enumerate(left):
        if ch == "(":
            stack.append(s1 - 1 + i)
    # Bracket matching is LIFO: the FIRST closer pairs with the LAST opener. Iterating closers in reverse
    # pairs last-with-last, which silently produces a PARALLEL duplex -- physically impossible, and it
    # matched RNAcofold's (correct) antiparallel answer on zero pairs. That fabricated a disagreement
    # between two tools that actually agree, which is exactly the P2 verdict this file exists to decide.
    closers = [s2 - 1 + i for i, ch in enumerate(right) if ch == ")"]
    for j in closers:
        if stack:
            pr.add((stack.pop(), j))
    return pr


def pairs_rnacofold(a: str, b: str):
    """Dot-bracket over A&B; only pairs that CROSS the & are intermolecular."""
    out = _run([f"{VPK}/RNAcofold", "--noPS"], inp=f"{a}&{b}\n")
    if not out:
        return None
    m = re.search(r"^([().&]+)", out.splitlines()[-2] if len(out.splitlines()) > 1 else "", re.M)
    if not m:
        for line in out.splitlines():
            if set(line.split()[0] if line.split() else "") <= set(".()&") and "&" in line:
                m = re.match(r"([().&]+)", line)
                break
    if not m:
        return None
    s = m.group(1)
    cut = s.index("&")
    flat = s.replace("&", "")
    stack, pr = [], set()
    for i, ch in enumerate(flat):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            i0 = stack.pop()
            if i0 < cut <= i:
                pr.add((i0, i - cut))
    return pr


def pairs_intarna(a: str, b: str):
    out = _run([INTA, "-t", a, "-q", b, "--outMode", "C"])
    if not out:
        return None
    for line in out.splitlines():
        f = line.split(";")
        if len(f) >= 9 and f[1].isdigit():
            s1, s2, hyb = int(f[1]), int(f[4]), f[7]
            left, right = hyb.split("&")
            stack = [s1 - 1 + i for i, ch in enumerate(left) if ch == "("]
            closers = [s2 - 1 + i for i, ch in enumerate(right) if ch == ")"]
            pr = set()
            for j in closers:                                  # LIFO, same fix as RNAduplex above
                if stack:
                    pr.add((stack.pop(), j))
            return pr
    return set()


def pairs_bifold(a: str, b: str):
    """RNAstructure joins the strands with a 3-nt linker; pairs spanning the join are intermolecular."""
    with tempfile.TemporaryDirectory() as d:
        fa, fb, ct = (os.path.join(d, x) for x in ("a.seq", "b.seq", "o.ct"))
        open(fa, "w").write(f";\na\n{a}1\n")
        open(fb, "w").write(f";\nb\n{b}1\n")
        env = dict(os.environ, DATAPATH=DATAPATH)
        if _run([f"{RNASTR}/bifold", fa, fb, ct], env=env, t=300) is None:
            return None
        try:
            lines = [l.split() for l in open(ct).read().splitlines() if l.strip()]
        except Exception:                                                      # noqa: BLE001
            return None
    # bifold writes MANY suboptimal structures into one ct, each preceded by its own header line. Taking
    # every numeric row concatenated 20 structures into one 1200-row block, so the inferred linker length
    # came out as 1143 and no pair could ever span it -- bifold returned an empty set on every single pair
    # and the empty set produces kappa exactly 0.000 against everyone, which reads as total disagreement.
    # Only the first structure is the MFE one.
    total = int(lines[0][0])
    body = [l for l in lines[1:] if len(l) >= 6 and l[0].isdigit()][:total]
    link = total - len(a) - len(b)
    if link < 0 or len(body) < total:
        return None
    pr = set()
    for l in body:
        i, j = int(l[0]) - 1, int(l[4]) - 1
        if j < 0 or i >= j:
            continue
        if i < len(a) and j >= len(a) + link:
            pr.add((i, j - len(a) - link))
    return pr


TOOLS = {"RNAduplex": pairs_rnaduplex, "RNAcofold": pairs_rnacofold,
         "IntaRNA": pairs_intarna, "bifold": pairs_bifold}
SOURCE = {"RNAduplex": "ViennaRNA", "RNAcofold": "ViennaRNA",
          "IntaRNA": "IntaRNA(ViennaRNA 参数)", "bifold": "RNAstructure"}


def work(job):
    tid, a, b = job
    out = {"task_id": tid, "a": a, "b": b}
    for name, fn in TOOLS.items():
        p = fn(a, b)
        if p is not None:
            # Both of this file's parser bugs are now impossible rather than merely fixed: a reused position
            # or an out-of-range index raises, and a parallel duplex raises. The second is the whole reason
            # the P2 verdict nearly came out backwards.
            checked = check_pair_set(p, len(a), len(b), name=f"{name}:{tid}")
            check_duplex_antiparallel(checked, name=f"{name}:{tid}")
            out[name] = [list(x) for x in checked]
        else:
            out[name] = None
    return out


def kappa(x, y):
    """Cohen's kappa on per-position 'this nucleotide is intermolecularly paired'."""
    n = len(x)
    if n == 0:
        return None
    po = sum(1 for u, v in zip(x, y) if u == v) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--src", default=os.path.join(HERE, "kinetic_calibrate_big.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "interaction_entry.json"))
    a = ap.parse_args()
    missing = [n for n, p in (("IntaRNA", INTA), ("bifold", f"{RNASTR}/bifold"),
                              ("RNAduplex", f"{VPK}/RNAduplex"))if not os.path.exists(p)]
    if missing:
        print(f"!! 缺少 {missing} —— 入场检查不能用替代品凑数,退出。")
        return

    # POSITIVE AND NEGATIVE CONTROL ON EVERY TOOL, EVERY RUN. bifold once returned an empty set on every
    # input, and an empty set scores kappa exactly 0.000 against everyone -- which reads as total
    # disagreement, i.e. as P2 passing. A tool that fails either control does not get to vote.
    POS = ("GGGAAACCCUUUGGGAAACCC", "GGGUUUCCC")
    NEG = ("AAAAAAAAAAAAAAAAAAAAA", "AAAAAAAAA")
    controls = {}
    for name, fn in TOOLS.items():
        cr = run_controls(name, fn, POS, NEG,
                          positive_ok=lambda v: v is not None and len(v) >= 4,
                          negative_ok=lambda v: v is not None and len(v) == 0)
        controls[name] = {"positive_ok": cr.positive_ok, "negative_ok": cr.negative_ok,
                          "usable": cr.usable, "note": cr.note}
        print(f"  对照 {name:10s} 正 {cr.positive_ok}  负 {cr.negative_ok}"
              f"{'' if cr.usable else '   <- 不可用,不参与判决'}")
    unusable = [n for n, c in controls.items() if not c["usable"]]
    if unusable:
        print(f"\n!! {unusable} 未通过对照 —— 它们的输出不进入 P2/P3,先修工具再跑。")
        json.dump({"gate": "controls", "controls": controls}, open(a.out, "w"),
                  ensure_ascii=False, indent=1)
        return
    print()

    d = json.load(open(a.src))
    jobs = []
    for r in d["rows"]:
        if r.get("error"):
            continue
        for c in r["candidates"]:
            s = c["seq"]
            if len(s) >= 3 * WINDOW:
                st = len(s) // 3
                win = s[st:st + WINDOW]
                trig = win[::-1].translate(str.maketrans("ACGU", "UGCA"))
                jobs.append((r["task_id"], s, trig))
                break
        if len(jobs) >= a.n:
            break
    print(f"第三个领域入场检查:{len(jobs)} 对(设计序列 + 其内部窗口的反向互补触发链)\n"
          f"  四家:RNAduplex / RNAcofold(ViennaRNA)· IntaRNA(自有代码库,ViennaRNA 参数)· bifold(RNAstructure)\n"
          f"  P2 杀死条件:逐对 kappa 普遍 >= {KAPPA_CONSENSUS} = 这个领域其实有共识,换领域\n", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    R, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            R.append(f.result())
            if n % 25 == 0 or n == len(jobs):
                print(f"    {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    names = list(TOOLS)
    ran = {t: sum(1 for r in R if r[t] is not None) for t in names}
    print(f"\n=== P1 能不能跑 ===")
    for t in names:
        print(f"  {t:10s} {ran[t]}/{len(R)}   来源:{SOURCE[t]}")
    usable = [t for t in names if ran[t] >= 0.9 * len(R)]
    if len(usable) < 3:
        print(f"  !! 只有 {len(usable)} 家能稳定跑 —— P1 不过,换领域")
        json.dump({"gate": "P1", "ran": ran}, open(a.out, "w"), ensure_ascii=False, indent=1)
        return

    print(f"\n=== P2 它们互不一致吗(逐对 Cohen's kappa,逐位点)===")
    ks, jac = {}, {}
    for i, t1 in enumerate(usable):
        for t2 in usable[i + 1:]:
            kk, jj = [], []
            for r in R:
                if r[t1] is None or r[t2] is None:
                    continue
                p1 = {tuple(p) for p in r[t1]}
                p2 = {tuple(p) for p in r[t2]}
                m = len(r["a"]) + len(r["b"])
                v1 = [0] * m
                v2 = [0] * m
                for x, y in p1:
                    v1[x] = 1
                    v1[len(r["a"]) + y] = 1
                for x, y in p2:
                    v2[x] = 1
                    v2[len(r["a"]) + y] = 1
                kv = kappa(v1, v2)
                if kv is not None:
                    kk.append(kv)
                u = p1 | p2
                jj.append(len(p1 & p2) / len(u) if u else 1.0)
            if kk:
                ks[f"{t1}~{t2}"] = sum(kk) / len(kk)
                jac[f"{t1}~{t2}"] = sum(jj) / len(jj)
    for k, v in sorted(ks.items(), key=lambda x: -x[1]):
        same = "  <- 同一软件包" if SOURCE[k.split('~')[0]] == SOURCE[k.split('~')[1]] else ""
        print(f"  {k:24s} kappa={v:+.3f}   逐对精确重合 Jaccard={jac[k]:.3f}{same}")
    hi = [k for k, v in ks.items() if v >= KAPPA_CONSENSUS]
    print(f"\n  kappa >= {KAPPA_CONSENSUS} 的对:{len(hi)}/{len(ks)}  {hi if hi else ''}")
    p2_ok = len(hi) < len(ks)

    print(f"\n=== P3 严格判据(四家给出【完全相同】的相互作用)能交付多少 ===")
    strict = 0
    for r in R:
        ps = [frozenset(map(tuple, r[t])) for t in usable if r[t] is not None]
        if len(ps) == len(usable) and len(set(ps)) == 1 and ps[0]:
            strict += 1
    rate = strict / max(1, len(R))
    print(f"  {strict}/{len(R)} = {rate:.1%}")
    p3_ok = rate < 0.20

    print(f"\n=== 入场判定 ===")
    print(f"  P1 至少 3 家能跑          {'通过' if len(usable) >= 3 else '不过'}({len(usable)} 家)")
    print(f"  P2 互不一致              {'通过' if p2_ok else '不过 —— 这个领域有共识,换领域'}")
    print(f"  P3 严格判据交付不了       {'通过' if p3_ok else '不过 —— 用严格判据即可,不需要标定'}")
    verdict = len(usable) >= 3 and p2_ok and p3_ok
    print(f"  -> {'可以进入,按协议做步骤 1-6' if verdict else '不进入。门槛不放宽,换领域。'}")
    json.dump({"gate": "entry", "controls": controls, "ran": ran, "usable": usable, "kappa": ks, "jaccard": jac,
               "strict_delivery": rate, "P1": len(usable) >= 3, "P2": p2_ok, "P3": p3_ok,
               "enter": bool(verdict), "window": WINDOW, "rows": R},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
