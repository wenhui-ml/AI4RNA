#!/usr/bin/env python3
"""The shipped operating point, re-derived to a standard a referee would accept. Four things were missing.

WHAT THE FIRST CALIBRATION ESTABLISHED, and it is real: the confirmation rule is a declared point in a
declared family, the whole (k, tau) grid is published rather than the flattering cell, and sensitivity is
never reported without a false-accept rate beside it. That is calibration rather than taste.

WHAT IT DID NOT ESTABLISH, audited honestly, because each of these is the first thing a referee would ask:

  1. THE NULL WAS TOO EASY. It is a uniform random ACGU string of matched length. A random string does not
     even fold to the target's NESTED SKELETON, so asking it to fool a rule about CROSSING stems tests
     almost nothing. The false-accept rate a deployed rule actually faces is the rate on near-misses.

  2. THE SPECIFICITY RESTED ON ONE EVENT. 0.4% is 1 of 250. Its exact 95% Clopper-Pearson interval is
     [0.01%, 2.21%] -- a factor of five wide, quoted to one significant figure as though it were settled.

  3. THE THRESHOLD WAS CHOSEN ON THE DATA IT IS REPORTED ON. The grid was scanned over all 250 natives and
     the operating point selected from that scan; no held-out split. Mild, but it is exactly the optimism a
     blind review is designed to catch.

  4. "THREE INDEPENDENT PREDICTORS" WAS NEVER MEASURED. They are independent IMPLEMENTATIONS; they share
     thermodynamic parameters and modelling assumptions. The first calibration's own numbers hint at the
     problem -- k=1 and k=3 have IDENTICAL false-accept rates, which is what correlated judges look like.
     If the grade k means less than it sounds, the certificate has to say so.

THIS SCRIPT FIXES ALL FOUR. Nothing here is a new idea; it is the same rule measured properly.

THE THREE NULLS, weakest to hardest, all judged by the identical rule:

    random      uniform ACGU, matched length                        the original, kept for comparability
    dishuffle   DINUCLEOTIDE-preserving shuffle of the native       preserves composition AND stacking
                (Altschul-Erikson Eulerian walk)                    propensity; destroys the structure
    nested      a sequence designed for the target's PSEUDOKNOT-    the near-miss. It folds the nested part
                FREE reduction, then judged against the FULL target correctly and has no reason whatsoever
                                                                    to form the crossing stems

`nested` is the one that matters. It is the population the rule actually has to discriminate against in
deployment -- something that gets the easy half right -- and no previous number in this project measured it.

THE SPLIT is deterministic and declared before any number is read: sort target ids, even index -> FIT,
odd -> TEST. The grid is scanned on FIT alone; every reported rate is computed on TEST alone. Both halves are
printed so the size of any optimism is visible rather than assumed absent.

EVERY RATE CARRIES AN EXACT 95% CLOPPER-PEARSON INTERVAL. A rate resting on one event and a rate resting on
forty are not the same evidence and must not print the same way.

ORACLE AGREEMENT IS QUANTIFIED, not asserted: pairwise Cohen's kappa on the indicator "this oracle reproduces
every crossing pair", computed on natives and on the hard null. If kappa is high, k is a weaker grade than
the word "independent" implies, and that belongs in the certificate rather than in a footnote.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

TAUS = (1.0, 0.9, 0.8, 0.67, 0.5)
KS = (1, 2, 3)
ORACLES = ("pkiss", "probknot", "shapeknots")
MAX_FALSE_ACCEPT = 0.01        # the selection rule on FIT, declared before the scan


# --------------------------------------------------------------------------- exact interval
def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial interval. Not a normal approximation: at 1/250 the normal one is meaningless."""
    from scipy.stats import beta
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def rate(k: int, n: int):
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "lo95": lo, "hi95": hi}


def fmt(r) -> str:
    if r["n"] == 0:
        return "   n=0"
    return f"{r['rate']:6.1%} [{r['lo95']:5.1%},{r['hi95']:5.1%}] n={r['n']}"


# --------------------------------------------------------------------------- the hard nulls
def dinucleotide_shuffle(seq: str, rng: random.Random, tries: int = 20):
    """Altschul-Erikson: a uniform random Eulerian walk over the dinucleotide multigraph.

    A MONOnucleotide shuffle would preserve only base composition, and stacking energy -- the thing that
    actually decides whether a helix forms -- depends on ADJACENT pairs. Preserving dinucleotide counts is
    what makes this a null about STRUCTURE rather than a null about composition, which is the whole point of
    the exercise: the first null could be dismissed as "you only showed GC-matched strings fail".
    """
    n = len(seq)
    if n < 4:
        return None
    last = seq[-1]
    for _ in range(tries):
        edges = {}
        for a, b in zip(seq, seq[1:]):
            edges.setdefault(a, []).append(b)
        # a random last-edge tree rooted at `last`: every vertex's chosen exit must reach `last`
        verts = list(edges)
        ok = True
        chosen = {}
        for v in verts:
            if v == last:
                continue
            chosen[v] = rng.choice(edges[v])
        for v in list(chosen):                       # every chosen edge must lead to `last`, no cycles
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in chosen:
                    ok = False
                    break
                seen.add(cur)
                cur = chosen[cur]
            if not ok:
                break
        if not ok:
            continue
        for v in verts:                              # shuffle the rest, keep the tree edge last
            rest = list(edges[v])
            if v in chosen:
                rest.remove(chosen[v])
            rng.shuffle(rest)
            edges[v] = rest + ([chosen[v]] if v in chosen else [])
        out, cur = [seq[0]], seq[0]
        for _ in range(n - 1):
            if not edges.get(cur):
                out = None
                break
            nxt = edges[cur].pop(0)
            out.append(nxt)
            cur = nxt
        if out and len(out) == n:
            s = "".join(out)
            if s != seq:
                return s
    return None


def nested_decoy(target: str, tries: int = 12):
    """Design for the target's PSEUDOKNOT-FREE reduction, then judge against the FULL target.

    This is the near-miss the rule must reject: it gets the nested half right by construction and has no
    reason at all to form the crossing stems. If the rule accepts these, `knot_confirmed` is measuring
    "folds roughly like the target" rather than "forms the knot", and the whole verdict is misnamed.
    """
    import RNA
    from rna_harness.benchmarks.rnainvbench import _round_only
    nested = _round_only(target)
    if nested == target or "(" not in nested:
        return None
    for _ in range(tries):
        s, _d = RNA.inverse_fold(None, nested)
        if s and RNA.fold(s)[0] == nested:
            return s
    return None


# --------------------------------------------------------------------------- judging
def judge(item):
    """Per-oracle crossing-pair recall for one (sequence, target). The exact quantity the rule uses."""
    tid, pop, target, seq = item
    from rna_harness.benchmarks.rnainvbench import _round_only, pk_pairs
    from rna_harness.oracles_pk import pkiss_pairs, probknot_pairs, shapeknots_pairs
    cross = pk_pairs(target) - pk_pairs(_round_only(target))
    if not cross or not seq:
        return None
    out = {}
    for name, fn in (("pkiss", pkiss_pairs), ("probknot", probknot_pairs),
                     ("shapeknots", shapeknots_pairs)):
        ps = fn(seq)
        out[name] = None if ps is None else len(ps & cross) / len(cross)
    return {"id": tid, "pop": pop, "n": len(seq), "n_cross": len(cross), "recall": out}


def confirmed(rec, k: int, tau: float) -> bool:
    got = sum(1 for o in ORACLES
              if rec["recall"].get(o) is not None and rec["recall"][o] >= tau)
    return got >= k


def cohen_kappa(a, b):
    """Agreement beyond chance on two binary indicator lists."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=3, help="dinucleotide shuffles per native")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "calibrate_strict.json"))
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("_cal", os.path.join(HERE, "calibrate.py"))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    natives = cal._native_sequences() if hasattr(cal, "_native_sequences") else cal._natives()
    tgt = {t.task_id: t.target for t in load_pseudobase()}
    ids = sorted(i for i in natives if i in tgt)
    # THE SPLIT, declared before any number is read: even index fits, odd index reports.
    fit_ids = set(ids[0::2])
    test_ids = set(ids[1::2])
    print(f"严格标定:{len(ids)} 个有天然序列的靶标 -> FIT {len(fit_ids)} / TEST {len(test_ids)}"
          f"(按 id 排序后偶/奇下标,跑前声明)\n", flush=True)

    if a.analyse_only and os.path.exists(a.out):
        recs = json.load(open(a.out))["records"]
    else:
        rng = random.Random(0)
        items, n_shuf_fail, n_dec_fail = [], 0, 0
        for tid in ids:
            t, s = tgt[tid], natives[tid]
            items.append((tid, "native", t, s))
            for j in range(a.shuffles):
                sh = dinucleotide_shuffle(s, rng)
                if sh is None:
                    n_shuf_fail += 1
                else:
                    items.append((f"{tid}#sh{j}", "dishuffle", t, sh))
            dc = nested_decoy(t)
            if dc is None:
                n_dec_fail += 1
            else:
                items.append((f"{tid}#nested", "nested", t, dc))
        print(f"  待判 {len(items)} 条(洗牌失败 {n_shuf_fail},嵌套诱饵构造失败 {n_dec_fail};两者都报出)",
              flush=True)
        from concurrent.futures import ProcessPoolExecutor
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs = [r for r in ex.map(judge, items) if r]
        print(f"  判完 {len(recs)} 条,{(time.time()-t0)/60:.1f} min\n", flush=True)
        # the original weak null, reused rather than recomputed, so the comparison is like for like
        for r in json.load(open(os.path.join(HERE, "calibrate_null_v2.json"))):
            rec = {"id": f"{r['id']}#rand", "pop": "random", "n": r["n"], "n_cross": None,
                   "recall": {o: (r.get(o) or {}).get("cross_recall") for o in ORACLES}}
            if any(v is not None for v in rec["recall"].values()):
                recs.append(rec)

    base = lambda r: r["id"].split("#")[0]                                        # noqa: E731
    P = {p: [r for r in recs if r["pop"] == p] for p in ("native", "dishuffle", "nested", "random")}
    print(f"  各总体样本数: " + "  ".join(f"{p}={len(v)}" for p, v in P.items()) + "\n")

    def counts(pop, k, tau, which):
        sub = [r for r in P[pop] if base(r["id"]) in which]
        return sum(1 for r in sub if confirmed(r, k, tau)), len(sub)

    # ---- scan the grid on FIT ONLY, by a rule declared before the scan ------------------------------
    print(f"  在 FIT 上扫格点,选择规则(跑前声明):在最难零假设 `nested` 上误收 <= {MAX_FALSE_ACCEPT:.0%} "
          f"的前提下,灵敏度最高者")
    print(f"    {'规则':<14}{'灵敏度(FIT)':>16}{'nested 误收':>16}{'dishuffle 误收':>18}")
    scan, best = [], None
    for tau in TAUS:
        for k in KS:
            sk, sn = counts("native", k, tau, fit_ids)
            nk, nn = counts("nested", k, tau, fit_ids)
            dk, dn = counts("dishuffle", k, tau, fit_ids)
            row = {"k": k, "tau": tau, "sens": rate(sk, sn), "nested": rate(nk, nn),
                   "dishuffle": rate(dk, dn)}
            scan.append(row)
            print(f"    k={k} tau={tau:<6}{fmt(row['sens']):>16}{fmt(row['nested']):>16}"
                  f"{fmt(row['dishuffle']):>18}")
            if nn and (nk / nn) <= MAX_FALSE_ACCEPT:
                if best is None or (sk / sn if sn else 0) > (best["sens"]["rate"] or 0):
                    best = row
    if best is None:
        print("\n  !! FIT 上没有任何 (k, tau) 能在硬零假设下守住误收上限 —— 这条判据不成立,到此为止")
        json.dump({"verdict": "no usable operating point on the hard null", "scan": scan,
                   "records": recs}, open(a.out, "w"), ensure_ascii=False, indent=1)
        return
    K, TAU = best["k"], best["tau"]
    print(f"\n  FIT 选出:k={K}, tau={TAU}   (出厂规则是 k=1, tau=1.0)")

    # ---- report on TEST ONLY -----------------------------------------------------------------------
    out = {"split": {"fit": len(fit_ids), "test": len(test_ids), "rule": "sorted ids, even=fit, odd=test"},
           "selection_rule": f"max sensitivity on FIT s.t. nested false-accept <= {MAX_FALSE_ACCEPT}",
           "chosen": {"k": K, "tau": TAU}, "scan_fit": scan, "test": {}}
    print(f"\n  === TEST 上的报告值(95% Clopper-Pearson 精确区间) ===")
    for label, (k, tau) in (("FIT 选出", (K, TAU)), ("出厂规则 k=1,tau=1.0", (1, 1.0))):
        row = {}
        for pop in ("native", "nested", "dishuffle", "random"):
            which = test_ids if pop != "random" else {base(r["id"]) for r in P["random"]} & test_ids
            kk, nn = counts(pop, k, tau, which)
            row[pop] = rate(kk, nn)
        out["test"][f"k={k},tau={tau}"] = row
        print(f"    [{label}]")
        print(f"      灵敏度(天然)      {fmt(row['native'])}")
        print(f"      误收 nested(最难) {fmt(row['nested'])}   <- 折对嵌套骨架但不该成结")
        print(f"      误收 dishuffle    {fmt(row['dishuffle'])}   <- 保二核苷酸组成的洗牌")
        print(f"      误收 random(旧)   {fmt(row['random'])}")

    # ---- how independent are the three judges, actually ---------------------------------------------
    print(f"\n  三家 oracle 的一致性(Cohen's kappa,指标 = 交叉对全复现):")
    kap = {}
    for pop in ("native", "nested"):
        sub = [r for r in P[pop] if all(r["recall"].get(o) is not None for o in ORACLES)]
        for i in range(3):
            for j in range(i + 1, 3):
                a_, b_ = ORACLES[i], ORACLES[j]
                x = [1 if r["recall"][a_] >= 1.0 else 0 for r in sub]
                y = [1 if r["recall"][b_] >= 1.0 else 0 for r in sub]
                kk = cohen_kappa(x, y)
                kap[f"{pop}:{a_}~{b_}"] = kk
                print(f"    {pop:<10}{a_:<11}~ {b_:<12}kappa={('%.3f' % kk) if kk is not None else 'n/a'}"
                      f"  (n={len(sub)})")
    out["kappa"] = kap
    out["records"] = recs
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
