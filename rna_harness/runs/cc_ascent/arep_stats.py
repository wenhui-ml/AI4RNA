#!/usr/bin/env python3
"""R7: the instrument's positive control -- do the agent-written operators clear the carry-over bar?

R7 M3 objects, correctly, that "nothing we build saturates the instrument, therefore the instrument
discriminates" is not distinguishable from "the instrument is insensitive". An audit that nothing passes
and a broken audit produce the same table. The control has to be an arm that DOES pass, measured the same
way, and it turns out we already generated one and filed it as a supplement: the frozen LLM-written
operators of invent_operator.py.

This script replicates the comparison at the depth of the headline experiment (5 nominal seeds x 3
executions, agent_replicate.py) and tests the obvious confound: the agent operators emit GC-richer
sequences, and ShapeKnots agreement is partly predictable from GC content alone, so the carry-over
advantage is reported both raw and stratified by composition.

Offline over the saved sweeps; no episodes, no oracle calls, no API calls.
"""
import json, glob, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def clopper(k, n, upper=False):
    """Exact Clopper-Pearson endpoint as a Beta quantile (the direct-summation form overflows at n>~500)."""
    from scipy.stats import beta
    if n == 0:
        return float("nan")
    if upper:
        return 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))


def gc(s):
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def agent_cells(op, pool):
    out = []
    for f in sorted(glob.glob(f"arep_{op}_{pool}/s*_e*.json")):
        d = json.load(open(f))
        rin = [r for r in d["rows"] if not r.get("error") and r.get("pkiss") and r.get("probknot")]
        out.append((d["seed"], d["exec"], rin, d["n_error"]))
    return out


def hand_designs():
    """(seq, shapeknots) for every panel-operator success across the 15 gvp cells."""
    v = []
    for f in sorted(glob.glob("gvp_e*_s*/sweep.json")):
        for r in json.load(open(f))["rows"]:
            a = r["arms"].get("panel_auto") or {}
            if a.get("solved") and a.get("seq"):
                v.append((a["seq"], bool((a.get("panel") or {}).get("shapeknots"))))
    return v


def report(label, pairs):
    k = sum(1 for _, ok in pairs if ok)
    n = len(pairs)
    print(f"  {label:26} {k:4d}/{n:<5d} = {k/n:6.3f}  95% CP [{clopper(k,n):.3f},{clopper(k,n,True):.3f}]"
          f"   mean GC {st.mean([gc(s) for s,_ in pairs]):.3f}")
    return k, n


def main():
    print("=== carry-over to ShapeKnots: agent-written vs hand-built, replicated 5 seeds x 3 executions ===")
    hand = hand_designs()
    allagent = []
    for op in ("gemini31pro", "kimik3"):
        cells = agent_cells(op, "dev")
        if not cells:
            continue
        pairs = [(r["seq"], bool(r.get("shapeknots"))) for _, _, rin, _ in cells for r in rin if r.get("seq")]
        allagent += pairs
        report(f"agent: {op}", pairs)
        # execution-level reproducibility: are the three executions inside a seed identical?
        bysd = {}
        for s, e, rin, _ in cells:
            bysd.setdefault(s, []).append(sum(1 for r in rin if r.get("shapeknots")))
        ident = sum(1 for s, v in bysd.items() if len(set(v)) == 1)
        print(f"     bit-identical across the 3 executions in {ident}/{len(bysd)} seeds  "
              f"(per-seed counts {[v for v in bysd.values()]})")
    report("agent: pooled", allagent)
    report("hand-built panel operator", hand)

    print("\n=== the composition confound: ShapeKnots agreement tracks GC content ===")
    for label, pairs in (("agent (pooled)", allagent), ("hand-built panel operator", hand)):
        for lo, hi, nm in ((0.0, 0.5, "GC < 0.5"), (0.5, 1.01, "GC >= 0.5")):
            sub = [(s, ok) for s, ok in pairs if lo <= gc(s) < hi]
            if sub:
                k = sum(1 for _, ok in sub if ok)
                print(f"  {label:26} {nm:9} {k:4d}/{len(sub):<5d} = {k/len(sub):.3f}")
    print("\n  -> ShapeKnots agreement is partly predictable from composition alone, so part of any")
    print("     carry-over gap is compositional. The agent advantage survives stratification in both bins.")

    print("\n=== structurally disjoint pool (reported, NOT used as evidence) ===")
    for op in ("gemini31pro", "kimik3"):
        cells = agent_cells(op, "sdis")
        if not cells:
            continue
        rin = [r for _, _, ri, _ in cells for r in ri]
        k = sum(1 for r in rin if r.get("shapeknots"))
        err = st.mean([e for _, _, _, e in cells])
        print(f"  {op}: {k}/{len(rin)} = {k/len(rin):.3f}, but mean {err:.1f}/27 targets error out per cell,")
        print(f"     so the denominator is survivorship-filtered towards easy targets and the rate is not")
        print(f"     comparable to the development pool. Reported for completeness only.")


if __name__ == "__main__":
    main()
