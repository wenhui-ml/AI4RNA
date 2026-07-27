#!/usr/bin/env python3
"""Read the rest of the paper in the Delta coordinate (pk_margin.py, extended to every pool and arm).

pk_margin.py showed that Delta_bp -- how close the best of 60 merely-COMPATIBLE random sequences lands,
with no optimizer -- predicts our pKiss failures at AUC 0.824 and carries 84% of Delta_gamer. Two questions
the paper still answers by assertion, and which the same probe settles:

  (1) Are the pools difficulty-matched? Sec. 5 says the structurally disjoint pool is "not automatically
      difficulty-matched" and that the cross-source 0/60 should not be read as absence of capability.
      Both are testable now: if the sdis pool is easier, its 6.9% against the development 5.3% means less
      than it appears; if the bpRNA pool is simply harder, the 0/60 needs no recall argument at all.

  (2) Does the POSITIVE CONTROL survive the same scrutiny we applied to ourselves? We bounded our own
      Delta_gamer by showing 84% of it sits on zero-margin targets. The agent-written operators beat us
      4x, and fairness requires putting their wins on the same axis rather than only our own.

Search-free throughout: random compatible sequences, folded once each.
"""
import json, math, os, random, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase          # noqa: E402
from rna_harness.oracles_pk import pkiss_pairs                          # noqa: E402

WC = [('A','U'),('U','A'),('G','C'),('C','G'),('G','U'),('U','G')]
NSAMP = int(os.environ.get("NSAMP", "60"))
OP = {'(': ')', '[': ']', '{': '}', '<': '>'}
CL = {v: k for k, v in OP.items()}


def pk_pairmap(db):
    st = {k: [] for k in OP}; p = {}
    for i, c in enumerate(db):
        if c in OP:
            st[c].append(i)
        elif c in CL:
            j = st[CL[c]].pop(); p[j] = i; p[i] = j
    return p


def probe(args):
    tid, db, seed = args
    rng = random.Random(seed); p = pk_pairmap(db); N = len(db)
    tgt = {(i, j) for i, j in p.items() if i < j}
    if not tgt:
        return tid, None, 0
    best = len(tgt) * 2
    for _ in range(NSAMP):
        s = ['A'] * N
        for i in range(N):
            if i not in p:
                s[i] = rng.choice('ACGU')
        for i, j in p.items():
            if i < j:
                a, b = rng.choice(WC); s[i] = a; s[j] = b
        got = pkiss_pairs(''.join(s))
        if got is None:
            continue
        best = min(best, len({(min(a, b), max(a, b)) for a, b in got} ^ tgt))
    return tid, best / len(tgt), len(tgt)


def run(pool):
    with mp.get_context("fork").Pool(24) as po:
        return {t: (d, n) for t, d, n in po.map(probe, pool)}


def med(v):
    v = sorted(v); return v[len(v) // 2] if v else float('nan')


def main():
    out = {}
    # ---- pools ----
    dev = [(t.task_id, t.target, 1000 + k) for k, t in enumerate(load_pseudobase(limit=60))]
    idj = [(t.task_id, t.target, 3000 + k) for k, t in enumerate(load_pseudobase(limit=251))][126:251]
    sd = json.load(open(os.path.join(HERE, "struct_split.json")))["rows"]
    sdis = [(r["id"], r["target"], 5000 + k) for k, r in enumerate(sd)]
    bp = os.path.join(HERE, "bprna_pk", "targets.json")
    xsrc = []
    if os.path.exists(bp):
        j = json.load(open(bp))
        rows = j if isinstance(j, list) else j.get("rows", [])
        xsrc = [(r.get("id", f"bprna:{k}"), r["target"], 7000 + k) for k, r in enumerate(rows)]

    print("=== (1) are the pools difficulty-matched on the search-free coordinate? ===")
    print(f"{'pool':34} {'n':>4} {'median Delta_norm':>18} {'frac Delta=0':>13}")
    res = {}
    for name, pool in (("development/evaluation (0-59)", dev), ("ID-disjoint (126-250)", idj),
                       ("structurally disjoint (24 clusters)", sdis), ("cross-source bpRNA", xsrc)):
        if not pool:
            continue
        r = run(pool); res[name] = r
        ds = [d for d, _ in r.values() if d is not None]
        print(f"{name:34} {len(ds):4d} {med(ds):18.3f} {sum(1 for d in ds if d == 0)/len(ds):13.2f}")
        out[name] = {t: v[0] for t, v in r.items()}

    # ---- (2) the positive control on the same axis ----
    print("\n=== (2) do the agent-written operators also win mostly on zero-margin targets? ===")
    devm = {t: v[0] for t, v in res["development/evaluation (0-59)"].items()}
    import glob
    for op in ("gemini31pro", "kimik3"):
        wins = {}
        cells = 0
        for f in sorted(glob.glob(os.path.join(HERE, f"arep_{op}_dev", "s*_e*.json"))):
            cells += 1
            for r in json.load(open(f))["rows"]:
                if r.get("error"):
                    continue
                if r.get("pkiss") and r.get("probknot") and r.get("shapeknots"):
                    wins[r["id"]] = wins.get(r["id"], 0) + 1
        z = sum(v for t, v in wins.items() if devm.get(t) == 0)
        nz = sum(v for t, v in wins.items() if devm.get(t, 1) > 0)
        print(f"  {op:12} {cells} cells | r_mix wins on Delta=0 targets {z:4d}, on Delta>0 targets {nz:4d}"
              f"  -> {100*z/max(1,z+nz):3.0f}% on zero-margin")
    # our own panel operator, same axis, for comparison
    pj = json.load(open(os.path.join(HERE, "pk_margin.json")))["rows"]
    z = sum(r["panel_mix"] for r in pj if r["delta_norm"] == 0)
    nz = sum(r["panel_mix"] for r in pj if r["delta_norm"] > 0)
    print(f"  {'panel operator':12} 15 cells | r_mix wins on Delta=0 {z:4d}, on Delta>0 {nz:4d}"
          f"  -> {100*z/max(1,z+nz):3.0f}% on zero-margin")
    json.dump(out, open(os.path.join(HERE, "pool_margin.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
