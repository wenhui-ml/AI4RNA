"""M66 arms A and C, which the declaration staked and which were still unrun.
Arm A: the corpus's own undesigned control -- a non-agentic RAG baseline on the same
targets. Arm C: does the covariate predict resolution INSIDE a stratum.
Estimator: R1's stratum-weighted matched contrast; null: M64 within-stratum permutation."""
import os, re, sys, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m66_load import load, UNIVERSE

SEED, NPERM = 20260902, 200000
BAD = {"Undisclosed", "NA"}
R = [r for r in load() if not (isinstance(r["info_resolved"], (int, float))
     and abs(r["info_resolved"] - 100.0*len(r["resolved"])/UNIVERSE) > 0.25)]
R = [r for r in R if r["model"] not in BAD and not r["model"].startswith("http")]
for r in R: r["rate"] = len(r["resolved"])/UNIVERSE

RAG = "RAG baseline"
rag_models = {r["model"] for r in R if r["agent"] == RAG}
print(f"non-agentic RAG-baseline arm: {len(rag_models)} models -> {sorted(rag_models)}")
shared = sorted(m for m in rag_models if any(r["model"] == m and r["agent"] != RAG for r in R))
print(f"models measured BOTH non-agentically and agentically: {len(shared)} -> {shared}\n")

# universe of instances = the 500; build 0/1 vectors
allinst = sorted({i for r in R for i in r["resolved"]})
idx = {v: k for k, v in enumerate(allinst)}
def vec(r):
    z = np.zeros(len(allinst), dtype=np.int8)
    for i in r["resolved"]: z[idx[i]] = 1
    return z

rows = []   # (stratum=(model,instance), treatment, outcome)
for m in shared:
    rb = max((r for r in R if r["model"] == m and r["agent"] == RAG), key=lambda r: r["rate"])
    ags = [r for r in R if r["model"] == m and r["agent"] != RAG]
    ab = max(ags, key=lambda r: r["rate"])
    A, B = vec(ab), vec(rb)            # A = agentic, B = non-agentic
    rows.append((m, ab["agent"], A, B, ab["rate"], rb["rate"]))
    b, c = int(((A == 1) & (B == 0)).sum()), int(((A == 0) & (B == 1)).sum())
    print(f"  {m[:30]:30s} agentic {ab['agent'][:16]:16s} {ab['rate']*100:5.1f}%  vs "
          f"RAG {rb['rate']*100:5.1f}%   b={b:3d} c={c:3d}")

if rows:
    # matched contrast: strata are (model, instance); weight = min(n_A,n_B) = 1 in every stratum
    diffs = np.concatenate([ (A.astype(float) - B.astype(float)) for _, _, A, B, _, _ in rows ])
    d = float(diffs.mean())
    rng = np.random.default_rng(SEED)
    # within-stratum label permutation: each stratum holds one A and one B, so a permutation
    # flips the sign of that stratum's difference. That is the exact M64 null here.
    n = diffs.size
    sgn = rng.integers(0, 2, size=(NPERM, n)).astype(np.int8) * 2 - 1
    null = (sgn * diffs).mean(axis=1)
    p = float((np.abs(null) >= abs(d)).sum() + 1) / (NPERM + 1)
    lo, hi = np.percentile(np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(20000)]), [2.5, 97.5])
    print(f"\n  arm A matched contrast (agentic minus non-agentic), strata = model x instance")
    print(f"    d = {d*100:+.2f} points   n_strata = {n}   permutation p = {p:.2g} "
          f"({NPERM:,} draws)   bootstrap CI [{lo*100:+.2f}, {hi*100:+.2f}]")
    print(f"    arm A verdict: {'SEPARATED at 0.005' if p <= 0.005 else 'not separated'}")

    # arm C: does the covariate predict resolution inside a stratum, beyond the model?
    # Restricted to the strata arm A identifies, this IS arm C's question.
    print(f"\n  arm C is answered on the same strata: knowing the scaffold changes the predicted")
    print(f"  outcome by {abs(d)*100:.2f} points within a fixed (model, target) stratum.")
    out = dict(rag_models=len(rag_models), shared_models=shared, n_strata=int(n),
               d_points=float(d*100), p=p, nperm=NPERM, ci_points=[float(lo*100), float(hi*100)],
               per_model=[dict(model=m, agentic=a, rate_agentic=ra, rate_rag=rb_,
                               b=int(((A==1)&(B==0)).sum()), c=int(((A==0)&(B==1)).sum()))
                          for m, a, A, B, ra, rb_ in rows], seed=SEED)
    json.dump(out, open("/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/m66_armA.json","w"), indent=1)
    print("\nwrote m66_armA.json")
else:
    print("arm A NOT IDENTIFIED: no model measured both ways")
