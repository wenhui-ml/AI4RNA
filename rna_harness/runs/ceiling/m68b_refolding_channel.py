"""M68 arm B: the refolding channel radius excision cannot close.

Excision removes the reagent chemistry AT the mutated positions. It does not remove the
fact that a G/C substitution and an A/U substitution perturb the accessible pairing
ensemble by different amounts, which would raise rho_i and rho_j far from the mutation
site and so raise R through its denominator. This computes that perturbation from the
folding model alone, with no experimental input, and runs it as a third mediator.
Offline, CPU, $0.
"""
import collections, json, math, os, sys, time
import numpy as np
import RNA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m16_rescue as M16
import m21_thermo_mediator as M21

SEED = 20260902
NPERM = 20000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- rebuild, carrying sequences
rows = M16.load()
BLANK = ("", "NA", "nan")
wt = {}
# the corpus stores the two halves of a compensatory pair in DIFFERENT columns: the mutant at
# the 5' position carries mutA alone, the one at the 3' position carries mutB alone. M16 only
# ever needed the first, so indexing singles by mutA alone finds no partner for position b.
singA, singB = collections.defaultdict(dict), collections.defaultdict(dict)
for d in rows:
    k = (d["puzzle"], d["method"])
    A, B = d["mutA"] in BLANK, d["mutB"] in BLANK
    if A and B:
        wt[k] = d
    elif B and not A:
        singA[k][int(float(d["mutA"]))] = d
    elif A and not B:
        singB[k][int(float(d["mutB"]))] = d

recs = []
for d in rows:
    if d["rf"] in ("", "NA", "nan") or d["sn"] not in ("1", "1.0"):
        continue
    k = (d["puzzle"], d["method"]); p = wt.get(k)
    if p is None:
        continue
    ss = int(float(d["ss"])); P = M16.parse_pairs(d["db"])
    try:
        ma, mb = int(float(d["mutA"])), int(float(d["mutB"]))
    except ValueError:
        continue
    a, b = ma + ss - 2, mb + ss - 2
    if P.get(a) != b:
        continue
    s = p["seq"]; pt = "".join(sorted({s[a], s[b]}))
    dep = 0
    while P.get(a - dep - 1) == b + dep + 1:
        dep += 1
    up = 0
    while P.get(a + up + 1) == b - up - 1:
        up += 1
    crossed = any(m < a < n < b or a < m < b < n for m, n in P.items() if m < n)
    recs.append({"puzzle": d["puzzle"], "method": d["method"], "pt": pt, "rf": float(d["rf"]),
                 "depth": min(dep, up), "crossed": bool(crossed), "design": k,
                 "wt_seq": s,
                 "sa": (singA[k].get(ma) or {}).get("seq"),
                 "sb": (singB[k].get(mb) or {}).get("seq"), "db": d["db"]})

# 对拍: the record set must be byte-identical to M21's on every field they share
base = M21.build_with_dg()
key = lambda r: (r["puzzle"], r["method"], r["pt"], round(r["rf"], 6), r["depth"], r["crossed"])
assert len(base) == len(recs), f"record count: M21 {len(base)} vs here {len(recs)}"
assert collections.Counter(map(key, base)) == collections.Counter(map(key, recs)), \
    "record set differs from M21 on a shared field -- that is a bug, not a new arm"
print(f"records: {len(recs)}  (identical to M21's set)")

have = [r for r in recs if r["sa"] and r["sb"]]
print(f"quadruples with BOTH single mutants present: {len(have)} "
      f"({100*len(have)/len(recs):.1f}%)   [K3 reports this fraction]")

# ---------------------------------------------------------------- predicted perturbation
BPP = {}
def bpp_vec(seq):
    if seq in BPP:
        return BPP[seq]
    fc = RNA.fold_compound(seq)
    fc.pf()
    m = fc.bpp()
    n = len(seq)
    v = np.zeros(n * (n + 1) // 2)
    t = 0
    for i in range(1, n + 1):
        row = m[i]
        for j in range(i + 1, n + 1):
            v[t] = row[j]; t += 1
    BPP[seq] = v
    return v

t0 = time.time()
for i, r in enumerate(have):
    w = bpp_vec(r["wt_seq"])
    da = float(np.abs(w - bpp_vec(r["sa"])).sum())
    db_ = float(np.abs(w - bpp_vec(r["sb"])).sum())
    r["pertA"], r["pertB"] = da, db_
    r["pert"] = math.sqrt(da * da + db_ * db_)      # the same form as R's denominator
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(have)}  cached seqs {len(BPP)}  {time.time()-t0:.0f}s")
print(f"folded {len(BPP)} distinct sequences in {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- S4: does it differ by type?
sb = M21.strat_base
P4 = [dict(r, rf=r["pert"]) for r in have]
d4, w4, u4 = M21.matched_diff(P4, "CG", "AU", sb)
null = np.array([M21.perm_diff(P4, "CG", "AU", sb, rng)[0] for _ in range(NPERM)], dtype=float)
p4 = (float((np.abs(null) >= abs(d4)).sum()) + 1) / (NPERM + 1)
print(f"\nS4  predicted perturbation, G-C minus A-U (matched): {d4:+.4f}  "
      f"({int(w4)} pair-weights, {u4} strata, permutation p={p4:.2g} over {NPERM} draws)")

# ---------------------------------------------------------------- S5: as a third mediator
edges = M21.bin_edges([dict(r, dG=r["pert"]) for r in have], 5)
for r in have:
    r["bin"] = M21.assign_bin(r["pert"], edges)
sd = M21.strat_dg
kept = M21.retained(have, "CG", "AU", sd)
base_d, base_w, base_u = M21.matched_diff(kept, "CG", "AU", sb)
med_d, med_w, med_u = M21.matched_diff(kept, "CG", "AU", sd)
ret = med_d / base_d if base_d else float("nan")
print(f"S5  on the {len(kept)} records the stratified estimator uses:")
print(f"      baseline (no perturbation dimension) {base_d:+.4f}  ({base_u} strata)")
print(f"      conditioned on predicted perturbation {med_d:+.4f}  ({med_u} strata)")
print(f"      retained = {100*ret:.1f}%   [S5 asks for >= 70%]")

OUT = dict(seed=SEED, n_records=len(recs), n_quadruples=len(have),
           frac_quadruples=round(len(have)/len(recs), 4), n_distinct_seqs=len(BPP),
           S4=dict(diff=d4, weight=int(w4), strata=u4, p=p4, nperm=NPERM),
           S5=dict(n_kept=len(kept), baseline=base_d, conditioned=med_d, retained=ret,
                   strata_base=base_u, strata_cond=med_u),
           S5_pass=bool(ret >= 0.70), S4_positive=bool(d4 > 0))
json.dump(OUT, open(os.path.join(HERE, "m68b_refolding_channel.json"), "w"), indent=1)
print("\nwrote m68b_refolding_channel.json")
