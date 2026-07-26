#!/usr/bin/env python3
"""R5 §4.3: is the index-based train/test split actually a STRUCTURAL split?

Splitting Pseudobase++ by target index guarantees no target is reused, but it does NOT guarantee the
held-out targets are structurally distant from the development/training ones. If the untouched targets have
near-duplicate neighbours in the touched set, "held-out transfer" could be template-neighbour
generalization. This script audits that directly.

For each STRICTLY-UNTOUCHED target (ids 126-250) we compute its nearest neighbour among the TOUCHED targets
(development [0:60] + inspected pool2 [60:110] + agent-invention training [110:126]) under three distances:
  * normalized dot-bracket (round-only + crossing-aware) Levenshtein distance,
  * base-pair-set Jaccard distance on the crossing pairs,
  * sequence identity of the IUPAC restraint strings (a homology proxy; Pseudobase++ ships restraints).
We then report the nearest-neighbour distribution, how many untouched targets have a near-duplicate
(NN distance below a threshold), and whether the panel operator's SUCCESSES concentrate on the
near-duplicate subset -- the failure mode that would make the transfer claim vacuous.

Pure computation over the released target lists; no oracle calls, no episodes.
"""
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase, pk_pairs, _round_only

HERE = os.path.dirname(os.path.abspath(__file__))
TOUCHED = range(0, 126)        # dev [0:60] + inspected pool2 [60:110] + agent training [110:126]
UNTOUCHED = range(126, 251)    # the strictly-untouched confirmatory pool
NEAR = 0.15                    # "near-duplicate": normalized edit distance <= 15%


def lev_norm(a, b):
    """Levenshtein distance normalized by max length (0 = identical, 1 = maximally different)."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)


def jaccard_dist(pa, pb):
    if not pa and not pb:
        return 0.0
    return 1.0 - len(pa & pb) / len(pa | pb)


def crossing_pairs(db):
    return pk_pairs(db) - pk_pairs(_round_only(db))


def main():
    tasks = load_pseudobase(limit=251)
    by_idx = {}
    for k, t in enumerate(tasks):
        by_idx[k] = t
    touched = [i for i in TOUCHED if i in by_idx]
    untouched = [i for i in UNTOUCHED if i in by_idx]
    print(f"split audit: {len(touched)} touched (ids 0-125) vs {len(untouched)} strictly-untouched (126-250)\n")

    rows = []
    for u in untouched:
        tu = by_idx[u]
        best = (2.0, None, None, None)
        for c in touched:
            tc = by_idx[c]
            d_db = lev_norm(tu.target, tc.target)
            if d_db < best[0]:
                d_bp = jaccard_dist(crossing_pairs(tu.target), crossing_pairs(tc.target))
                d_sq = lev_norm(tu.meta.get("seq_restr") or "", tc.meta.get("seq_restr") or "")
                best = (d_db, c, d_bp, d_sq)
        rows.append({"id": u, "nn_touched": best[1], "nn_db_dist": round(best[0], 4),
                     "nn_bp_jaccard": round(best[2], 4) if best[2] is not None else None,
                     "nn_restraint_dist": round(best[3], 4) if best[3] is not None else None,
                     "len": tu.n})

    ds = sorted(r["nn_db_dist"] for r in rows)
    n = len(ds)

    def q(p):
        return ds[min(n - 1, int(p * n))]
    print("nearest-neighbour normalized dot-bracket distance, untouched -> touched:")
    print(f"  min {ds[0]:.3f}  q10 {q(.10):.3f}  median {q(.50):.3f}  q90 {q(.90):.3f}  max {ds[-1]:.3f}")
    exact = sum(1 for d in ds if d == 0.0)
    near = sum(1 for d in ds if d <= NEAR)
    print(f"  exact structural duplicates of a touched target: {exact}/{n}")
    print(f"  near-duplicates (NN distance <= {NEAR}):          {near}/{n}")
    bpj = sorted(r["nn_bp_jaccard"] for r in rows if r["nn_bp_jaccard"] is not None)
    if bpj:
        print(f"  crossing-pair-set Jaccard distance to that NN: median {bpj[len(bpj)//2]:.3f}, min {bpj[0]:.3f}")
    sqd = [r["nn_restraint_dist"] for r in rows if r["nn_restraint_dist"] is not None]
    if sqd:
        sqd.sort()
        print(f"  IUPAC-restraint distance to that NN:           median {sqd[len(sqd)//2]:.3f}, min {sqd[0]:.3f}")

    # Do the operator's successes concentrate on the near-duplicate subset?
    nearset = {f"pseudobase:{r['id']}" for r in rows if r["nn_db_dist"] <= NEAR}
    farset = {f"pseudobase:{r['id']}" for r in rows if r["nn_db_dist"] > NEAR}
    print(f"\nsuccess concentration on near-duplicates (panel operator, bigpool seeds 0-2):")
    tot_near = tot_far = 0
    for s in (0, 1, 2):
        p = os.path.join(HERE, f"bigpool_s{s}", "sweep.json")
        if not os.path.exists(p):
            print(f"  seed {s}: bigpool_s{s}/sweep.json missing"); continue
        d = json.load(open(p))
        hit_near = hit_far = 0
        for r in d["rows"]:
            pa = r["arms"].get("panel_auto") or {}
            if (pa.get("panel") or {}).get("unanimous"):
                if r["id"] in nearset:
                    hit_near += 1
                elif r["id"] in farset:
                    hit_far += 1
        tot_near += hit_near; tot_far += hit_far
        print(f"  seed {s}: r_mix successes on near-dup {hit_near}/{len(nearset)}, "
              f"on structurally-distant {hit_far}/{len(farset)}")
    if len(nearset) and len(farset):
        rn, rf = tot_near / (3 * len(nearset)), tot_far / (3 * len(farset))
        print(f"  pooled rate: near-dup {rn:.4f} vs distant {rf:.4f}  "
              f"(ratio {rn/rf:.2f}x)" if rf else "  pooled: distant rate 0")

    json.dump({"near_threshold": NEAR, "n_untouched": n, "n_exact_dup": exact, "n_near_dup": near,
               "rows": rows}, open(os.path.join(HERE, "split_audit.json"), "w"), indent=1)
    print(f"\n-> split_audit.json")


if __name__ == "__main__":
    main()
