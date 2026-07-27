#!/usr/bin/env python3
"""R6 owed experiment #1: build a STRUCTURALLY DISJOINT test pool.

The R5/R6 split audit established that the index-based split is not a structural split: 39/125 of the
ID-disjoint targets are exact structural duplicates of a touched target and 94/125 lie within 15% normalized
dot-bracket distance. That is why the paper can only claim an ID-held-out evaluation and not transfer.

This script builds the split the evidence actually calls for, by CLUSTER rather than by index:

  1. cluster ALL 251 Pseudobase++ crossing targets by single linkage at 15% normalized dot-bracket distance
     (the same metric and threshold split_audit.py used, so the two analyses are commensurable);
  2. mark a cluster DIRTY if it contains any target from the touched ids 0-125 -- development [0:60],
     inspected pool2 [60:110], or agent-invention training [110:126];
  3. the test pool is the union of the CLEAN clusters: every one of its targets is >15% away from every
     target that entered any stage of our development, and so is the whole cluster it belongs to.

Assigning whole clusters (not individual targets) is the point: a per-target filter would still admit a
target whose cluster-mate is touched, which is the leak the audit found.

Writes struct_split.json for sweep.py's TARGETS_JSON interface, and prints the audit of the resulting pool so
the disjointness can be checked rather than trusted.

Pure computation over the released target list; no oracle calls, no episodes.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase   # noqa: E402

NEAR = 0.15
TOUCHED = range(0, 126)


def lev_norm(a, b):
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)


def main():
    tasks = load_pseudobase(limit=251)
    idx = {t.task_id: i for i, t in enumerate(tasks)}
    db = {t.task_id: t.target for t in tasks}
    npk = {t.task_id: t.meta.get("n_pk_pairs") for t in tasks}
    ids = [t.task_id for t in tasks]

    # ---- single-linkage clustering over all 251 ----
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            if lev_norm(db[ia], db[ib]) <= NEAR:
                ra, rb = find(ia), find(ib)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    clusters = list(groups.values())

    def touched(cl):
        return any(idx[t] in TOUCHED for t in cl)

    clean = [cl for cl in clusters if not touched(cl)]
    dirty = [cl for cl in clusters if touched(cl)]
    pool = sorted((t for cl in clean for t in cl), key=lambda x: idx[x])

    print(f"all {len(ids)} crossing targets -> {len(clusters)} structural clusters "
          f"(single linkage, normalized dot-bracket d <= {NEAR})")
    print(f"  clusters containing a TOUCHED target (ids 0-125): {len(dirty)} "
          f"({sum(len(c) for c in dirty)} targets)")
    print(f"  CLEAN clusters:                                   {len(clean)} "
          f"({len(pool)} targets)  <- the structurally disjoint test pool")
    print(f"  clean-cluster sizes: {sorted((len(c) for c in clean), reverse=True)}")

    # ---- verify disjointness rather than assume it ----
    tset = [t for t in ids if idx[t] in TOUCHED]
    worst = min(min(lev_norm(db[p], db[q]) for q in tset) for p in pool)
    print(f"\n  VERIFICATION: minimum distance from any test target to any touched target = {worst:.4f}")
    print(f"  (must exceed {NEAR}; a value <= {NEAR} would mean the cluster filter failed)")
    assert worst > NEAR, "structural disjointness violated"
    ov = set(pool) & set(tset)
    assert not ov, f"id overlap with touched set: {ov}"
    print("  no id overlap with the touched set; whole clusters assigned, not individual targets.")

    v = [npk[t] for t in pool if npk.get(t) is not None]
    av = [npk[t] for t in ids if npk.get(t) is not None]
    dv = [npk[t] for t in tset if npk.get(t) is not None]
    print(f"\n  difficulty profile (crossing-pair count):")
    print(f"    test pool     n={len(v):3d}  mean {sum(v)/len(v):.2f}  min/max {min(v)}/{max(v)}")
    print(f"    touched set   n={len(dv):3d}  mean {sum(dv)/len(dv):.2f}  min/max {min(dv)}/{max(dv)}")
    print(f"    all 251       n={len(av):3d}  mean {sum(av)/len(av):.2f}  min/max {min(av)}/{max(av)}")
    print("    -> report this: a structurally disjoint pool is NOT automatically difficulty-matched, and any")
    print("       rate difference against the development pool confounds disjointness with difficulty.")

    out = {"note": "structurally disjoint test pool: union of 15%-single-linkage clusters containing no "
                   "target from ids 0-125 (development, inspected pool2, invention training)",
           "threshold": NEAR, "n_clusters_total": len(clusters), "n_clusters_clean": len(clean),
           "min_dist_to_touched": round(worst, 4),
           "rows": [{"id": t, "target": db[t], "orig_index": idx[t], "n_pk_pairs": npk[t]} for t in pool]}
    p = os.path.join(HERE, "struct_split.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p} ({len(pool)} targets) -- use via TARGETS_JSON with sweep.py")


if __name__ == "__main__":
    main()
