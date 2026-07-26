#!/usr/bin/env python3
"""R6 Major Issue 2 + data items 1-2: CLUSTER-AWARE uncertainty, and the near-duplicate contrast with an
interval.

Two consequences of the R5 structural split audit were left unpropagated:

  (1) Pseudobase++ contains exact and near structural duplicates. A TARGET-level bootstrap treats them as
      independent draws and therefore UNDERSTATES uncertainty. The fix is to make the STRUCTURAL CLUSTER the
      resampling unit: cluster the pool by normalized dot-bracket distance (single linkage at the same 15%
      threshold the split audit used), then resample clusters with replacement, each drawn cluster
      contributing all of its targets. This is done here for the headline Delta_gamer.

  (2) The R5 revision reported that the panel operator's success rate on the near-duplicate subset of the
      125 untouched targets (0.075) was not higher than on the structurally distant subset (0.097), but gave
      only the ratio -- no interval, no separation of exact duplicates from 15%-neighbours, and no difficulty
      control. This script reports Delta_duplicate = p_distant - p_near with a cluster-aware interval, splits
      exact duplicates from near-but-not-exact ones, and adds a crossing-pair-count difficulty profile per
      stratum so the reader can see whether the strata are comparable at all.

Pure offline computation over the saved sweeps and the released target list; no oracle calls, no episodes.
"""
import json, os, random, statistics as st, sys

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase   # noqa: E402

IDS = [f"pseudobase:{i}" for i in range(60)]
NEAR = 0.15
B = 20000


# ----------------------------------------------------------------- structural clustering
def lev_norm(a, b):
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


def single_linkage(ids, db, thresh=NEAR):
    """Union-find single-linkage clustering at `thresh` normalized dot-bracket distance."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            if lev_norm(db[ia], db[ib]) <= thresh:
                ra, rb = find(ia), find(ib)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ----------------------------------------------------------------- (1) cluster-robust Delta_gamer
def _uni(rows, arm=None):
    out = {}
    for r in rows:
        u = (((r.get("arms", {}) or {}).get(arm) or {}).get("panel") or {}).get("unanimous") \
            if arm else r.get("unanimous")
        out[r["id"]] = 1 if u else 0
    return out


def load_primary():
    PA = {0: _uni(json.load(open(os.path.join(HERE, "panel_auto/panel_panel_auto.json")))["rows"]),
          1: _uni(json.load(open(os.path.join(HERE, "seed1/panel/panel_panel_auto.json")))["rows"]),
          2: _uni(json.load(open(os.path.join(HERE, "seed2/panel/panel_panel_auto.json")))["rows"]),
          3: _uni(json.load(open(os.path.join(HERE, "seed3/sweep.json")))["rows"], "panel_auto"),
          4: _uni(json.load(open(os.path.join(HERE, "seed4/sweep.json")))["rows"], "panel_auto")}
    GA = {s: _uni(json.load(open(os.path.join(HERE, f"gamer_s{s}/sweep.json")))["rows"], "autonomous")
          for s in range(5)}
    return PA, GA


def cluster_delta(clusters, PA, GA, seeds_too):
    """Bootstrap Delta = p_panel - p_gamer resampling CLUSTERS (and optionally seeds)."""
    S = list(range(5))
    D = {(s, t): PA[s].get(t, 0) - GA[s].get(t, 0) for s in S for t in IDS}
    obs = sum(D.values()) / len(D)
    K = len(clusters)
    out = []
    for _ in range(B):
        ss = [S[random.randrange(len(S))] for _ in range(len(S))] if seeds_too else S
        tot = n = 0
        for _ in range(K):
            cl = clusters[random.randrange(K)]
            for t in cl:
                for s in ss:
                    tot += D[(s, t)]
                    n += 1
        out.append(tot / n)
    out.sort()
    return obs, out[int(0.025 * B)], out[int(0.975 * B)]


# ----------------------------------------------------------------- (2) near-duplicate contrast
def duplicate_contrast(audit, npk):
    """Delta_duplicate = p_distant - p_near on the 125 untouched targets, with a cluster-aware interval."""
    big = {}
    for s in (0, 1, 2):
        d = json.load(open(os.path.join(HERE, f"bigpool_s{s}", "sweep.json")))
        big[s] = _uni(d["rows"], "panel_auto")
    rows = {r["id"]: r for r in audit["rows"]}          # id is an int here
    strata = {"exact": [], "near": [], "distant": []}
    for i in range(126, 251):
        r = rows.get(i)
        if r is None:
            continue
        d = r["nn_db_dist"]
        k = "exact" if d == 0.0 else ("near" if d <= NEAR else "distant")
        strata[k].append(f"pseudobase:{i}")

    def rate(ids):
        num = den = 0
        for t in ids:
            for s in (0, 1, 2):
                if t in big[s]:
                    num += big[s][t]
                    den += 1
        return num, den

    print("=== (2) near-duplicate contrast on the 125 untouched targets, with intervals ===")
    print(f"  stratum definition: nearest-neighbour normalized dot-bracket distance to the TOUCHED set")
    for k in ("exact", "near", "distant"):
        num, den = rate(strata[k])
        v = [npk[t] for t in strata[k] if npk.get(t) is not None]
        prof = f"mean crossing pairs={sum(v)/len(v):.2f}" if v else "n/a"
        print(f"  {k:8s} (d{'=0' if k=='exact' else ('<=.15' if k=='near' else '>.15')}): "
              f"{len(strata[k]):3d} targets, panel-operator r_mix {num}/{den} = "
              f"{(num/den if den else float('nan')):.4f}   {prof}")

    # cluster-aware paired-free bootstrap of the difference distant - (exact+near)
    nearset = strata["exact"] + strata["near"]
    dist = strata["distant"]
    # cluster within each stratum by structure, so duplicates inside a stratum are one unit
    tasks = {t.task_id: t.target for t in load_pseudobase(limit=251)}
    cl_near = single_linkage(nearset, tasks)
    cl_dist = single_linkage(dist, tasks)

    def strat_rate(clusters, draw):
        tot = n = 0
        for _ in range(len(clusters)):
            cl = clusters[random.randrange(len(clusters))] if draw else None
            for t in (cl or []):
                for s in (0, 1, 2):
                    tot += big[s].get(t, 0)
                    n += 1
        return tot / n if n else float("nan")

    nn, dn = rate(nearset), rate(dist)
    p_near, p_dist = nn[0] / nn[1], dn[0] / dn[1]
    obs = p_dist - p_near
    draws = sorted(strat_rate(cl_dist, True) - strat_rate(cl_near, True) for _ in range(B))
    lo, hi = draws[int(0.025 * B)], draws[int(0.975 * B)]
    print(f"\n  near-or-exact (n={len(nearset)}, {len(cl_near)} structural clusters): "
          f"p = {p_near:.4f}")
    print(f"  distant       (n={len(dist)}, {len(cl_dist)} structural clusters): p = {p_dist:.4f}")
    print(f"  Delta_duplicate = p_distant - p_near = {obs:+.4f}  "
          f"cluster bootstrap 95% CI [{lo:+.4f},{hi:+.4f}]")
    print(f"  -> the interval {'EXCLUDES' if lo > 0 or hi < 0 else 'SPANS'} zero: "
          f"{'a difference is established' if (lo > 0 or hi < 0) else 'this sample can exclude neither a substantial positive nor a substantial negative difference, so the correct statement is that no CONCENTRATION of success on near-duplicates was OBSERVED, not that the failure mode is absent'}")


def main():
    tasks = {t.task_id: t.target for t in load_pseudobase(limit=251)}
    npk = {t.task_id: t.meta.get("n_pk_pairs") for t in load_pseudobase(limit=251)}

    print("=== (1) structural clustering of the primary n=60 pool ===")
    clusters = single_linkage(IDS, tasks)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    singl = sum(1 for c in clusters if len(c) == 1)
    print(f"  single linkage at normalized dot-bracket distance <= {NEAR}")
    print(f"  {len(IDS)} targets -> {len(clusters)} clusters "
          f"({singl} singletons; largest {sizes[0]}; sizes {sizes[:12]}{'...' if len(sizes) > 12 else ''})")
    print(f"  effective sample size for a cluster bootstrap is {len(clusters)}, not {len(IDS)}.\n")

    PA, GA = load_primary()
    print("=== cluster-robust interval for the headline Delta_gamer (5 seeds, n=60) ===")
    for label, cl, seeds_too in (
            ("target-level, seeds fixed", [[t] for t in IDS], False),
            ("target-level + seeds (two-way)", [[t] for t in IDS], True),
            ("CLUSTER-level, seeds fixed", clusters, False),
            ("CLUSTER-level + seeds", clusters, True)):
        obs, lo, hi = cluster_delta(cl, PA, GA, seeds_too)
        print(f"  {label:32s} Delta={obs:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
              f"{'clears 0' if lo > 0 else 'SPANS 0'}")
    print()

    audit = json.load(open(os.path.join(HERE, "split_audit.json")))
    duplicate_contrast(audit, npk)


if __name__ == "__main__":
    main()
