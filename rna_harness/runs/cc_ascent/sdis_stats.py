#!/usr/bin/env python3
"""R6 owed experiment #1, analysis: gamer vs panel operator on the STRUCTURALLY DISJOINT pool.

Reads sdis_e{0,1,2}_s{0..4}/sweep.json -- 27 targets in 24 clean structural clusters, every one >15%
normalized dot-bracket distance from every target that entered any stage of development
(build_struct_split.py), 5 nominal seeds x 3 executions, both arms in the same worker call.

This is the evaluation the split audit said the paper owed. Two things it can settle and one it cannot:

  * it CAN say whether the direction of Delta_gamer survives on structurally novel targets;
  * it CAN report the absolute rates there against the development pool;
  * it CANNOT deliver a well-powered estimate, because Pseudobase++ is structurally exhausted: 251 crossing
    targets collapse to 60 clusters, and after removing every cluster touching ids 0-125 only 27 targets
    remain. At the development-pool base rate (~5%) the expected count is ~1.4/27. We therefore report the
    power limit as a FINDING ABOUT THE BENCHMARK, and use the distribution-free cell-level sign test as the
    primary statistic rather than an interval that would be uninformative at this n.

Offline over the saved sweeps; no episodes, no oracle calls.
"""
import json, math, os, random, statistics as st

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
EXECS = [0, 1, 2]
ARMS = {"autonomous": "gamer (pKiss only)", "panel_auto": "panel operator"}
B = 20000
METRICS = [("r_mix", "all three predictors"), ("r_in", "pKiss AND ProbKnot")]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(0, m + 1)), 1.0)


def load_cell(e, s, need):
    p = os.path.join(HERE, f"sdis_e{e}_s{s}", "sweep.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if len(d.get("rows") or []) < need:
        print(f"  (skipping incomplete cell e{e}/s{s}: {len(d.get('rows') or [])}/{need})")
        return None
    out = {}
    for r in d["rows"]:
        cells, ok = {}, True
        for a in ARMS:
            v = r["arms"].get(a)
            if v is None or v.get("error"):
                ok = False
                break
            pnl = v.get("panel") or {}
            cells[a] = {"r_in": bool(v.get("solved")) and bool(pnl.get("probknot")),
                        "r_mix": bool(pnl.get("unanimous")), "pkiss": bool(v.get("solved"))}
        if ok:
            out[r["id"]] = cells
    return out


def main():
    split = json.load(open(os.path.join(HERE, "struct_split.json")))
    need = len(split["rows"])
    # cluster membership: rebuild from the saved split (clusters were the unit of assignment)
    print(f"structurally disjoint pool: {need} targets in {split['n_clusters_clean']} clean clusters "
          f"(of {split['n_clusters_total']} total); min distance to any touched target = "
          f"{split['min_dist_to_touched']}\n")

    data = {}
    for e in EXECS:
        for s in SEEDS:
            d = load_cell(e, s, need)
            if d:
                data[(s, e)] = d
    if not data:
        raise SystemExit("no sdis sweeps found")
    cells = sorted(data)
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(ids)
    print(f"{n} common targets across {len(cells)} (seed, execution) cells\n")

    for key, desc in METRICS:
        print(f"=== {key} ({desc}) on the STRUCTURALLY DISJOINT pool ===")
        for a in ARMS:
            cnt = [sum(data[c][t][a][key] for t in ids) for c in cells]
            print(f"  {ARMS[a]:22} " + " ".join(f"{v:2d}" for v in cnt) +
                  f"   mean {sum(cnt)/len(cnt):5.2f}/{n}  rate {sum(cnt)/len(cnt)/n:.4f}  "
                  f"sd {st.stdev(cnt):.2f}")
        D = {(c, t): int(data[c][t]["panel_auto"][key]) - int(data[c][t]["autonomous"][key])
             for c in cells for t in ids}
        obs = sum(D.values()) / len(D)
        tb = sum(1 for c in cells for t in ids
                 if data[c][t]["panel_auto"][key] and not data[c][t]["autonomous"][key])
        tc = sum(1 for c in cells for t in ids
                 if data[c][t]["autonomous"][key] and not data[c][t]["panel_auto"][key])
        wins = sum(1 for c in cells
                   if sum(data[c][t]["panel_auto"][key] for t in ids)
                   > sum(data[c][t]["autonomous"][key] for t in ids))
        ties = sum(1 for c in cells
                   if sum(data[c][t]["panel_auto"][key] for t in ids)
                   == sum(data[c][t]["autonomous"][key] for t in ids))
        nz = len(cells) - ties
        psign = min(1.0, 2 * sum(math.comb(nz, k) * 0.5 ** nz for k in range(wins, nz + 1))) if nz else 1.0
        print(f"  paired Delta = {obs:+.4f} ({obs*n:+.2f} targets in {n})")
        print(f"  pooled discordance b={tb} (panel only), c={tc} (gamer only); "
              f"McNemar exact p={mcnemar_exact(tb, tc):.3g}")
        print(f"  cell-level sign test: panel > gamer in {wins}/{len(cells)} executions "
              f"({ties} ties, excluded), two-sided p={psign:.4g}")
        # target-level bootstrap for completeness, flagged as underpowered
        out = []
        for _ in range(B):
            tot = m = 0
            for _ in range(n):
                t = ids[random.randrange(n)]
                for c in cells:
                    tot += D[(c, t)]
                    m += 1
            out.append(tot / m)
        out.sort()
        lo, hi = out[int(0.025 * B)], out[int(0.975 * B)]
        print(f"  target bootstrap 95% CI [{lo:+.4f},{hi:+.4f}]  "
              f"{'clears 0' if lo > 0 else 'SPANS 0'}   (n={n}: underpowered by construction)\n")

    print("Interpretation. The benchmark, not the method, is the binding constraint here: Pseudobase++'s 251\n"
          "crossing targets contain only 24 structural clusters disjoint from our development, so a clean\n"
          "transfer test on this benchmark cannot be well powered at a ~5% base rate. Report the direction and\n"
          "the sign test; do not report the interval as if it settled the question.")


if __name__ == "__main__":
    main()
