#!/usr/bin/env python3
"""R6 Major Issue 1 (+ Major Issue 4): EXECUTION-LEVEL REPLICATION AND A FULLY SPECIFIED BOOTSTRAP FOR THE
HEADLINE COMPARISON Delta_gamer = p_panel - p_gamer.

Reads gvp_e{0,1,2}_s{0,1,2}/sweep.json -- the two arms of the headline comparison (autonomous = pKiss-only
"gamer", panel_auto = two-oracle panel operator) run on the SAME 60 targets, at the SAME 3 nominal seeds, with
3 independent executions per nominal seed, BOTH ARMS EVALUATED INSIDE THE SAME WORKER CALL so the pairing holds
at the (target, seed, execution) level.

BOOTSTRAP SPECIFICATION (stated in full because the R5 revision did not).
  Estimand.  Delta = mean over (target, seed, execution) of [ 1{panel solves} - 1{gamer solves} ].
  Pairing.   The unit of analysis is the PAIRED DIFFERENCE D[(s,e,t)] = panel - gamer within one cell. Both
             arms therefore share, by construction, identical resampling weights at every level; an arm can
             never be resampled against a different target/seed/execution than its counterpart.
  Level 1 (nominal seed).  Draw |S|=5 nominal seeds with replacement from {0,1,2,3,4}.
  Level 2 (execution, NESTED within nominal seed).  Executions are not crossed with seeds -- execution e of
             seed 0 has nothing to do with execution e of seed 1 -- so they are resampled WITHIN each drawn
             seed. Two nesting variants are reported:
               (a) "full-nest": within each drawn seed draw |E|=3 executions with replacement, giving 9 cells
                   per replicate. This matches the design that was actually run and is the primary interval.
               (b) "one-exec": within each drawn seed draw 1 execution, giving 3 cells per replicate. This
                   simulates a study with a single execution per seed and is therefore WIDER; it is the
                   variant the R5 panel-ablation table used, and is reported as a conservative sensitivity.
  Level 3 (target).  Draw n targets with replacement, SYNCHRONOUSLY for every drawn cell and both arms, so a
             target enters or leaves the replicate for the whole comparison at once.
  Structural duplicates.  Pseudobase++ contains exact and near structural duplicates, so a target-level draw
             overstates the effective sample size. We therefore ALSO report every interval with the STRUCTURAL
             CLUSTER as the level-3 unit (single linkage at 15% normalized dot-bracket distance, the same
             threshold as split_audit.py); a drawn cluster contributes all of its targets.
  Interval type.  Percentile (2.5/97.5) of B=20,000 replicates. NOT BCa and not studentized: with only 3
             nominal seeds the resampling distribution is coarse, so we report these as SENSITIVITY INTERVALS
             and do not attach a stable frequentist coverage claim to them.
  RNG.  Python `random`, Mersenne Twister, seeded at 20260726 at module import (below); a rerun of this file
        reproduces every number in the paper's Table for this comparison.

Offline over the saved sweeps; no episodes, no oracle calls.
"""
import json, math, os, random, statistics as st, sys

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase   # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
EXECS = [0, 1, 2]
ARMS = {"autonomous": "gamer (pKiss only)", "panel_auto": "panel operator"}
B = 20000
NEAR = 0.15
METRICS = [("r_mix", "all three predictors"), ("r_in", "pKiss AND ProbKnot")]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    return min(2 * sum(math.comb(n, k) * 0.5 ** n for k in range(0, m + 1)), 1.0)


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


def single_linkage(ids, db, thresh=NEAR):
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
    g = {}
    for i in ids:
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def load_cell(e, s, need=60):
    p = os.path.join(HERE, f"gvp_e{e}_s{s}", "sweep.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if len(d.get("rows") or []) < need:
        print(f"  (skipping incomplete cell e{e}/s{s}: {len(d.get('rows') or [])}/{need} rows)")
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
                        "r_mix": bool(pnl.get("unanimous")),
                        "pkiss": bool(v.get("solved")), "stats": v.get("stats")}
        if ok:
            out[r["id"]] = cells
    return out


def boot(D, cells_by_seed, units, variant):
    """Nested percentile bootstrap. `units` is a list of lists of target ids (singletons => target-level)."""
    out = []
    K = len(units)
    for _ in range(B):
        cs = []
        for _ in SEEDS:
            s = SEEDS[random.randrange(len(SEEDS))]
            pool = cells_by_seed[s]
            k = len(pool) if variant == "full-nest" else 1
            cs.extend(pool[random.randrange(len(pool))] for _ in range(k))
        tot = n = 0
        for _ in range(K):
            for t in units[random.randrange(K)]:
                for c in cs:
                    tot += D[(c, t)]
                    n += 1
        out.append(tot / n)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main():
    data = {}
    for e in EXECS:
        for s in SEEDS:
            d = load_cell(e, s)
            if d:
                data[(s, e)] = d
    if not data:
        raise SystemExit("no gvp sweeps found")
    cells = sorted(data)
    ids = sorted(set.intersection(*[set(v) for v in data.values()]),
                 key=lambda x: int(x.split(":")[1]))
    n = len(ids)
    print(f"paired gamer-vs-panel replication: {n} common targets; {len(cells)} (seed, execution) cells: "
          + ", ".join(f"s{s}/e{e}" for s, e in cells) + "\n")

    tasks = {t.task_id: t.target for t in load_pseudobase(limit=60)}
    clusters = single_linkage(ids, tasks)
    print(f"structural clusters of the pool (single linkage, d<= {NEAR}): {len(clusters)} clusters "
          f"(sizes {sorted((len(c) for c in clusters), reverse=True)[:8]}...)\n")
    cells_by_seed = {s: [c for c in cells if c[0] == s] for s in SEEDS}

    for key, desc in METRICS:
        print(f"=== {key} ({desc}) ===")
        hdr = "  ".join(f"s{s}/e{e}" for s, e in cells)
        print(f"  {'arm':22} {hdr}   mean  exec-sd  seed-sd")
        for a in ARMS:
            cnt = {c: sum(data[c][t][a][key] for t in ids) for c in cells}
            per_seed = {s: [cnt[c] for c in cells_by_seed[s]] for s in SEEDS}
            ex = [st.stdev(v) for v in per_seed.values() if len(v) > 1]
            sm = [sum(v) / len(v) for v in per_seed.values() if v]
            print(f"  {ARMS[a]:22} " + "  ".join(f"{cnt[c]:5d}" for c in cells) +
                  f"  {sum(cnt.values())/len(cnt):5.2f}  {(sum(ex)/len(ex) if ex else float('nan')):6.2f}  "
                  f"{(st.stdev(sm) if len(sm) > 1 else float('nan')):6.2f}")

        D = {(c, t): int(data[c][t]["panel_auto"][key]) - int(data[c][t]["autonomous"][key])
             for c in cells for t in ids}
        obs = sum(D.values()) / len(D)
        print(f"\n  paired Delta = p_panel - p_gamer = {obs:+.4f}   "
              f"({obs*60:+.2f} targets in 60)")
        print("  95% percentile intervals, by resampling design (B=20,000):")
        for uname, units in (("target", [[t] for t in ids]), ("CLUSTER", clusters)):
            for variant in ("full-nest", "one-exec"):
                lo, hi = boot(D, cells_by_seed, units, variant)
                tag = "clears 0" if lo > 0 else "SPANS 0"
                print(f"    seeds+executions+{uname:8s} [{variant:9s}]  [{lo:+.4f},{hi:+.4f}]  {tag}")

        # per-cell paired McNemar, so the reader can see the pairing is exact
        print("  per-cell paired McNemar (panel-only b, gamer-only c):")
        line, tb, tc, wins = [], 0, 0, 0
        for c in cells:
            b = sum(1 for t in ids if data[c][t]["panel_auto"][key] and not data[c][t]["autonomous"][key])
            cc = sum(1 for t in ids if data[c][t]["autonomous"][key] and not data[c][t]["panel_auto"][key])
            tb += b
            tc += cc
            pcount = sum(data[c][t]["panel_auto"][key] for t in ids)
            gcount = sum(data[c][t]["autonomous"][key] for t in ids)
            wins += (pcount > gcount)
            line.append(f"s{c[0]}/e{c[1]}: b={b},c={cc},p={mcnemar_exact(b, cc):.3g}")
        for i in range(0, len(line), 3):
            print("    " + "   ".join(line[i:i + 3]))
        # Distribution-free CELL-LEVEL sign test: the 9 executions are independent runs, so the number of
        # cells in which the panel operator strictly exceeds the gamer is Binomial(9, 1/2) under the null.
        # This is immune to the execution-variance problem, because the comparison is made WITHIN each cell.
        psign = min(1.0, 2 * sum(math.comb(len(cells), k) * 0.5 ** len(cells)
                                 for k in range(wins, len(cells) + 1)))
        print(f"  cell-level sign test: panel strictly > gamer in {wins}/{len(cells)} executions, "
              f"two-sided p={psign:.4g}")
        print(f"  pooled discordance over all cells: b={tb} (panel only), c={tc} (gamer only)"
              f"{'  -- the gamer never wins a target in any cell' if tc == 0 else ''}")
        print()

    # ---- realized budget for the two arms of the HEADLINE comparison ----
    print("=== realized budget per arm, headline comparison (mean per target over the 9 cells) ===")
    print(f"  {'arm':22} {'pKiss':>8} {'ProbKnot':>9} {'iters':>8} {'cands':>8} {'oracle s':>9}")
    for a in ARMS:
        acc = {k: [] for k in ("n_pkiss", "n_probknot", "n_iters", "n_candidates", "oracle_s")}
        for c in cells:
            for t in ids:
                stt = data[c][t][a].get("stats")
                if stt:
                    for k in acc:
                        if stt.get(k) is not None:
                            acc[k].append(stt[k])
        if not acc["n_pkiss"]:
            print(f"  {ARMS[a]:22} (no stats recorded)")
            continue
        m = {k: sum(v) / len(v) for k, v in acc.items() if v}
        print(f"  {ARMS[a]:22} {m['n_pkiss']:8.1f} {m['n_probknot']:9.1f} {m['n_iters']:8.1f} "
              f"{m['n_candidates']:8.1f} {m['oracle_s']:9.1f}")
    print("\n  -> the arms share a matched NOMINAL cap; realized compute differs and is reported as such.")


if __name__ == "__main__":
    main()
