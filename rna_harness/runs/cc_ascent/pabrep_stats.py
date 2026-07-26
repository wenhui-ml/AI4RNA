#!/usr/bin/env python3
"""R5 §3 + §4: three-level variance (target / nominal seed / EXECUTION) and realized-budget accounting.

The operators terminate on wall-clock, so a nominal seed does not fix the run. This script pools the three
executions of the panel-operator ablation -- pablate_s{0,1,2} (execution 0) and pabrep{1,2}_s{0,1,2} -- and
reports, for every arm and both primary metrics:

  (1) the per-(seed, execution) counts, so execution spread is visible directly;
  (2) a variance decomposition: within-seed (execution) sd vs between-seed sd, i.e. how much of the
      seed-to-seed spread the paper previously attributed to "algorithmic variance" is in fact re-run noise;
  (3) a THREE-level bootstrap of the paired difference (full operator - ablation) resampling targets,
      nominal seeds AND executions, which is the interval the two-level bootstrap understated;
  (4) the REALIZED budget per arm (pKiss calls, ProbKnot calls, iterations, candidates, oracle seconds),
      because identical wall-clock and identical iteration caps do NOT imply identical realized compute.

Offline over the saved sweeps; no episodes, no oracle calls.
"""
import json, math, os, random, statistics as st

random.seed(20260726)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
# (execution label, directory template)
EXECS = [("e0", "pablate_s{}"), ("e1", "pabrep1_s{}"), ("e2", "pabrep2_s{}")]
ARMS = ["panel_auto", "panel_randwalk", "panel_blindrand", "panel_norepair",
        "panel_nodecompose", "panel_nocompose"]
SHORT = {"panel_auto": "full", "panel_randwalk": "-objective", "panel_blindrand": "-obj,-bias",
         "panel_norepair": "-repair", "panel_nodecompose": "-decompose", "panel_nocompose": "-compose"}
METRICS = [("r_in", "pKiss AND ProbKnot"), ("r_mix", "all three")]


def load(path, need=60):
    """id -> arm -> {r_in, r_mix, stats}; drops any target where some arm errored (keeps arms paired).
    Returns None for a MISSING or still-INCOMPLETE sweep, so an in-flight run cannot silently shrink the
    common-target intersection."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if len(d.get("rows") or []) < need:
        print(f"  (skipping incomplete sweep {os.path.basename(os.path.dirname(path))}: "
              f"{len(d.get('rows') or [])}/{need} rows)")
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
            cells[a] = {"r_in": bool(v.get("solved")), "r_mix": bool(pnl.get("unanimous")),
                        "stats": v.get("stats")}
        if ok:
            out[r["id"]] = cells
    return out


def main():
    data = {}   # (seed, exec) -> loaded
    for lbl, tmpl in EXECS:
        for s in SEEDS:
            d = load(os.path.join(HERE, tmpl.format(s), "sweep.json"))
            if d:
                data[(s, lbl)] = d
    if not data:
        raise SystemExit("no sweeps found")
    cells = sorted(data)
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(ids)
    print(f"three-level analysis: {n} common targets; {len(cells)} (seed, execution) cells present:")
    print("  " + ", ".join(f"s{s}/{e}" for s, e in cells) + "\n")

    for key, desc in METRICS:
        print(f"=== {key} ({desc}) ===")
        hdr = "  ".join(f"s{s}/{e}" for s, e in cells)
        print(f"  {'arm':13} {hdr}   mean   exec-sd  seed-sd")
        table = {}
        for a in ARMS:
            counts = {c: sum(data[c][t][a][key] for t in ids) for c in cells}
            table[a] = counts
            vals = list(counts.values())
            # within-seed (execution) sd, pooled over seeds; and between-seed sd of the seed means
            per_seed = {s: [counts[(s2, e)] for (s2, e) in cells if s2 == s] for s in SEEDS}
            ex = [st.stdev(v) for v in per_seed.values() if len(v) > 1]
            exec_sd = sum(ex) / len(ex) if ex else float("nan")
            seed_means = [sum(v) / len(v) for v in per_seed.values() if v]
            seed_sd = st.stdev(seed_means) if len(seed_means) > 1 else float("nan")
            print(f"  {SHORT[a]:13} " + "  ".join(f"{counts[c]:5d}" for c in cells) +
                  f"   {sum(vals)/len(vals):5.2f}  {exec_sd:6.2f}   {seed_sd:6.2f}")

        # three-level bootstrap of the paired difference full - ablation
        print("    three-level bootstrap (targets + seeds + executions), 95% interval on full - ablation:")
        for a in ARMS[1:]:
            D = {(c, t): int(data[c][t]["panel_auto"][key]) - int(data[c][t][a][key])
                 for c in cells for t in ids}
            obs = sum(D.values()) / len(D)
            B, rates = 20000, []
            byseed = {s: [c for c in cells if c[0] == s] for s in SEEDS}
            for _ in range(B):
                # resample nominal seeds, then an execution within each drawn seed, then targets
                cs = []
                for _ in SEEDS:
                    s = SEEDS[random.randrange(len(SEEDS))]
                    pool = byseed[s]
                    cs.append(pool[random.randrange(len(pool))])
                ts = [ids[random.randrange(n)] for _ in range(n)]
                rates.append(sum(D[(c, t)] for c in cs for t in ts) / (len(cs) * len(ts)))
            rates.sort()
            lo, hi = rates[int(0.025 * B)], rates[int(0.975 * B)]
            verdict = "SEPARATED" if lo > 0 else "not separated (interval spans 0)"
            print(f"      vs {SHORT[a]:13} Delta={obs:+.4f}  [{lo:+.4f},{hi:+.4f}]  {verdict}")
        print()

    # ---- realized budget accounting (R5 §4) ----
    print("=== realized budget per arm (mean over targets x cells; nominal caps: 90s repair, <=800 iters,")
    print("    ProbKnot gated at pk_bp<=3 with <=120 calls/target) ===")
    print(f"  {'arm':13} {'pKiss calls':>12} {'ProbKnot':>10} {'iters':>8} {'cands':>8} {'oracle s':>9} {'repair s':>9}")
    for a in ARMS:
        acc = {k: [] for k in ("n_pkiss", "n_probknot", "n_iters", "n_candidates", "oracle_s", "repair_wall_s")}
        for c in cells:
            for t in ids:
                stt = data[c][t][a].get("stats")
                if stt:
                    for k in acc:
                        if stt.get(k) is not None:
                            acc[k].append(stt[k])
        if not acc["n_pkiss"]:
            print(f"  {SHORT[a]:13} (no stats recorded in these sweeps)")
            continue
        m = {k: sum(v) / len(v) for k, v in acc.items() if v}
        print(f"  {SHORT[a]:13} {m['n_pkiss']:12.1f} {m['n_probknot']:10.1f} {m['n_iters']:8.1f} "
              f"{m['n_candidates']:8.1f} {m['oracle_s']:9.1f} {m['repair_wall_s']:9.1f}")
    print("\n  -> if these differ across arms, the arms share a matched NOMINAL budget, not a matched")
    print("     REALIZED budget; report it as such.")


def headline_arm_reproducibility():
    """The decision-relevant number: how reproducible is the PAPER'S HEADLINE ARM at a fixed nominal seed?
    Pools every independent execution of panel_auto we have at seeds 0-2, including the ORIGINAL runs that
    Table 3 reports (different file layout, hence handled separately from main())."""
    IDS = [f"pseudobase:{i}" for i in range(60)]

    def uni(path, arm=None):
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        if len(d.get("rows") or []) < 60:
            return None
        out = set()
        for r in d["rows"]:
            u = (((r.get("arms", {}) or {}).get(arm) or {}).get("panel") or {}).get("unanimous") \
                if arm else r.get("unanimous")
            if u:
                out.add(r["id"])
        return len(out)

    srcs = {0: ["panel_auto/panel_panel_auto.json"], 1: ["seed1/panel/panel_panel_auto.json"],
            2: ["seed2/panel/panel_panel_auto.json"]}
    print("\n=== reproducibility of the headline arm (panel_auto r_mix) at a FIXED nominal seed ===")
    allspreads = []
    for s in SEEDS:
        vals = [v for v in (uni(p) for p in srcs[s]) if v is not None]
        for _, tmpl in EXECS:
            v = uni(os.path.join(HERE, tmpl.format(s), "sweep.json"), "panel_auto")
            if v is not None:
                vals.append(v)
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            allspreads.append(spread)
            print(f"  nominal seed {s}: r_mix over {len(vals)} independent executions = {vals}  "
                  f"(range {min(vals)}-{max(vals)}, spread {spread}/60, sd {st.stdev(vals):.2f})")
        else:
            print(f"  nominal seed {s}: only {len(vals)} execution")
    if allspreads:
        print(f"  -> mean execution spread {sum(allspreads)/len(allspreads):.1f}/60 at a FIXED seed, against a")
        print(f"     pooled method effect of +0.080 = 4.8/60. Same order of magnitude.")


if __name__ == "__main__":
    main()
    headline_arm_reproducibility()
