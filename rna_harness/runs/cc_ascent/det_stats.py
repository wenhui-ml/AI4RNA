#!/usr/bin/env python3
"""R8: the ablation and the headline comparison in the DETERMINISTIC, COMPUTE-MATCHED regime.

What changed and why it matters for the argument. Every ablation number in the paper up to R7 was measured
under wall-clock stopping rules, which cost the analysis two things:
  (a) execution-level nondeterminism at a fixed seed, of the same order as the method effect, so each cell
      had to be averaged over 3 executions and the component attribution under r_mix came out
      "unresolvable at n=60";
  (b) matched NOMINAL caps but unmatched REALIZED compute -- the undirected arms ran to the wall and spent
      33-63% MORE pKiss calls than the full operator.
Deterministic mode removes both at the root (counted budgets, identical pKiss allowance, ViennaRNA's own RNG
seeded per attempt). So each (arm, seed, target) cell here is a single noise-free observation, and the
bootstrap drops back to two levels (targets + seeds) with nothing hidden underneath.

Reports, per arm and for both primary metrics: per-seed counts, the paired difference against the full
operator with a two-level interval, the same against the single-oracle gamer (the headline comparison), and
the realized budget -- which is the check that compute-matching actually took effect rather than being
asserted.

Offline over det_s{0..4}/sweep.json; no episodes, no oracle calls.
"""
import json, os, random, statistics as st

random.seed(20260801)
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
GAMER = "autonomous"
FULL = "panel_auto"
ABL = ["panel_randwalk", "panel_blindrand", "panel_norepair", "panel_nodecompose", "panel_nocompose"]
ARMS = [GAMER, FULL] + ABL
SHORT = {GAMER: "gamer (1-oracle)", FULL: "full operator", "panel_randwalk": "- objective",
         "panel_blindrand": "- obj, - bias", "panel_norepair": "- repair",
         "panel_nodecompose": "- decompose", "panel_nocompose": "- compose"}
# CAREFUL: the sweep's `solved` flag is each TOOL'S OWN success predicate, and they differ. The gamer
# (DecomposePKTool, oracle="pkiss") reports pKiss-exact; the panel arms report pKiss AND ProbKnot. So the
# `solved` column is NOT comparable between the gamer and the panel arms, and the gamer is excluded from the
# in-objective block below. Only r_mix -- recomputed by the sweep's own _panel() from the emitted sequence,
# identically for every arm -- supports a cross-arm comparison, which is why the headline Delta_gamer is an
# r_mix quantity.
METRICS = [("r_in", "each arm's OWN in-loop predicate (gamer: pKiss only; panel arms: pKiss AND ProbKnot)"),
           ("r_mix", "all three predictors -- the only cross-arm comparable metric")]


def load(s, need=60):
    p = os.path.join(HERE, f"det_s{s}", "sweep.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if len(d.get("rows") or []) < need:
        print(f"  (skipping incomplete det_s{s}: {len(d.get('rows') or [])}/{need})")
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


def paired_ci(D, seeds, ids, B=20000):
    """Two-level percentile bootstrap: resample nominal seeds and targets. No execution level, because
    determinism removes it -- that is the point of this run, not an omission."""
    n = len(ids)
    rates = []
    for _ in range(B):
        ss = [seeds[random.randrange(len(seeds))] for _ in seeds]
        ts = [ids[random.randrange(n)] for _ in range(n)]
        rates.append(sum(D[(s, t)] for s in ss for t in ts) / (len(ss) * n))
    rates.sort()
    return rates[int(0.025 * B)], rates[int(0.975 * B)]


def block(data, seeds, ids, key, ref, others, label):
    print(f"  paired difference vs {label} ({key}), two-level interval:")
    for a in others:
        D = {(s, t): int(data[s][t][ref][key]) - int(data[s][t][a][key]) for s in seeds for t in ids}
        obs = sum(D.values()) / len(D)
        lo, hi = paired_ci(D, seeds, ids)
        sig = "SEPARATED" if lo > 0 else ("REVERSED" if hi < 0 else "not separated")
        print(f"    {SHORT[ref]:16} - {SHORT[a]:16} Delta={obs:+.4f}  [{lo:+.4f},{hi:+.4f}]  {sig}")


def main():
    data = {s: d for s in SEEDS if (d := load(s)) is not None}
    if not data:
        raise SystemExit("no deterministic sweeps found")
    seeds = sorted(data)
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    n = len(ids)
    print(f"deterministic regime: {len(seeds)} seeds {seeds}, {n} common targets, "
          f"1 noise-free execution each\n")

    for key, desc in METRICS:
        print(f"=== {key} ({desc}) ===")
        print(f"  {'arm':17} " + "  ".join(f"s{s}" for s in seeds) + "   mean   seed-sd")
        for a in ARMS:
            c = [sum(data[s][t][a][key] for t in ids) for s in seeds]
            sd = st.stdev(c) if len(c) > 1 else float("nan")
            print(f"  {SHORT[a]:17} " + "  ".join(f"{x:2d}" for x in c) + f"   {sum(c)/len(c):5.2f}  {sd:6.2f}")
        block(data, seeds, ids, key, FULL, ABL, "the full operator")
        if key == "r_mix":
            block(data, seeds, ids, key, FULL, [GAMER], "the gamer (headline Delta_gamer)")
        else:
            print("  (gamer omitted here: its own predicate is pKiss-only, so this column is not\n"
                  "   comparable across arms -- see the note in this script's header)")
        print()

    print("=== realized budget (mean per target over all cells) -- the compute-matching check ===")
    print(f"  {'arm':17} {'pKiss':>8} {'ProbKnot':>9} {'iters':>7} {'oracle s':>9}")
    pk = {}
    for a in ARMS:
        acc = {k: [] for k in ("n_pkiss", "n_probknot", "n_iters", "oracle_s")}
        for s in seeds:
            for t in ids:
                stt = data[s][t][a].get("stats")
                if stt:
                    for k in acc:
                        if stt.get(k) is not None:
                            acc[k].append(stt[k])
        if not acc["n_pkiss"]:
            print(f"  {SHORT[a]:17} (no stats)")
            continue
        m = {k: sum(v) / len(v) for k, v in acc.items() if v}
        pk[a] = m["n_pkiss"]
        print(f"  {SHORT[a]:17} {m['n_pkiss']:8.1f} {m['n_probknot']:9.1f} {m['n_iters']:7.1f} "
              f"{m['oracle_s']:9.1f}")
    if pk:
        panel_pk = {a: v for a, v in pk.items() if a != "panel_norepair"}
        lo, hi = min(panel_pk.values()), max(panel_pk.values())
        print(f"\n  pKiss-call spread across searching arms: {lo:.1f}-{hi:.1f} "
              f"(+{100*(hi-lo)/lo:.0f}% max-over-min)")
        print("  under the wall-clock regime the same spread was 87.3-142.7 (+63%).")


if __name__ == "__main__":
    main()
