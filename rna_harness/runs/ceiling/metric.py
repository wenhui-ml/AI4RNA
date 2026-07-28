#!/usr/bin/env python3
"""The unified evaluation surface: one curve S(B), two orthogonal coordinates.

Capability and efficiency were being reported as two unrelated multiples, which made them impossible to
compare ("is 1.69x capability better or worse than 2.26x efficiency?"). They are not two facts. They are
two readings of one measured object:

    S(B) = max over {b_i} of  sum_i [ 1 - (1 - p_i)^{b_i} ]      s.t.   sum_i b_i c_i <= B

  p_i  per-trajectory success probability on target i, under INDEPENDENT adjudication
  c_i  expected oracle-CPU-seconds of one trajectory on target i
  b_i  trajectories bought

The form is derived, not fitted: disjoint seeds plus the benchmark's rule that a target counts once per
execution give exactly 1-(1-p)^b. independence.py checked the assumption the derivation needs and it holds
where it matters -- p-hat fitted on 8 trajectories predicts the 2- and 4-trajectory arms to within 4%,
inside their bootstrap intervals.

Each term is concave increasing in b_i, so the constrained maximum is exact greedy water-filling on the
marginal gains p_i(1-p_i)^{b} / c_i. That matters: S(B) is then a property of the SYSTEM (its p_i and c_i),
not of whatever scheduler happened to be used, which is what makes it a fair comparator.

Three quantities come out, and they answer three different questions:

    S_inf  = #{i : p_i > 0}          EFFECT      how many targets the system can reach at all
    B_half = argmin_B S(B) >= S_inf/2  EFFICIENCY  the oracle-CPU-seconds scale
    U      = log-budget average of S   a single scalar, if a scalar is needed to drive a search

and separately

    S*(B) - S_policy(B)              SCHEDULER SLACK   how far the current allocation is from optimal

Budget is in oracle-CPU-SECONDS, not oracle calls, because the three oracles differ threefold in unit cost
(pKiss 210 ms, ProbKnot 120, ShapeKnots 65 -- sk_cost.py) and ViennaRNA's fold scales as n^3. Seconds are
the only common currency, and they are what the project actually pays.
"""
import json, os
from collections import defaultdict
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- the curve
def water_fill(p, c, B, bmax=4000):
    """Exact concave greedy. Returns (solves, allocation). Marginal gain of the (b+1)-th trajectory on
    target i is p_i (1-p_i)^b, at cost c_i, so the greedy order is by gain-per-second."""
    p = np.asarray(p, float); c = np.asarray(c, float)
    live = np.where(p > 0)[0]
    if len(live) == 0 or B <= 0:
        return 0.0, np.zeros(len(p))
    b = np.zeros(len(p))
    gain = p[live] / c[live]
    spent = 0.0
    # heap-free: vectorised loop, each step funds the best marginal until the budget is gone
    for _ in range(bmax * max(1, len(live))):
        k = int(np.argmax(gain))
        i = live[k]
        if spent + c[i] > B:
            # nothing affordable at the current best; try the cheapest affordable, else stop
            aff = [j for j, idx in enumerate(live) if spent + c[idx] <= B]
            if not aff:
                break
            k = max(aff, key=lambda j: gain[j]); i = live[k]
            if spent + c[i] > B:
                break
        b[i] += 1
        spent += c[i]
        gain[k] = p[i] * (1 - p[i]) ** b[i] / c[i]
        if b[i] >= bmax:
            gain[k] = -1.0
        if gain.max() <= 0:
            break
    return float(np.sum(1.0 - (1.0 - p) ** b)), b


def curve(p, c, grid):
    return np.array([water_fill(p, c, B)[0] for B in grid])


def summarise(p, c, grid):
    s = curve(p, c, grid)
    s_inf = int(np.sum(np.asarray(p) > 0))
    half = s_inf / 2.0
    idx = np.searchsorted(s, half)
    if idx == 0:
        b_half = grid[0]
    elif idx >= len(grid):
        b_half = float("inf")
    else:
        w = (half - s[idx - 1]) / max(s[idx] - s[idx - 1], 1e-12)
        b_half = float(np.exp(np.log(grid[idx - 1]) + w * (np.log(grid[idx]) - np.log(grid[idx - 1]))))
    U = float(np.trapezoid(s, np.log(grid)) / (np.log(grid[-1]) - np.log(grid[0])))
    return s, s_inf, b_half, U


# ---------------------------------------------------------------- systems, from the gate experiment
recs = [json.load(open(os.path.join(H, "gate", f))) for f in os.listdir(os.path.join(H, "gate"))
        if f.endswith(".json")]
recs = [r for r in recs if "error" not in r]
tids = sorted({r["id"] for r in recs}, key=lambda t: int(t.split(":")[-1]))
SYS = {}
for arm in ("off_150", "off_400", "on_400"):
    R = [r for r in recs if r["arm"] == arm]
    w = defaultdict(list); s = defaultdict(list)
    for r in R:
        w[r["id"]].append(int(bool((r.get("panel") or {}).get("unanimous"))))
        s[r["id"]].append(float(r.get("oracle_s", 0.0)))
    SYS[arm] = {"p": np.array([np.mean(w[t]) for t in tids]),
                "c": np.array([max(np.mean(s[t]), 1e-6) for t in tids])}

GRID = np.logspace(1, 5.2, 60)          # 10 s to ~160 000 oracle-CPU-seconds
print("=== the three measured systems, 60 crossing targets, 8 seeds each ===")
print("    budget axis = oracle-CPU-seconds; metric = panel-unanimous, independently adjudicated")
print("    (see the USD section at the end: seconds are only the right currency while no LLM is in")
print("     the loop, which is true of every operator here today and will not stay true)\n")
print(f"  {'system':10} {'S_inf':>6} {'B_half (s)':>11} {'U':>7} {'c per traj':>11} "
      f"{'reachable targets':>18}")
OUT = {}
for arm in ("off_150", "off_400", "on_400"):
    p, c = SYS[arm]["p"], SYS[arm]["c"]
    s, s_inf, b_half, U = summarise(p, c, GRID)
    OUT[arm] = {"s": s, "s_inf": s_inf, "b_half": b_half, "U": U}
    print(f"  {arm:10} {s_inf:6d} {b_half:11.0f} {U:7.2f} {np.mean(c):11.1f} "
          f"{str([t.split(':')[-1] for t in tids if p[tids.index(t)] > 0][:8]):>18}")

print("\n=== the curves, S(B) ===")
print(f"  {'B (oracle-s)':>13} " + " ".join(f"{a:>9}" for a in OUT))
for k in range(0, len(GRID), 6):
    print(f"  {GRID[k]:13.0f} " + " ".join(f"{OUT[a]['s'][k]:9.2f}" for a in OUT))

print("\n=== where do the curves cross? ===")
for a, b in (("off_150", "on_400"), ("off_400", "on_400")):
    d = OUT[b]["s"] - OUT[a]["s"]
    sign = np.sign(d)
    cross = [k for k in range(1, len(GRID)) if sign[k] > 0 and sign[k - 1] <= 0]
    if cross:
        k = cross[0]
        w = -d[k - 1] / max(d[k] - d[k - 1], 1e-12)
        Bx = float(np.exp(np.log(GRID[k - 1]) + w * (np.log(GRID[k]) - np.log(GRID[k - 1]))))
        print(f"  {b} overtakes {a} at B = {Bx:.0f} oracle-s "
              f"({Bx/3600:.2f} core-hours for the whole 60-target benchmark)")
        print(f"    below that, {a} is better; above it the gap grows to "
              f"{OUT[b]['s_inf'] - OUT[a]['s_inf']} targets")
    else:
        better = b if OUT[b]["s"][-1] > OUT[a]["s"][-1] else a
        print(f"  no crossing on this grid; {better} dominates throughout")

print("\n=== scheduler slack: how far is 'one trajectory per target' from optimal? ===")
print(f"  {'system':10} {'B of the flat policy':>21} {'flat S':>8} {'optimal S at same B':>21} "
      f"{'slack':>7}")
for arm in ("off_150", "on_400"):
    p, c = SYS[arm]["p"], SYS[arm]["c"]
    Bflat = float(np.sum(c))                       # one trajectory on every target
    flat = float(np.sum(1.0 - (1.0 - p) ** 1))
    opt, _ = water_fill(p, c, Bflat)
    print(f"  {arm:10} {Bflat:21.0f} {flat:8.2f} {opt:21.2f} {opt/max(flat,1e-9):7.2f}x")
print("  this is the quantity the trajectory experiment moved: it did not raise S_inf (12 -> 10 distinct")
print("  targets) but it closed part of this gap. Capability and scheduling are separate coordinates.")

print("\n=== a known bias, stated: S_inf from 8 seeds is a LOWER bound ===")
for ptrue in (0.05, 0.10, 0.20):
    print(f"  a target with true p = {ptrue:.2f} shows 0 wins in 8 seeds with probability "
          f"{(1-ptrue)**8:.2f}")
print("  so S_inf is underestimated, and the two coordinates have OPPOSITE sampling needs:")
print("    measuring S_inf  -> many seeds, few targets")
print("    measuring B_half -> few seeds, many targets")
print("  a single sweep cannot estimate both well; the eval suite should budget for each separately.")

print("\n" + "=" * 78)
print("=== the same surface in USD, with token capex amortised over a reuse count N ===")
import sys
sys.path.insert(0, os.path.abspath(os.path.join(H, "..", "..", "..")))
from rna_harness.cognitive.cost import CPU_USD_PER_SEC                        # noqa: E402

print(f"  CPU ${CPU_USD_PER_SEC:.3e}/core-s. Seconds and dollars are proportional while nothing calls an")
print("  LLM at solve time, so the CURVE SHAPE is unchanged -- what dollars add is commensurability with")
print("  the token cost of producing the operator, which seconds cannot express.\n")
print(f"  {'system':10} {'S_inf':>6} {'B_half ($)':>11} {'$/solve at S_inf/2':>19} "
      f"{'$ for a full pass':>18}")
for arm in OUT:
    p, c = SYS[arm]["p"], SYS[arm]["c"]
    bh_usd = OUT[arm]["b_half"] * CPU_USD_PER_SEC
    per = bh_usd / max(OUT[arm]["s_inf"] / 2.0, 1e-9)
    print(f"  {arm:10} {OUT[arm]['s_inf']:6d} {bh_usd:11.4f} {per:19.4f} "
          f"{float(np.sum(c))*CPU_USD_PER_SEC:18.4f}")

INVENT = [("$0.08  (1 agent episode)", 0.08), ("$0.40  (5 episodes)", 0.40),
          ("$2.00  (25 episodes)", 2.00), ("$20.00 (250 episodes)", 20.00)]
print("\n  effective $/solve once the operator's INVENTION capex is amortised over N solves")
print(f"  (marginal $/solve: off_150 {918.4*CPU_USD_PER_SEC:.5f}, on_400 {1220.8*CPU_USD_PER_SEC:.5f})")
print(f"  {'capex':>26} " + " ".join(f"{f'N={n}':>10}" for n in (1, 22, 100, 1000)) +
      f" {'breakeven N':>12}")
marg = 1220.8 * CPU_USD_PER_SEC
for lab, cap in INVENT:
    cells = " ".join(f"{marg + cap/n:10.4f}" for n in (1, 22, 100, 1000))
    print(f"  {lab:>26} {cells} {cap/marg:12.0f}")
print("  N = 22 is one gated pass over the 60-target benchmark, so the right-hand column is also")
print("  'how many full benchmark passes the invention cost is worth'. A single agent episode ($0.08)")
print("  costs about as much as ONE ENTIRE PASS ($0.093) -- at this scale CPU is nearly free and the")
print("  token spend of the R&D loop, not of solving, is the dominant cost.")

print("\n  when do RUNTIME tokens (an LLM inside the solve loop) start to matter?")
from rna_harness.cognitive.cost import llm_cost_usd                            # noqa: E402
att = 111.9 * CPU_USD_PER_SEC
for per, lab in ((1, "one LLM call per attempt"), (8, "one per target (~8 attempts)"),
                 (60, "one per benchmark pass")):
    add = llm_cost_usd(2000, 500) / per
    print(f"    {lab:<32} +${add:.5f}/attempt = {100*add/att:5.1f}% on top of ${att:.5f}")
print("    one LLM call at 2k/0.5k costs $0.00175, about the same as one solve ATTEMPT ($0.00155),")
print("    and one at 20k/4k costs $0.016, about the same as one finished SOLVE ($0.017).")

json.dump({"cpu_usd_per_sec": CPU_USD_PER_SEC,
           **{a: {"p": SYS[a]["p"].tolist(), "c": SYS[a]["c"].tolist(),
                  "s_inf": OUT[a]["s_inf"], "b_half_s": OUT[a]["b_half"],
                  "b_half_usd": OUT[a]["b_half"] * CPU_USD_PER_SEC, "U": OUT[a]["U"],
                  "pass_usd": float(np.sum(SYS[a]["c"])) * CPU_USD_PER_SEC,
                  "grid": GRID.tolist(), "S": OUT[a]["s"].tolist()} for a in OUT}},
          open(os.path.join(H, "metric.json"), "w"), indent=1)
print("\nwrote metric.json")
