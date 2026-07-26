#!/usr/bin/env python3
"""R6: pool EVERY independent execution of the headline arm at each nominal seed.

The R5 revision estimated execution-level spread from three executions per nominal seed (the panel-ablation
sweep). The R6 gamer-vs-panel replication adds three more per seed, run in a DIFFERENT sweep composition
(2 arms per worker rather than 6), which changes machine load and therefore how much work a wall-clock-bounded
operator gets through. Pooling all of them gives the honest reproducibility picture for a nominal seed, and
lets the paper compare execution spread against the method effect on the same scale.

Offline over the saved sweeps; no episodes, no oracle calls.
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
# (label, directory template) for every independent execution of the two arms at a nominal seed
EXECS = [("ablate-sweep e0", "pablate_s{}"), ("ablate-sweep e1", "pabrep1_s{}"),
         ("ablate-sweep e2", "pabrep2_s{}"), ("paired-sweep e0", "gvp_e0_s{}"),
         ("paired-sweep e1", "gvp_e1_s{}"), ("paired-sweep e2", "gvp_e2_s{}")]


def uni(path, arm, need=60):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if len(d.get("rows") or []) < need:
        return None
    return sum(bool((((r.get("arms", {}) or {}).get(arm) or {}).get("panel") or {}).get("unanimous"))
               for r in d["rows"])


def report(arm, label):
    print(f"=== every independent execution of {label} ({arm}), r_mix/60, by nominal seed ===")
    spreads = []
    for s in SEEDS:
        vals = []
        for lbl, tmpl in EXECS:
            v = uni(os.path.join(HERE, tmpl.format(s), "sweep.json"), arm)
            if v is not None:
                vals.append((lbl, v))
        if not vals:
            continue
        nums = [v for _, v in vals]
        spreads.append(max(nums) - min(nums))
        print(f"  seed {s}: " + ", ".join(f"{v}" for _, v in vals) +
              f"   n={len(nums)}  range {min(nums)}-{max(nums)}  spread {max(nums)-min(nums)}  "
              f"mean {sum(nums)/len(nums):.2f}  sd {st.stdev(nums):.2f}")
    if spreads:
        print(f"  -> mean execution spread at a FIXED nominal seed = {sum(spreads)/len(spreads):.1f}/60\n")


if __name__ == "__main__":
    report("panel_auto", "the panel operator")
    report("autonomous", "the gamer")
    print("Read against the paired method effect (+0.056 = 3.3/60 over the 9 paired cells): execution spread\n"
          "at a fixed nominal seed EXCEEDS the method effect for the panel operator, while the gamer is nearly\n"
          "deterministic. This is why the comparison must be read paired WITHIN an execution, never as a\n"
          "difference of two separately-reported counts.")
