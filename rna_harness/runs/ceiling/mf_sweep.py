#!/usr/bin/env python3
"""Does the mean-field regime add anything over Delta? The test that decides whether the tool is real.

meanfield.py gives every helix a composition index C and every target a regime signature. On eight
targets the signatures were internally consistent (both 400nt targets independent; three multiloop-rich
mid-length targets frustrated; Bug 18 strongly cooperative) and contradicted my prior twice.

Eight is not evidence. This runs the <150nt Eterna subset, where Delta already predicts failure at
AUC 0.860, and asks whether the regime carries information Delta does not:

  * AUC of each regime statistic alone
  * AUC of Delta alone, on the same targets
  * whether the regimes separate targets that Delta scores the same

If the regime adds nothing over Delta it is a redescription and should be dropped. That is the outcome
this script is written to be able to return.
"""
import json, math, os, statistics as st, sys
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
from rna_harness.benchmarks.eterna100 import load_eterna100 as L       # noqa: E402
from meanfield import analyse                                          # noqa: E402


def job(a):
    name, db, ns = a
    try:
        return analyse(name, db, ns)
    except Exception as e:                                             # noqa: BLE001
        return {"target": name, "n": len(db), "error": repr(e)[:120]}


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum(1 if a > b else .5 if a == b else 0 for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    T = {f"eterna100:{t['id']}:{t['name']}": t['target'] for t in L()}
    rows = json.load(open(os.path.join(HERE, "..", "..", "_archive_runs",
                                       "eterna_paired", "memon", "rows_partial.json")))
    by = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    todo = [(t, T[t], 60) for t in by if t in T and by[t][0]["n"] < 150]
    print(f"{len(todo)} targets under 150nt, 60 samples each", flush=True)
    with mp.get_context("fork").Pool(20) as p:
        res = p.map(job, todo)
    out = []
    for r in res:
        if "error" in r or not r.get("helices"):
            continue
        C = [h["C"] for h in r["helices"]]
        out.append({"task": r["target"], "n": r["n"], "n_hel": r["n_helices"],
                    "solved": any(x["solved"] for x in by[r["target"]]),
                    "medC": round(st.median(C), 3),
                    "frac_coop": round(sum(1 for c in C if c > 0.35) / len(C), 3),
                    "frac_frus": round(sum(1 for c in C if c < -0.35) / len(C), 3),
                    "frac_ind": round(sum(1 for c in C if -0.35 <= c <= 0.35) / len(C), 3)})
    json.dump(out, open(os.path.join(HERE, "mf_sweep.json"), "w"), indent=1)

    dm = {r["task"]: r for r in json.load(open(os.path.join(HERE, "info_budget.json")))}
    U = [r for r in out if not r["solved"]]; S = [r for r in out if r["solved"]]
    print(f"\n{len(out)} targets analysed: {len(U)} unsolved, {len(S)} solved\n")
    print(f"{'statistic':22} {'median unsolved':>16} {'median solved':>14} {'AUC -> failure':>15}")
    for k, sign in (("medC", -1), ("frac_coop", 1), ("frac_frus", 1), ("frac_ind", -1)):
        print(f"  {k:20} {st.median(r[k] for r in U):16.3f} {st.median(r[k] for r in S):14.3f} "
              f"{auc([sign*r[k] for r in U], [sign*r[k] for r in S]):15.3f}")
    dU = [dm[r['task']]['delta_kcal'] for r in U if r['task'] in dm]
    dS = [dm[r['task']]['delta_kcal'] for r in S if r['task'] in dm]
    print(f"  {'Delta (kcal/mol)':20} {st.median(dU):16.1f} {st.median(dS):14.1f} "
          f"{auc(dU, dS):15.3f}")

    # does the regime separate targets Delta scores alike?
    print("\ncontrol: among targets in the MIDDLE Delta tercile, does the regime still separate?")
    ds = sorted(dm[r['task']]['delta_kcal'] for r in out if r['task'] in dm)
    lo, hi = ds[len(ds)//3], ds[2*len(ds)//3]
    mid = [r for r in out if r['task'] in dm and lo <= dm[r['task']]['delta_kcal'] <= hi]
    mU = [r for r in mid if not r['solved']]; mS = [r for r in mid if r['solved']]
    print(f"  middle tercile: {len(mid)} targets ({len(mU)} unsolved, {len(mS)} solved), "
          f"Delta in [{lo:.1f},{hi:.1f}]")
    if mU and mS:
        for k, sign in (("medC", -1), ("frac_frus", 1), ("frac_coop", 1)):
            print(f"    AUC({k:10} -> failure | Delta held) = "
                  f"{auc([sign*r[k] for r in mU], [sign*r[k] for r in mS]):.3f}")
    else:
        print("    (one class empty in the middle tercile -- cannot test)")


if __name__ == "__main__":
    main()
