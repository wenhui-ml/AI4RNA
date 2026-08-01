#!/usr/bin/env python3
"""R9 (reviewer Q5): does the compute control for D2 actually bracket the agent operators' compute?

Why this exists. D2 -- "the agent-written operators' carry-over advantage was not bought with compute" --
rests on ONE control: tripling the hand-built operator's pKiss allowance (82.2 -> 196.2 realized calls per
target) moves carry-over from 0.108 to 0.108. That control is only informative if 196.2 calls actually
brackets what the agent operators spend. Their source asks for 15 restarts x 3000 iterations against our
6 x 800, so it may not, and D2 would then be a control that never reached the regime it was meant to rule out.
Rather than argue from the constants in the source -- which say what the code REQUESTS, not what it spends
under the 20 s per-target cap the replication imposed -- we measure realized calls.

Method. Re-exec each frozen operator in the same sandbox namespace the replication used, but with counting
wrappers around the two crossing-aware primitives, and run it on the primary pool at one seed under the same
20 s per-target cap. Counters are read in-process, so a target killed by the cap still reports the calls it
made before dying (that is the honest number: capped work is work done).

Reading the result. If the agents' realized pKiss use is at or below 196.2, the compute control covers them
and D2 stands as reported. If it is far above, D2 must be restated as "not bought by a 3x allowance", which
is a weaker claim, and the paper has to say so.

Search-only: no LLM calls, no $ cost.
"""
import json, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from rna_harness.oracles_pk import pkiss_pairs, probknot_pairs                 # noqa: E402
from rna_harness.benchmarks.rnainvbench import load_pseudobase                 # noqa: E402
import invent_operator as IO                                                   # noqa: E402

OPS = [("gemini31pro", "google_gemini31propreview"), ("kimik3", "moonshotai_kimik3")]
CAP_S = 20
N_TARGETS = int(os.environ.get("ABUDGET_N", "60"))
SEED = 0

# Hand-built reference points. The FIRST is the like-for-like one: the agent operators were replicated under
# the same 20 s wall-clock cap as the gamer-vs-panel sweep, and in that sweep the hand-built panel operator
# realized 128.0 pKiss calls per target (carryover.py / Sec. 5). The other two are the deterministic
# counted-budget regime in which the transplant chain and the compute control were run, quoted so the D2
# control's reach is visible on its own scale.
HANDBUILT = {"hand-built, same 20s regime": 128.0,
             "hand-built, deterministic": 82.2,
             "3x allowance (the D2 control)": 196.2}
LIKE_FOR_LIKE = "hand-built, same 20s regime"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def counting_namespace(counts):
    """The replication's PRIMS, with the two crossing-aware oracles wrapped in counters."""
    def pk(seq):
        counts["pkiss"] += 1
        return set(pkiss_pairs(seq))

    def pb(seq):
        counts["probknot"] += 1
        return set(probknot_pairs(seq))

    prims = dict(IO.PRIMS)
    prims["fold_pkiss_pairs"] = pk
    prims["fold_probknot_pairs"] = pb
    return prims


def build(code, counts):
    """Same sandbox as make_design(), minus the fork -- we need the counters in this process."""
    import random, math, itertools
    import builtins as _b
    safe = {k: getattr(_b, k) for k in (
        "range", "len", "list", "dict", "set", "frozenset", "tuple", "str", "int", "float", "bool",
        "min", "max", "sum", "sorted", "enumerate", "zip", "abs", "map", "filter", "any", "all",
        "reversed", "round", "print", "isinstance", "chr", "ord", "Exception", "ValueError", "KeyError",
        "IndexError", "TypeError", "ZeroDivisionError", "StopIteration") if hasattr(_b, k)}
    ns = {"__builtins__": safe, "random": random, "math": math, "itertools": itertools,
          **counting_namespace(counts)}
    exec(code, ns)
    fn = ns.get("design")
    if not callable(fn):
        raise ValueError("no design() defined")
    return fn


def main():
    tasks = load_pseudobase(limit=N_TARGETS)
    signal.signal(signal.SIGALRM, _alarm)
    print(f"realized crossing-oracle calls per target, frozen agent operators, seed {SEED}, "
          f"n={len(tasks)}, {CAP_S}s cap (the replication's cap)\n")
    print(f"  {'operator':22} {'pKiss/target':>13} {'ProbKnot/target':>16} {'timeouts':>9} {'median pKiss':>13}")
    out = {}
    for name, folder in OPS:
        code = json.load(open(os.path.join(HERE, "invent_operator", folder, "result.json")))["best_code"]
        per_pk, per_pb, to = [], [], 0
        for t in tasks:
            counts = {"pkiss": 0, "probknot": 0}
            try:
                fn = build(code, counts)
            except Exception as e:
                print(f"  {name}: cannot build ({e})")
                break
            signal.alarm(CAP_S)
            try:
                fn(t.target, SEED)
            except _Timeout:
                to += 1
            except Exception:
                pass
            finally:
                signal.alarm(0)
            per_pk.append(counts["pkiss"]); per_pb.append(counts["probknot"])
        if not per_pk:
            continue
        per_pk.sort(); med = per_pk[len(per_pk) // 2]
        mean_pk = sum(per_pk) / len(per_pk); mean_pb = sum(per_pb) / len(per_pb)
        out[name] = {"pkiss": mean_pk, "probknot": mean_pb, "timeouts": to, "median": med, "n": len(per_pk)}
        print(f"  {name:22} {mean_pk:13.1f} {mean_pb:16.1f} {to:9d} {med:13.1f}")

    print(f"\n  {'hand-built reference':22} {'pKiss/target':>13}")
    for k, v in HANDBUILT.items():
        print(f"  {k:22} {v:13.1f}")

    if out:
        worst = max(v["pkiss"] for v in out.values())
        ctrl = HANDBUILT["3x allowance (the D2 control)"]
        base = HANDBUILT[LIKE_FOR_LIKE]
        print(f"\n  agents' highest realized pKiss use: {worst:.1f} per target")
        print(f"  hand-built, same 20s regime:        {base:.1f} per target")
        print(f"  the D2 compute control reaches:     {ctrl:.1f} per target")
        if worst > ctrl:
            print(f"  -> the control does NOT bracket the agents ({worst/ctrl:.1f}x short). D2 must be restated")
            print("     as 'not bought by a 3x allowance', and the stronger claim is untested.")
        elif worst < base:
            print(f"  -> the control brackets them by {ctrl/worst:.1f}x, and the direction is the opposite of the")
            print(f"     worry: the agents spend {base/worst:.1f}x FEWER pKiss calls per target than the hand-built")
            print("     operator they beat, in the same regime. D2 is not merely 'not bought with compute' --")
            print("     the advantage is obtained at a fraction of the compute. That is a stronger claim than")
            print("     the allowance control alone supports, and it is the one the measurement licenses.")
        else:
            print(f"  -> the control brackets the agents by {ctrl/worst:.1f}x: D2 stands as reported.")


if __name__ == "__main__":
    main()
