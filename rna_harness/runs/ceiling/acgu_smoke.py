#!/usr/bin/env python3
"""Is the ACGU-composition flag even plumbed through, and does it change what the solver returns?

A 3-target smoke, run BEFORE the pre-registered A/B, for one reason: every run this project has ever done
carries `ACGUoff`, so the flag has never once travelled from our adapter to DesiRNA. If it does nothing, the
overnight experiment is pointless and should not be launched.
"""
import os, sys, time
sys.path.insert(0, "/data/wenhui/openclaude_rna")


def one(job):
    tid, target, args = job
    os.environ["DESIRNA_ARGS"] = args
    from rna_harness.oracles_pk import _oracle_pairs_cached
    _oracle_pairs_cached.cache_clear()
    from rna_harness.cognitive.certify import certify_design
    t0 = time.time()
    try:
        c = certify_design(target, budget="cheap", seed=0)
    except Exception as e:                                                       # noqa: BLE001
        return {"tid": tid, "args": args, "error": repr(e)[:160]}
    s = c.sequence or ""
    d = c.detail or {}
    return {"tid": tid, "args": args, "verdict": c.verdict, "seq": s,
            "gc": (s.count("G") + s.count("C")) / len(s) if s else None,
            "k": len((d.get("knot") or {}).get("confirmed_by") or []),
            "screened": d.get("n_candidates_screened") or 0, "wall_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from rna_harness.benchmarks.rnainvbench import load_pseudobase
    tasks = list(load_pseudobase())[:3]
    jobs = [(str(t.task_id), str(t.target), a) for t in tasks for a in ("", "-acgu on")]
    print(f"{len(jobs)} 次运行(3 靶标 × 2 臂),6 进程", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        R = list(ex.map(one, jobs))
    by = {}
    for r in R:
        by.setdefault(r["tid"], {})[r["args"] or "off"] = r
    for tid, arms in by.items():
        o, n = arms.get("off", {}), arms.get("-acgu on", {})
        same = o.get("seq") and o.get("seq") == n.get("seq")
        print(f"{tid:16s} off: {o.get('verdict','?'):16s} k={o.get('k')} GC={o.get('gc') and round(o['gc'],3)} "
              f"筛={o.get('screened')} {o.get('wall_s')}s")
        print(f"{'':16s} on : {n.get('verdict','?'):16s} k={n.get('k')} GC={n.get('gc') and round(n['gc'],3)} "
              f"筛={n.get('screened')} {n.get('wall_s')}s   序列{'相同 ← 标志可能没传下去' if same else '不同'}")
        for x in (o, n):
            if x.get("error"):
                print(f"{'':16s} 错误 {x['args'] or 'off'}: {x['error']}")
