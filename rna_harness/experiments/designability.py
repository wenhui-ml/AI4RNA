"""Designability check (attribution, no LLM creds needed).

Separates 'agent-limited (learnable)' from 'oracle-limited / hard under turner2004' for a set
of Eterna100 tasks, per the TECH_REPORT attribution rule (let the LLM internalize search; only
credit true learning). Two authoritative, credit-free signals:

  Stage 1 -- human sample solution under OUR oracle. Each Eterna100 puzzle ships a human
    solution. If it folds to the target under the canonical verifier (turner2004, 37C,
    dangles=2), the target is PROVABLY designable under our exact oracle by a concrete
    sequence -> any agent failure on it is agent-side (a last-mile / strategy gap, learnable).
    Caveat: Eterna solutions were found under the game's engine, so a sample that does NOT
    fold under turner2004 is INCONCLUSIVE (different energy model), not proof of hardness.

  Stage 2 (optional) -- strong classical search under turner2004 for the inconclusive ones,
    to see if SOME turner2004 solution exists even when the human sample doesn't transfer.

Usage:
  python -m rna_harness.experiments.designability --ids 6 16 35 ... [--search] [--restarts 200]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from ..benchmarks import eterna100_tasks
from ..verifier import verify


def _sample_designability(ids: Optional[List[int]], max_len: Optional[int]) -> List[Dict]:
    tasks = eterna100_tasks(ids=ids, max_len=max_len)
    out = []
    for t in sorted(tasks, key=lambda z: z.n):
        s = (t.meta.get("sample_solution") or "").strip().upper().replace("T", "U")
        rec = {"id": t.meta.get("puzzle_id"), "name": t.meta.get("name"), "n": t.n,
               "task_id": t.task_id, "sample_present": bool(s)}
        if s and len(s) == t.n:
            v = verify(t, s, compute_cross_oracle=False)
            rec.update({
                "sample_solved_our_oracle": bool(v.solved),
                "sample_mfe_match": bool(v.mfe_match),
                "sample_is_umfe": bool(v.is_umfe),
                "sample_bp": v.ensemble.get("bp_distance_to_target") if v.ensemble else None,
            })
        else:
            rec.update({"sample_solved_our_oracle": None, "sample_mfe_match": None,
                        "sample_is_umfe": None, "sample_bp": None,
                        "note": f"sample len {len(s)} != n {t.n}" if s else "no sample"})
        out.append(rec)
    return out


def _search_one(task, restarts: int, cores: int) -> Dict:
    """Strong classical turner2004 search (harness parallel_search SA) for one target."""
    from ..mcp.motif_server import parallel_search  # deterministic multi-restart SA, fan-out cores
    try:
        r = parallel_search(task.target, restarts=restarts, max_cores=cores)
    except Exception as e:
        return {"search_solved": None, "search_error": str(e)[:200]}
    seq = r.get("best_sequence") or r.get("sequence")
    solved = False
    bp = None
    if seq and len(seq) == task.n:
        v = verify(task, seq, compute_cross_oracle=False)
        solved = bool(v.solved)
        bp = v.ensemble.get("bp_distance_to_target") if v.ensemble else None
    return {"search_solved": solved, "search_bp": bp, "search_n_solved": r.get("n_solved"),
            "search_restarts": restarts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--search", action="store_true", help="stage 2: classical search on inconclusive samples")
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--cores", type=int, default=int(os.environ.get("RNA_MAX_CORES", "48")))
    ap.add_argument("--search-max-n", type=int, default=160, help="skip search on targets longer than this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    recs = _sample_designability(args.ids, args.max_len)

    if args.search:
        by_id = {t.meta.get("puzzle_id"): t for t in eterna100_tasks(ids=args.ids, max_len=args.max_len)}
        for rec in recs:
            # only search targets whose human sample did NOT transfer to our oracle
            if rec.get("sample_solved_our_oracle") is False and rec["n"] <= args.search_max_n:
                print(f"  [search] id={rec['id']} n={rec['n']} {rec['name']} ...", flush=True)
                rec.update(_search_one(by_id[rec["id"]], args.restarts, args.cores))

    n = len(recs)
    prov = sum(1 for r in recs if r.get("sample_solved_our_oracle") is True)
    mfe_only = sum(1 for r in recs if r.get("sample_solved_our_oracle") is False and r.get("sample_mfe_match"))
    inconcl = sum(1 for r in recs if r.get("sample_solved_our_oracle") is False)
    searched_solved = sum(1 for r in recs if r.get("search_solved") is True)
    print(f"\n=== DESIGNABILITY: {n} tasks ===")
    print(f"  provably designable under OUR oracle (human sample solves): {prov}/{n}")
    print(f"  human sample folds MFE-match but not the strict criterion:  {mfe_only}")
    print(f"  sample does NOT transfer to turner2004 (inconclusive):      {inconcl}/{n}")
    if args.search:
        print(f"  of inconclusive+short, classical search found a turner2004 solution: {searched_solved}")
    print()
    hdr = f"{'id':>4} {'n':>4} {'sampSolv':8} {'mfe':4} {'umfe':4} {'sampBP':6}"
    if args.search:
        hdr += f" {'srchSolv':8} {'srchBP':6}"
    hdr += " :: name"
    print(hdr)
    for r in recs:
        row = (f"{str(r['id']):>4} {r['n']:>4} {str(r.get('sample_solved_our_oracle')):8} "
               f"{str(r.get('sample_mfe_match'))[:1]:4} {str(r.get('sample_is_umfe'))[:1]:4} "
               f"{str(r.get('sample_bp')):6}")
        if args.search:
            row += f" {str(r.get('search_solved')):8} {str(r.get('search_bp')):6}"
        row += f" :: {r['name']}"
        print(row)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n": n, "provably_designable": prov, "inconclusive": inconcl,
                       "records": recs}, f, indent=2)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
