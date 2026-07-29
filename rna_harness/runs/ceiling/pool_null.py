"""Matched null for the APPLIED policy: best-of-N candidates, not one sequence.

calibrate.py measured the criterion on ONE sequence per target (the native) against ONE random sequence per
target: 84.0% sensitivity, 0.4% false accept. knot_deliverable.py then applies it as "does ANY of ~50
candidates pass", which takes a maximum over ~50 draws x 3 oracles. A per-draw false-accept rate of 0.4%
compounds: 1-(1-0.004)^50 is about 18%. So the applied policy's false-accept rate is NOT the calibrated one,
and reporting 124/125 without this control would repeat a mistake this branch has already made twice.

This is the matched control: N RANDOM sequences per target, same N as the real pools, same criterion.
"""
import json, os, random, sys, time
sys.path.insert(0, '/data/wenhui/openclaude_rna')
OUT = '/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/pool_null'
os.makedirs(OUT, exist_ok=True)


def work(item):
    tid, target, n_draw, seed = item
    dest = os.path.join(OUT, f"t{tid}.json")
    if os.path.exists(dest):
        return dest
    from rna_harness.cognitive.certify import _knot_recall
    g = random.Random(seed)
    hit, first = False, None
    for j in range(n_draw):
        seq = "".join(g.choice("ACGU") for _ in range(len(target)))
        kr = _knot_recall(seq, target)
        if kr["knot_confirmed"]:
            hit, first = True, j
            break
    rec = {"id": tid, "n": len(target), "n_draw": n_draw, "knot_confirmed": hit, "first_hit": first}
    json.dump(rec, open(dest, "w"))
    return dest


if __name__ == "__main__":
    A = '/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/agent/pk_desirna.json'
    rows = {int(r["task_id"].split(":")[1]): r for r in json.load(open(A))["rows"]}
    held = sorted(i for i in rows if i % 2 == 1)
    kd = json.load(open('/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/knot_deliverable.json'))
    npool = {r["id"]: r["n_pool"] for r in kd["per_target"]}
    sub = held[::4]                       # every 4th target: 32 of 125, enough to bound the rate
    items = [(i, rows[i]["target"], npool.get(i, 50), 1000 + i) for i in sub]
    print(f"matched null: {len(items)} targets x their real pool size, random sequences", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, it) for it in items]), 1):
            f.result()
            if n % 8 == 0:
                print(f"  {n}/{len(items)}  {(time.time()-t0)/60:.1f} min", flush=True)
    import glob
    R = [json.load(open(f)) for f in glob.glob(OUT + "/*.json")]
    R = [r for r in R if r["id"] in set(sub)]
    fa = sum(r["knot_confirmed"] for r in R)
    print(f"\n=== applied-policy false accept (best of the SAME pool size, random sequences) ===")
    print(f"  {fa}/{len(R)} = {fa/max(len(R),1):.1%}")
    print(f"  per-draw calibrated rate was 0.4%; analytic compounding at n=50 predicts ~18%")
    print(f"  real designs pass at 124/125 = 99.2%")
    if len(R):
        print(f"  -> separation: {124/125:.1%} (designs) vs {fa/len(R):.1%} (random pools)")
    json.dump({"n_targets": len(R), "false_accept": fa, "rate": fa/max(len(R),1), "per_target": R},
              open('/data/wenhui/openclaude_rna/rna_harness/runs/ceiling/pool_null.json', 'w'), indent=1)
