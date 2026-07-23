#!/usr/bin/env python3
"""Positive-side STRINGENCY (chomsky_climb): strengthen the positive verdict beyond exact-match.

A panel-unanimous design already has the target = the predicted structure under three independent engines.
Here we add a thermodynamic-DOMINANCE rung: using pKiss suboptimal enumeration, we check that the target is
the pseudoknot MFE AND compute the free-energy GAP to the best ALTERNATIVE fold (the lowest-energy structure
whose pair set differs from the target). A large gap means the target is not a marginal MFE but the dominant
fold; we report how many certified positives clear a dominance margin. Pair sets are compared by position
(bracket-type agnostic), so pKiss's [] / {} convention matches the target's () / []. Offline, no LLM.
"""
import json, os, re, subprocess, sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from rna_harness.benchmarks.rnainvbench import load_pseudobase

HERE = os.path.dirname(os.path.abspath(__file__))
PK = "/home/ubuntu/miniconda3/envs/pk/bin/pKiss"
PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
LINE = re.compile(r"^\s*(-?\d+\.\d+)\s+([.()\[\]{}<>]+)\s*$")


def parse_pairs(db):
    st = {o: [] for o, _ in PAIRS}; cl = {c: o for o, c in PAIRS}; P = set()
    for i, ch in enumerate(db):
        if ch in st:
            st[ch].append(i)
        elif ch in cl and st[cl[ch]]:
            P.add((st[cl[ch]].pop(), i))
    return frozenset(P)


PK_BIN_DIR = os.path.dirname(PK)


def subopt(seq, window=250, timeout=60):
    env = dict(os.environ)
    env["PATH"] = PK_BIN_DIR + ":" + env.get("PATH", "")  # pKiss is a perl wrapper needing its bin on PATH
    r = subprocess.run([PK, "--mode", "subopt", "--absoluteDeviation", str(window), seq],
                       capture_output=True, text=True, timeout=timeout, env=env)
    rows = []
    for ln in r.stdout.splitlines():
        m = LINE.match(ln)
        if m:
            rows.append((float(m.group(1)), m.group(2)))
    return rows


def gap_for(seq, target, window=250):
    tp = parse_pairs(target)
    rows = subopt(seq, window)
    if not rows:
        return None
    emfe = rows[0][0]
    mfe_is_target = (parse_pairs(rows[0][1]) == tp)
    ealt = next((e for e, st in rows if parse_pairs(st) != tp), None)
    g = (ealt - emfe) if ealt is not None else window / 100.0  # no alternative within window
    return {"emfe": emfe, "mfe_is_target": mfe_is_target, "gap": round(g, 2),
            "alt_in_window": ealt is not None}


def collect_unanimous():
    """(seq, target, source) for every panel-unanimous panel_auto design across seeds."""
    id2t = {t.task_id: t.target for t in load_pseudobase(limit=60)}
    out = []
    # seed0/1/2: run.json (seq) + panel_panel_auto.json (unanimous)
    for tag, rj, pj in [("seed0", "panel_auto/run.json", "panel_auto/panel_panel_auto.json"),
                        ("seed1", "seed1/panel/run.json", "seed1/panel/panel_panel_auto.json"),
                        ("seed2", "seed2/panel/run.json", "seed2/panel/panel_panel_auto.json")]:
        try:
            uni = {r["id"] for r in json.load(open(os.path.join(HERE, pj)))["rows"] if r.get("unanimous")}
            for r in json.load(open(os.path.join(HERE, rj)))["rows"]:
                s = r["by_policy"].get("panel_auto", {}).get("seq")
                if r["id"] in uni and s and r["id"] in id2t:
                    out.append((s, id2t[r["id"]], f"{tag}:{r['id']}"))
        except Exception as e:
            print(f"  (skip {tag}: {repr(e)[:60]})")
    # seed3/4: sweep.json (seq + target + panel inline)
    for tag in ("seed3", "seed4"):
        try:
            for r in json.load(open(os.path.join(HERE, tag, "sweep.json")))["rows"]:
                a = r["arms"]["panel_auto"]
                if (a.get("panel") or {}).get("unanimous") and a.get("seq"):
                    out.append((a["seq"], r["target"], f"{tag}:{r['id']}"))
        except Exception as e:
            print(f"  (skip {tag}: {repr(e)[:60]})")
    return out


def main():
    designs = collect_unanimous()
    print(f"panel-unanimous certified designs collected: {len(designs)}", flush=True)
    res, gaps = [], []
    for seq, target, src in designs:
        try:
            g = gap_for(seq, target)
        except Exception as e:
            g = {"error": repr(e)[:80]}
        if g and "gap" in g:
            gaps.append(g["gap"])
            res.append({"src": src, **g})
            print(f"  {src:16} MFE={g['emfe']:.2f} target_is_MFE={g['mfe_is_target']} gap={g['gap']} kcal", flush=True)
    gaps.sort()
    n = len(gaps)
    summary = {"n_certified": len(designs), "n_scored": n,
               "target_is_MFE_all": all(r["mfe_is_target"] for r in res) if res else None,
               "mean_gap": round(sum(gaps) / n, 2) if n else None,
               "median_gap": gaps[n // 2] if n else None,
               "ge_1kcal": sum(g >= 1.0 for g in gaps), "ge_2kcal": sum(g >= 2.0 for g in gaps),
               "rows": res}
    json.dump(summary, open(os.path.join(HERE, "positive_stringency.json"), "w"), indent=1)
    print(f"\nSTRINGENCY: {n} certified designs; target is the pKiss MFE for "
          f"{sum(r['mfe_is_target'] for r in res)}/{n}; mean gap {summary['mean_gap']} kcal, "
          f"median {summary['median_gap']}; gap>=1 kcal: {summary['ge_1kcal']}/{n}; "
          f">=2 kcal: {summary['ge_2kcal']}/{n}.", flush=True)
    print("-> positive_stringency.json", flush=True)


if __name__ == "__main__":
    main()
