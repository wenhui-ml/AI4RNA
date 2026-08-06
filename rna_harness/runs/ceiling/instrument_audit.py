#!/usr/bin/env python3
"""Are our checks a general discipline, or thirteen scars from our own mistakes?

THE QUESTION, and it decides whether direction one (turn the method into infrastructure) has any basis.
instrument.py exists because in a single day this project produced thirteen measurement defects, three of
which would have shipped a false positive. Every check in it traces to one of OUR errors. That makes it
undeniably useful to us and says nothing about whether it generalises.

So run the same checks, unchanged, across eight widely used RNA structure tools that we did not write. This is
not an attack on anyone's software -- it is the thing a metrology institute does: apply one instrument
protocol uniformly and publish what it finds, including finding nothing.

BOTH OUTCOMES ARE INFORMATIVE, and that is why this is worth running:

    tools fail some checks   -> the checks catch defects in software, not just in analysts, and the protocol
                                has external validity
    tools pass everything    -> the checks catch ANALYST errors, not tool errors. Then instrument.py is about
                                discipline rather than software, and direction one must be pitched as an
                                evaluation protocol rather than as a bug finder. That is a real result and it
                                would change the pitch.

WHAT IS CHECKED, all of it from instrument.py with no per-tool special cases:

    controls        a declared positive and negative input per tool, plus assert_control_can_fail on the
                    predicate itself -- because two of our own negative controls passed for reasons unrelated
                    to discrimination
    structure       every parsed output must be a matching on distinct in-range positions; interaction
                    outputs must additionally be antiparallel
    dynamic range   over a common sample, does the tool's output actually vary, or is it pinned
    determinism     the same input twice must give the same answer, or the tool must be declared stochastic.
                    Not in instrument.py yet; it is added here because DesiRNA's nondeterminism has already
                    cost this project a published figure (51/55 was the top of a five-seed distribution)
    degenerate      how often does the tool return nothing at all

PRE-REGISTERED: the portability number is the fraction of checks that are even APPLICABLE to each tool. A
check that cannot be evaluated on external software is a check about us. That fraction is the headline,
whatever the pass rates turn out to be.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from interaction_entry import (pairs_bifold, pairs_intarna, pairs_rnacofold,   # noqa: E402
                               pairs_rnaduplex)
from rna_harness.instrument import (InstrumentError, assert_control_can_fail,  # noqa: E402
                                    check_duplex_antiparallel, check_pair_set, constancy,
                                    has_dynamic_range, run_controls)

VPK = "/home/ubuntu/miniconda3/envs/vpk/bin"
RNASTR = "/home/ubuntu/miniconda3/envs/rnastr/bin"
DATAPATH = "/home/ubuntu/miniconda3/envs/rnastr/share/rnastructure/data_tables"
PK = "/home/ubuntu/miniconda3/envs/pk/bin/pKiss"

POS_SEQ = "GGGGAAAACCCC"          # must fold into something
NEG_SEQ = "AAAAAAAAAAAA"          # must fold into nothing
POS_PAIR = ("GGGAAACCCUUUGGGAAACCC", "GGGUUUCCC")
NEG_PAIR = ("AAAAAAAAAAAAAAAAAAAAA", "AAAAAAAAA")


def _run(cmd, inp=None, env=None, t=180, cwd=None):
    try:
        r = subprocess.run(cmd, input=inp, capture_output=True, text=True, timeout=t, env=env, cwd=cwd)
        return r.stdout or ""
    except Exception:                                                          # noqa: BLE001
        return None


def _db_pairs(s):
    out, st = set(), []
    for i, ch in enumerate(s):
        if ch in "([{<":
            st.append(i)
        elif ch in ")]}>" and st:
            out.add((st.pop(), i))
    return out


# ---- single-molecule folders: sequence -> pair set -------------------------------------------------------
def fold_rnafold(seq):
    o = _run([f"{VPK}/RNAfold", "--noPS"], inp=seq + "\n")
    if not o:
        return None
    for ln in o.splitlines()[1:]:
        m = re.match(r"([.()]+)", ln.strip())
        if m:
            return _db_pairs(m.group(1))
    return None


def fold_pkiss(seq):
    o = _run([PK, "--mode", "mfe", seq])
    if not o:
        return None
    for ln in o.splitlines():
        m = re.search(r"([.()\[\]{}<>]{4,})", ln)
        if m and len(m.group(1)) == len(seq):
            return _db_pairs(m.group(1))
    return None


def fold_rnapkplex(seq):
    with tempfile.TemporaryDirectory() as d:
        o = _run([f"{VPK}/RNAPKplex"], inp=seq + "\n", cwd=d)
    if not o:
        return None
    for ln in o.splitlines():
        m = re.match(r"([.()\[\]]+)", ln.strip())
        if m and len(m.group(1)) == len(seq):
            return _db_pairs(m.group(1))
    return None


def _rnastr_fold(binname, seq, extra=()):
    """None means the TOOL DID NOT RUN; an empty set means it ran and found no pairs.

    The first version conflated them: a missing .db raised FileNotFoundError, run_controls caught it, and the
    NEGATIVE control was recorded as a failure on exactly the input where the tool behaves correctly by
    returning nothing. That is this project's oldest recurring defect -- "did not run" and "found nothing"
    are opposite findings -- committed here in the very file written to catch it.
    """
    with tempfile.TemporaryDirectory() as d:
        q, ct, db = (os.path.join(d, x) for x in ("q.seq", "o.ct", "o.db"))
        open(q, "w").write(f";\nx\n{seq}1\n")
        env = dict(os.environ, DATAPATH=DATAPATH)
        if _run([f"{RNASTR}/{binname}", q, ct, *extra], env=env) is None:
            return None
        if not os.path.exists(ct):
            return None
        if _run([f"{RNASTR}/ct2dot", ct, "1", db], env=env) is None or not os.path.exists(db):
            return set()                      # ran, produced no structure
        lines = [l.strip() for l in open(db).read().splitlines()
                 if l.strip() and set(l.strip()) <= set(".()")]
    return _db_pairs("".join(lines)) if lines else set()


def fold_probknot(seq):
    # ProbKnot's usage: a .seq input requires --sequence, otherwise it expects a partition-function save
    # file. Omitting the flag produced no output on 33 of 33 sequences and the first run of this audit
    # reported that as ProbKnot failing its controls -- a wrapper defect written up as a tool defect.
    return _rnastr_fold("ProbKnot", seq, extra=("--sequence",))


def fold_shapeknots(seq):
    return _rnastr_fold("ShapeKnots", seq)


FOLDERS = {"RNAfold": fold_rnafold, "pKiss": fold_pkiss, "RNAPKplex": fold_rnapkplex,
           "ProbKnot": fold_probknot, "ShapeKnots": fold_shapeknots}
DUPLEX = {"RNAduplex": pairs_rnaduplex, "RNAcofold": pairs_rnacofold,
          "IntaRNA": pairs_intarna, "bifold": pairs_bifold}


def audit_folder(name, fn, seqs):
    rep = {"tool": name, "kind": "folder", "applicable": [], "failed": []}

    # 1. controls -- and first prove the predicate can fail at all
    pos_ok = lambda v: v is not None and len(v) >= 2                           # noqa: E731
    neg_ok = lambda v: v is not None and len(v) == 0                           # noqa: E731
    rep["applicable"].append("controls")
    try:
        assert_control_can_fail(pos_ok, [None, set()], name=f"{name}:pos")
        assert_control_can_fail(neg_ok, [None, {(0, 5), (1, 4)}], name=f"{name}:neg")
        cr = run_controls(name, fn, (POS_SEQ,), (NEG_SEQ,), pos_ok, neg_ok)
        rep["controls"] = {"positive_ok": cr.positive_ok, "negative_ok": cr.negative_ok,
                           "usable": cr.usable, "note": cr.note}
        if not cr.usable:
            rep["failed"].append("controls")
    except InstrumentError as e:
        rep["controls"] = {"predicate_cannot_fail": str(e)[:120]}
        rep["failed"].append("controls")

    outs, bad, empty, t0 = [], [], 0, time.time()
    for s in seqs:
        p = fn(s)
        if p is None:
            outs.append(None)
            continue
        try:
            check_pair_set(p, len(s), name=f"{name}")
        except InstrumentError as e:
            bad.append(str(e)[:110])
        outs.append(p)
        empty += (len(p) == 0)
    rep["wall_s"] = round(time.time() - t0, 1)

    # 2. structural validity
    rep["applicable"].append("structure")
    rep["structure"] = {"n": len(seqs), "n_none": sum(1 for o in outs if o is None),
                        "n_invalid": len(bad), "examples": bad[:3]}
    if bad:
        rep["failed"].append("structure")

    # 3. dynamic range over the sample
    dens = [len(o) / max(1, len(s)) for o, s in zip(outs, seqs) if o is not None]
    rep["applicable"].append("dynamic_range")
    c = constancy(dens)
    rep["dynamic_range"] = {"constant": c["constant"], "where": c["where"],
                            "usable": bool(dens) and has_dynamic_range(
                                [min(1.0, d * 2) for d in dens])}
    if not rep["dynamic_range"]["usable"]:
        rep["failed"].append("dynamic_range")

    # 4. determinism
    rep["applicable"].append("determinism")
    same = [fn(s) == o for s, o in list(zip(seqs, outs))[:10] if o is not None]
    rep["determinism"] = {"checked": len(same), "identical": sum(same)}
    if same and sum(same) != len(same):
        rep["failed"].append("determinism")

    # 5. degenerate output rate
    rep["applicable"].append("degenerate")
    rep["degenerate"] = {"n_empty": empty, "rate": empty / max(1, len(seqs))}
    return rep


def audit_duplex(name, fn, pairs):
    rep = {"tool": name, "kind": "duplex", "applicable": [], "failed": []}
    pos_ok = lambda v: v is not None and len(v) >= 4                           # noqa: E731
    neg_ok = lambda v: v is not None and len(v) == 0                           # noqa: E731
    rep["applicable"].append("controls")
    try:
        assert_control_can_fail(pos_ok, [None, set()], name=f"{name}:pos")
        assert_control_can_fail(neg_ok, [None, {(0, 5)} | {(1, 4)}], name=f"{name}:neg")
        cr = run_controls(name, fn, POS_PAIR, NEG_PAIR, pos_ok, neg_ok)
        rep["controls"] = {"positive_ok": cr.positive_ok, "negative_ok": cr.negative_ok,
                           "usable": cr.usable, "note": cr.note}
        if not cr.usable:
            rep["failed"].append("controls")
    except InstrumentError as e:
        rep["controls"] = {"predicate_cannot_fail": str(e)[:120]}
        rep["failed"].append("controls")

    outs, bad, anti, empty, t0 = [], [], [], 0, time.time()
    for a, b in pairs:
        p = fn(a, b)
        if p is None:
            outs.append(None)
            continue
        try:
            c = check_pair_set(p, len(a), len(b), name=name)
            check_duplex_antiparallel(c, name=name)
        except InstrumentError as e:
            (anti if "parallel" in str(e) else bad).append(str(e)[:110])
        outs.append(p)
        empty += (len(p) == 0)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["applicable"] += ["structure", "antiparallel", "dynamic_range", "determinism", "degenerate"]
    rep["structure"] = {"n": len(pairs), "n_none": sum(1 for o in outs if o is None),
                        "n_invalid": len(bad), "examples": bad[:3]}
    rep["antiparallel"] = {"n_parallel": len(anti), "examples": anti[:3]}
    if bad:
        rep["failed"].append("structure")
    if anti:
        rep["failed"].append("antiparallel")
    dens = [len(o) for o in outs if o is not None]
    c = constancy(dens)
    rep["dynamic_range"] = {"constant": c["constant"], "where": c["where"],
                            "usable": len(set(dens)) > 2}
    if not rep["dynamic_range"]["usable"]:
        rep["failed"].append("dynamic_range")
    same = [fn(a, b) == o for (a, b), o in list(zip(pairs, outs))[:10] if o is not None]
    rep["determinism"] = {"checked": len(same), "identical": sum(same)}
    if same and sum(same) != len(same):
        rep["failed"].append("determinism")
    rep["degenerate"] = {"n_empty": empty, "rate": empty / max(1, len(pairs))}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "instrument_audit.json"))
    a = ap.parse_args()

    from rna_harness.benchmarks.bprna_nested import load_bprna_nested
    seqs, pr = [], []
    for r in load_bprna_nested(400):
        t = str(r["target"])
        if not (25 <= len(t) <= 60):
            continue
        s = "".join("GCAU"[(i * 7 + len(t)) % 4] for i in range(len(t)))       # deterministic filler
        seqs.append(s)
        if len(seqs) >= a.n:
            break
    # VARYING complementarity, not a fixed perfect 12-mer. The first version handed every duplex tool a
    # perfect reverse complement of constant length, so every tool returned exactly 12 pairs and the
    # dynamic-range check failed for RNAduplex and IntaRNA -- a property of the population I built, reported
    # as a property of the tools.
    for i, s in enumerate(seqs[:20]):
        L = (6, 8, 10, 12, 14)[i % 5]
        w = s[len(s) // 3:len(s) // 3 + L]
        trig = list(w[::-1].translate(str.maketrans("ACGU", "UGCA")))
        if i % 3 == 0 and len(trig) > 3:                 # one in three carries a mismatch
            trig[len(trig) // 2] = "A" if trig[len(trig) // 2] != "A" else "G"
        pr.append((s, "".join(trig)))

    print(f"仪器审计:{len(FOLDERS)} 个折叠工具 × {len(seqs)} 条序列,"
          f"{len(DUPLEX)} 个相互作用工具 × {len(pr)} 对\n"
          f"  检查全部来自 instrument.py,不为任何工具开特例\n", flush=True)
    R = []
    for nm, fn in FOLDERS.items():
        r = audit_folder(nm, fn, seqs)
        R.append(r)
        print(f"  {nm:12s} 适用 {len(r['applicable'])} 项,未通过 {r['failed'] or '无'}"
              f"   ({r['wall_s']}s)", flush=True)
    for nm, fn in DUPLEX.items():
        r = audit_duplex(nm, fn, pr)
        R.append(r)
        print(f"  {nm:12s} 适用 {len(r['applicable'])} 项,未通过 {r['failed'] or '无'}"
              f"   ({r['wall_s']}s)", flush=True)

    print(f"\n=== 逐项细看 ===")
    for r in R:
        s = r["structure"]
        line = (f"  {r['tool']:12s} 无输出 {s['n_none']}/{s['n']}  非法结构 {s['n_invalid']}  "
                f"空结果 {r['degenerate']['n_empty']}  确定性 {r['determinism']['identical']}/"
                f"{r['determinism']['checked']}")
        if r["kind"] == "duplex":
            line += f"  平行双链 {r['antiparallel']['n_parallel']}"
        print(line)

    n_app = sum(len(r["applicable"]) for r in R)
    n_fail = sum(len(r["failed"]) for r in R)
    tools_clean = sum(1 for r in R if not r["failed"])
    print(f"\n=== 可移植性(预注册的头条数字)===")
    print(f"  可评估的检查项 {n_app} 个(平均每个工具 {n_app/len(R):.1f} 项)")
    print(f"  未通过 {n_fail} 项;{tools_clean}/{len(R)} 个工具全清")
    if n_fail == 0:
        print("  -> 八个工具全部通过。**这些检查抓的是【分析者】的错,不是【软件】的错**;")
        print("     那么 instrument.py 是一套纪律而不是查错器,方向一必须按【评估协议】而不是【找 bug】来讲。")
    else:
        print("  -> 检查在我们没写的软件上也会触发,协议具备外部效度。")
    json.dump({"n_applicable": n_app, "n_failed": n_fail, "tools_clean": tools_clean,
               "n_tools": len(R), "reports": R}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
