#!/usr/bin/env python3
"""M63: does a corpus nobody designed carry the placement structure?

Declaration: m63_undesigned_placement_declaration.json, frozen, committed and pushed BEFORE this
file existed. Departures go in m63_declaration_erratum.json; the declaration is not edited.

The manuscript states this test twice and does not run it. The statistic is the paper's own:
within one molecule, the mean helix depth of its G-C pairs minus the mean helix depth of its A-U
pairs, with depth from Definition 2 and nothing else.

Four sources, three of them undesigned:
  V   the designed corpus (OK7a), recomputed per molecule so it is commensurate
  A1  Rfam seed alignments, consensus structure projected onto individual sequences
  A2  Pseudobase++, natural pseudoknotted RNAs, sequences and structures as deposited
  C   ViennaRNA inverse_fold on the corpus's OWN target structures: matched by construction,
      and the only control that separates "the designers chose it" from "optimising a folding
      model implies it"

COST: $0. Rfam.seed.gz downloaded once. Everything else offline, CPU only.
"""
from __future__ import annotations
import collections, csv, gzip, json, math, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, "/data/wenhui/openclaude_rna/rna_harness/tools")
import m16_rescue as M16

RFAM = "/data/wenhui/openknot_data/Rfam.seed.gz"
PKB = "/data/wenhui/RNAInvBench/rna_design_algorithms/antaRNA/Constraints/Pseudobase++"
OK7A = "/data/wenhui/openknot_data/Data/OK7a_M2R_data.v4.5.1.csv"
SEED, MAXSEQ, RESTARTS, MINPAIR, NBOOT = 20260901, 5, 8, 4, 2000
BUDGET_S = 90          # per target, wall clock; the declaration fixed restarts but not a bound
CANON = {"CG", "GC", "AU", "UA", "GU", "UG"}
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


# ------------------------------------------------------------------ the paper's own statistic
def pairs_from_db(db):
    """Definition 2 needs a pairing map. Multiple bracket families, as the manuscript's parser does."""
    OPEN, CLOSE = "([{<ABCD", ")]}>abcd"
    st = {c: [] for c in OPEN}; out = {}
    for i, c in enumerate(db):
        if c in OPEN:
            st[c].append(i)
        elif c in CLOSE:
            s = st[OPEN[CLOSE.index(c)]]
            if s:
                j = s.pop(); out[j] = i; out[i] = j
    return out


def depth_of(P, a, b):
    """Definition 2: min(outward stacking run, inward stacking run)."""
    m = 0
    while P.get(a - m - 1) == b + m + 1:
        m += 1
    n = 0
    while P.get(a + n + 1) == b - n - 1:
        n += 1
    return min(m, n)


def gap_of(seq, P):
    """Per-molecule placement gap. None if the molecule does not meet the admission rule."""
    cg, au = [], []
    for a, b in P.items():
        if a >= b:
            continue
        t = seq[a] + seq[b]
        if t not in CANON:
            continue
        d = depth_of(P, a, b)
        pt = "".join(sorted({seq[a], seq[b]}))
        if pt == "CG":
            cg.append(d)
        elif pt == "AU":
            au.append(d)
    if len(cg) < MINPAIR or len(au) < MINPAIR:
        return None
    return (float(np.mean(cg)) - float(np.mean(au)), len(cg), len(au),
            len([1 for a, b in P.items() if a < b]))


def summarise(name, rows, rng, cluster=None):
    """rows: list of (cluster_key, gap, ncg, nau, npair, gc, npaired). Cluster bootstrap."""
    if not rows:
        return {"name": name, "n": 0}
    g = np.array([r[1] for r in rows], float)
    keys = [r[0] for r in rows]
    by = collections.defaultdict(list)
    for k, v in zip(keys, g):
        by[k].append(v)
    ks = sorted(by)
    bs = []
    for _ in range(NBOOT):
        pick = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
        vals = np.concatenate([by[p] for p in pick])
        bs.append(vals.mean())
    bs = np.array(bs)
    return {"name": name, "n": len(rows), "n_clusters": len(ks),
            "gap": round(float(g.mean()), 4), "median": round(float(np.median(g)), 4),
            "sd": round(float(g.std(ddof=1)), 4),
            "ci": [round(float(np.quantile(bs, 0.025)), 4),
                   round(float(np.quantile(bs, 0.975)), 4)],
            "frac_negative": round(float((g < 0).mean()), 4)}


def show(d):
    if not d.get("n"):
        log(f"  {d['name']:<26} 无合格分子"); return
    log(f"  {d['name']:<26} n={d['n']:>6}  簇={d['n_clusters']:>5}  g={d['gap']:+.4f}"
        f"  [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  中位 {d['median']:+.4f}"
        f"  为负的分子占 {100*d['frac_negative']:.1f}%")


# ------------------------------------------------------------------ sources
def load_designed():
    """OK7a: one row per (puzzle, method) design, its sequence and its declared target."""
    csv.field_size_limit(10 ** 9)
    seen = {}
    with open(OK7A) as f:
        for d in csv.DictReader(f):
            k = (d["puzzle"], d["method"])
            if k in seen:
                continue
            ss = int(float(d["sub_start"])) if d["sub_start"] not in ("", "NA") else 1
            se = int(float(d["sub_end"])) if d["sub_end"] not in ("", "NA") else len(d["sequence"])
            seen[k] = (d["sequence"], d["target_structure"], ss, se)
    return seen


def designed_rows():
    """F1 of the erratum: the released target string is the same length as the sequence, so its
    pairing map is ALREADY in sequence coordinates. M16's sub_start offset applies to the mutation
    indices, which are given on the design window; it does not apply to the structure."""
    out = []
    for (pz, meth), (seq, db, ss, se) in load_designed().items():
        assert len(db) == len(seq), (len(db), len(seq))
        Q = pairs_from_db(db)
        g = gap_of(seq, Q)
        if g:
            sub = "".join(seq[i] for i in sorted(Q))
            gc = (sub.count("G") + sub.count("C")) / max(len(sub), 1)
            out.append((pz, g[0], g[1], g[2], g[3], gc, len(Q) // 2, meth, db, seq, (ss, se)))
    return out


def rfam_rows():
    """Consensus structure projected onto each individual seed sequence."""
    out = []
    fam = None; ss = None; seqs = []
    def flush():
        if not (fam and ss and seqs):
            return
        Pc = pairs_from_db(ss)
        for nm, aln in seqs[:MAXSEQ]:
            idx = [i for i, c in enumerate(aln) if c not in ".-~_"]
            pos = {c: k for k, c in enumerate(idx)}
            s = "".join(aln[i] for i in idx).upper().replace("T", "U")
            Q = {}
            for a, b in Pc.items():
                if a < b and a in pos and b in pos:
                    x, y = pos[a], pos[b]
                    if s[x] + s[y] in CANON:
                        Q[x] = y; Q[y] = x
            g = gap_of(s, Q)
            if g:
                gc = (s.count("G") + s.count("C")) / max(len(s), 1)
                out.append((fam, g[0], g[1], g[2], g[3], gc, len(Q) // 2))
    with gzip.open(RFAM, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#=GF AC"):
                fam = line.split()[-1]
            elif line.startswith("#=GC SS_cons"):
                ss = line.split(None, 2)[2]
            elif line.startswith("//"):
                flush(); fam = None; ss = None; seqs = []
            elif line and not line.startswith("#"):
                p = line.split(None, 1)
                if len(p) == 2:
                    seqs.append((p[0], p[1]))
    flush()
    return out


def pkb_rows():
    out = []
    with open(PKB) as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i + 2 < len(lines) + 1:
        if i >= len(lines) or not lines[i].startswith(">"):
            i += 1; continue
        nm = lines[i][1:]
        s = lines[i + 1].upper().replace("T", "U")
        db = lines[i + 2]
        i += 3
        if len(s) != len(db):
            continue
        P = pairs_from_db(db)
        g = gap_of(s, P)
        if g:
            gc = (s.count("G") + s.count("C")) / max(len(s), 1)
            out.append((nm, g[0], g[1], g[2], g[3], gc, len(P) // 2))
    return out


# ------------------------------------------------------------------ C: the matched control
def inverse_rows(des, rng):
    """ViennaRNA inverse_fold on the corpus's OWN target structures. No designer, no human."""
    import RNA
    out = []
    seen = {}
    for (pz, _g, _a, _b, _c, _gc, _np, meth, db, seq, win) in des:
        if pz in seen:
            continue
        ss, se = win
        lo, hi = ss - 1, se - 1               # the region the design programs were given
        Q = {a: b for a, b in pairs_from_db(db).items() if lo <= a <= hi and lo <= b <= hi}
        if len(Q) < 8:
            continue
        sub = "".join("(" if (k in Q and Q[k] > k) else (")" if k in Q else ".")
                      for k in range(lo, hi + 1))
        seen[pz] = sub
    import time
    n_try = n_hit = 0                          # F2/F3: bounded in Python, on the design window
    for pz, sub in seen.items():
        t0 = time.time()
        for r in range(RESTARTS):
            if time.time() - t0 > BUDGET_S:
                break
            start = "".join("ACGU"[int(x)] for x in rng.integers(0, 4, len(sub)))
            n_try += 1
            s, dist = RNA.inverse_fold(start, sub)
            if dist > 0:                      # only sequences that actually fold to the target
                continue
            n_hit += 1
            s = s.upper().replace("T", "U")
            P2 = pairs_from_db(sub)
            g = gap_of(s, P2)
            if g:
                gc = (s.count("G") + s.count("C")) / max(len(s), 1)
                out.append((pz, g[0], g[1], g[2], g[3], gc, len(P2) // 2))
    log(f"    反折叠:{len(seen)} 个目标结构,{n_try} 次尝试,{n_hit} 次折到目标,"
        f"{len(out)} 条合格分子")
    return out


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    OUT = {"seed": SEED}
    log("=== M63:没有设计者的语料,带不带 placement 结构 ===")
    log("  声明冻结在代码之前。Rfam 下载一次,其余全离线,$0。\n")

    log("【V】设计语料,按分子重算,使下面每个数都可比")
    des = designed_rows()
    dv = [(r[0], r[1]) + r[2:7] for r in des]
    V = summarise("OK7a designed", dv, rng)
    show(V)
    log(f"    (论文按格算的是 -0.3815;逐方法 -0.127 到 -0.712)")
    OUT["V_designed"] = V

    log("\n【A】没有设计者的语料")
    rf = rfam_rows()
    A1 = summarise("Rfam seed", rf, rng); show(A1)
    pk = pkb_rows()
    A2 = summarise("Pseudobase++", pk, rng); show(A2)
    OUT["A1_rfam"] = A1; OUT["A2_pseudobase"] = A2

    log("\n【C】同一批目标结构上的纯热力学反折叠(匹配对照)")
    iv = inverse_rows(des, rng)
    C = summarise("RNAinverse (same targets)", iv, rng); show(C)
    OUT["C_inverse_fold"] = C

    log("\n【B】把长度与 GC 限制到设计语料自己的范围之后")
    npl = [r[6] for r in dv]; gcl = [r[5] for r in dv]
    plo, phi = min(npl), max(npl); glo, ghi = min(gcl), max(gcl)
    log(f"    设计语料的配对数范围 [{plo}, {phi}],GC 范围 [{glo:.3f}, {ghi:.3f}]")
    keep = lambda rows: [r for r in rows if plo <= r[6] <= phi and glo <= r[5] <= ghi]
    B1 = summarise("Rfam, matched", keep(rf), rng); show(B1)
    B2 = summarise("Pseudobase++, matched", keep(pk), rng); show(B2)
    OUT["B1_rfam_matched"] = B1; OUT["B2_pkb_matched"] = B2
    OUT["matching_window"] = {"pairs": [plo, phi], "gc": [round(glo, 4), round(ghi, 4)]}

    # ---------------------------------------------------------------- scoring
    log("\n=== 预注册记分(按冻结声明原文)===")
    gd = V.get("gap"); gr = A1.get("gap"); gp = A2.get("gap"); gi = C.get("gap")
    S = {}
    S["S1"] = (gd is not None and gd < 0, f"设计语料 g={gd}")
    S["S2"] = (gr is not None and abs(gr) < abs(gd), f"|Rfam| {abs(gr):.4f} vs |设计| {abs(gd):.4f}")
    ratio = abs(gd) / abs(gr) if (gr and abs(gr) > 1e-9) else float("inf")
    S["S3"] = (ratio >= 1.5, f"比值 {ratio:.2f}")
    S["S4"] = (gr is not None and gp is not None and (gr > 0) == (gp > 0),
               f"Rfam {gr:+.4f} / Pseudobase {gp:+.4f}")
    S["S5"] = (B1.get("gap") is not None and abs(B1["gap"]) < abs(gd),
               f"匹配后 |Rfam| {abs(B1.get('gap', 0)):.4f} vs |设计| {abs(gd):.4f}")
    S["S6"] = ((gi is not None and abs(gi) < abs(gd)),
               (f"|反折叠| {abs(gi):.4f} vs |设计| {abs(gd):.4f}" if gi is not None
                else "反折叠臂为空,按失败计"))
    S["S7"] = ((gi is not None and gi < 0),
               (f"反折叠 g={gi}" if gi is not None else "反折叠臂为空,按失败计"))
    npass = 0
    for k in sorted(S):
        ok, txt = S[k]; npass += ok
        log(f"  {k}  {'成立' if ok else '**失败**'}  {txt}")

    kills = []
    if (gr is not None and gd is not None and (gr < 0) == (gd < 0)
            and abs(gr) >= 0.5 * abs(gd) and A1["ci"][0] * A1["ci"][1] > 0):
        kills.append("K1_placement_is_not_a_designer_choice")
    if gi is not None and abs(gi) >= 0.5 * abs(gd):
        kills.append("K2_thermodynamics_not_designers")
    if (gr is not None and (abs(gr) < 0.5 * abs(gd) or (gr > 0) != (gd > 0))):
        kills.append("K3_the_prediction_holds")
    for nm, d in (("Rfam", A1), ("Pseudobase++", A2)):
        if d.get("n", 0) < 200:
            kills.append(f"K4_underpowered[{nm}]")
    OUT["scoring"] = {k: {"pass": bool(v[0]), "note": v[1]} for k, v in S.items()}
    OUT["kill_conditions_fired"] = kills
    OUT["n_pass"] = npass
    log(f"\n  裁定:{kills if kills else '无 kill condition 触发'}   {npass}/7 成立")

    json.dump(OUT, open(os.path.join(HERE, "m63_undesigned_placement.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m63_undesigned_placement.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写出 m63_undesigned_placement.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ------------------------------------------------------------------ C': the repaired control
def nested_core(db, win):
    """F4: the pairs in the primary bracket family, inside the design window. A nested folder can
    be asked for this; it cannot be asked for a pseudoknot."""
    ss, se = win
    lo, hi = ss - 1, se - 1
    st, Q = [], {}
    for i, c in enumerate(db):
        if not (lo <= i <= hi):
            continue
        if c == "(":
            st.append(i)
        elif c == ")" and st:
            j = st.pop(); Q[j] = i; Q[i] = j
    sub = "".join("(" if (k in Q and Q[k] > k) else (")" if k in Q else ".")
                  for k in range(lo, hi + 1))
    return Q, sub, lo


def core_arms(des, rng):
    """Inverse-folded sequences and the designed sequences, on the SAME nested-core pair set."""
    import RNA, time
    targets, dsg = {}, []
    for (pz, _g, _a, _b, _c, _gc, _np, meth, db, seq, win) in des:
        Q, sub, lo = nested_core(db, win)
        if len(Q) // 2 < 8:
            continue
        targets.setdefault(pz, (sub, lo))
        s = seq[lo:lo + len(sub)]
        P = pairs_from_db(sub)
        g = gap_of(s, P)
        if g:
            gc = (s.count("G") + s.count("C")) / max(len(s), 1)
            dsg.append((pz, g[0], g[1], g[2], g[3], gc, len(P) // 2))
    out = []
    n_try = n_hit = 0
    for pz, (sub, _lo) in targets.items():
        t0 = time.time()
        P = pairs_from_db(sub)
        for r in range(RESTARTS):
            if time.time() - t0 > BUDGET_S:
                break
            start = "".join("ACGU"[int(x)] for x in rng.integers(0, 4, len(sub)))
            n_try += 1
            s, dist = RNA.inverse_fold(start, sub)
            if dist > 0:
                continue
            n_hit += 1
            s = s.upper().replace("T", "U")
            g = gap_of(s, P)
            if g:
                gc = (s.count("G") + s.count("C")) / max(len(s), 1)
                out.append((pz, g[0], g[1], g[2], g[3], gc, len(P) // 2))
    log(f"    嵌套核心:{len(targets)} 个目标,{n_try} 次尝试,{n_hit} 次折到目标,"
        f"{len(out)} 条合格反折叠分子;同一批 pair 上的设计序列 {len(dsg)} 条")
    return out, dsg


def cprime():
    rng = np.random.default_rng(SEED)
    log("\n=== M63 C':修好的匹配对照(声明未改,缺陷记在 erratum)===")
    des = designed_rows()
    inv, dsg = core_arms(des, rng)
    A = summarise("designed, nested core", dsg, rng); show(A)
    B = summarise("RNAinverse, nested core", inv, rng); show(B)
    out = {"designed_core": A, "inverse_core": B}
    if A.get("gap") is not None and B.get("gap") is not None and abs(A["gap"]) > 1e-9:
        out["share_reproduced"] = round(B["gap"] / A["gap"], 4)
        log(f"  纯热力学反折叠复现了设计序列该差值的 {100*B['gap']/A['gap']:.1f}%")
    d = json.load(open(os.path.join(HERE, "m63_undesigned_placement.json")))
    d["Cprime_repaired_control"] = out
    d["Cprime_note"] = ("F4 of the erratum: arm C as declared cannot run, because every target in "
                        "this corpus is pseudoknotted. C' runs it on the nested core with the "
                        "designed sequences scored on the same pairs. S6 and S7 stay FAILED on the "
                        "empty arm; they are not regraded.")
    json.dump(d, open(os.path.join(HERE, "m63_undesigned_placement.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(HERE, "m63_cprime.log"), "w").write("\n".join(LOG) + "\n")
    log("\n写回 m63_undesigned_placement.json")
    return 0
